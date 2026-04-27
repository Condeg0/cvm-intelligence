"""Brazilian number format parser.

Handles the Brazilian convention where `.` is the thousands separator and
`,` is the decimal separator. Example: ``1.234.567,89`` → ``1234567.89``.

All financial values must pass through this module before being stored or
compared against CSV ground truth.
"""

from __future__ import annotations

import logging
import re

logger = logging.getLogger(__name__)

# Matches optional sign, digits with . separators, and optional , decimal part
_BR_NUMBER_RE = re.compile(
    r"^(?P<sign>[+-])?\s*(?P<integer>[\d]{1,3}(?:\.[\d]{3})*|[\d]+)"
    r"(?:,(?P<decimal>[\d]+))?$"
)

# Values are often reported in thousands (R$ mil) or millions (R$ milhões)
SCALE_FACTORS: dict[str, float] = {
    "mil": 1_000.0,
    "milhões": 1_000_000.0,
    "bilhões": 1_000_000_000.0,
}

# Currency and unit prefixes to strip before parsing
_STRIP_RE = re.compile(
    r"R\$\s*"          # "R$ "
    r"|R\$"            # "R$"
    r"|US\$\s*"        # "US$ "
    r"|US\$"
    r"|\bR\b\s*"       # lone "R " (malformed)
    r"|mil\b.*$"       # " mil" suffix (handled by scale, not the number itself)
    ,
    re.IGNORECASE,
)


def parse_br_number(text: str) -> float | None:
    """Parse a Brazilian-formatted number string to a Python float.

    Handles:
    - Thousands separator: ``.`` (e.g. ``1.234.567``)
    - Decimal separator: ``,`` (e.g. ``1.234,56``)
    - Leading sign: ``+`` or ``-``

    Does NOT handle parenthesised negatives — use :func:`normalize_cell` for
    raw table cells.

    Args:
        text: Cleaned number string, e.g. ``"1.234.567,89"`` or ``"-45.678"``.

    Returns:
        Parsed float value, or ``None`` if the string cannot be parsed.
    """
    if not text:
        return None

    text = text.strip()
    m = _BR_NUMBER_RE.match(text)
    if not m:
        return None

    sign = -1.0 if m.group("sign") == "-" else 1.0
    integer_part = m.group("integer").replace(".", "")  # remove thousands dots
    decimal_part = m.group("decimal") or "0"

    try:
        value = float(f"{integer_part}.{decimal_part}")
    except ValueError:
        return None

    return sign * value


def normalize_cell(cell: str | None) -> float | None:
    """Normalize a raw table cell to a float financial value.

    Handles:
    - ``None`` / empty → ``None``
    - Dash variants (``-``, ``—``, ``–``) → ``0.0``
    - Parenthesised negatives: ``(1.234,56)`` → ``-1234.56``
    - Currency symbols: ``R$ 1.234,56`` → ``1234.56``
    - Trailing/leading whitespace
    - Zero variants: ``0``, ``-``, ``--``, ``—``

    Args:
        cell: Raw cell string from a pdfplumber table (may be ``None``).

    Returns:
        Normalized float, or ``None`` for empty / non-numeric cells.
    """
    if cell is None:
        return None

    text = str(cell).strip()
    if not text:
        return None

    # Dash variants represent zero or missing value; treat as 0.0
    if text in ("-", "--", "—", "–", "- -", "–––"):
        return 0.0

    # Parenthesised negative: "(1.234,56)" → -1234.56
    negative = False
    if text.startswith("(") and text.endswith(")"):
        text = text[1:-1].strip()
        negative = True
    elif text.startswith("-"):
        negative = True
        text = text[1:].strip()
    elif text.startswith("+"):
        text = text[1:].strip()

    # Strip currency prefixes
    text = _STRIP_RE.sub("", text).strip()

    # Remove any remaining whitespace that may have been embedded
    text = text.replace(" ", "")

    value = parse_br_number(text)
    if value is None:
        logger.debug("normalize_cell: could not parse %r", cell)
        return None

    return -value if negative else value


def detect_scale(header_text: str) -> float:
    """Detect the numeric scale (unit multiplier) from a table header string.

    CVM tables express values with different units. Common patterns:
    - ``"(em R$ mil)"`` or ``"(Reais Mil)"`` → multiply by 1,000
    - ``"(em R$ milhões)"`` → multiply by 1,000,000
    - ``"(em R$)"`` or no indicator → use as-is (multiply by 1)

    Args:
        header_text: Table header or caption text (e.g. first-row text).

    Returns:
        Multiplicative scale factor (1.0 if not detected).
    """
    lower = header_text.lower()
    if "bilh" in lower:
        return 1_000_000_000.0
    if "milh" in lower:
        return 1_000_000.0
    if "mil" in lower:
        return 1_000.0
    return 1.0
