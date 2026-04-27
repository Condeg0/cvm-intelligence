"""Rule-based metric extraction from parsed CVM filing text blocks.

CVM DFP/ITR PDFs store financial data as text blocks in the format:
  ``{account_code}\\n{description}\\n{current_value}\\n{prior_value}``

This module locates the CONSOLIDATED financial statement pages (pages whose
header contains "DFs Consolidadas"), parses each row block, and extracts the
nine target metrics by matching row descriptions against Portuguese label
patterns.

Deterministic and never uses an LLM. All extracted values are in the scale
declared by the page header (e.g. "(Reais Mil)" → values are in R$ thousands).
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Target metric label patterns (Portuguese, case-insensitive)
# ---------------------------------------------------------------------------

METRIC_PATTERNS: dict[str, list[re.Pattern[str]]] = {
    "revenue": [
        re.compile(r"receita\s+de\s+venda", re.IGNORECASE),
        re.compile(r"receita\s+l[íi]quida", re.IGNORECASE),
        re.compile(r"receita\s+operacional\s+l[íi]quida", re.IGNORECASE),
        # Banks / financial institutions:
        re.compile(r"receitas?\s+da\s+intermedia[çc][ãa]o\s+financeira", re.IGNORECASE),
        re.compile(r"receitas?\s+de\s+intermedia[çc][ãa]o\s+financeira", re.IGNORECASE),
    ],
    "cogs": [
        re.compile(r"custo\s+dos?\s+bens", re.IGNORECASE),
        re.compile(r"custo\s+dos?\s+produtos", re.IGNORECASE),
        re.compile(r"custo\s+dos?\s+servi[çc]os", re.IGNORECASE),
        # Banks / financial institutions:
        re.compile(r"despesas?\s+da\s+intermedia[çc][ãa]o\s+financeira", re.IGNORECASE),
        re.compile(r"despesas?\s+de\s+intermedia[çc][ãa]o\s+financeira", re.IGNORECASE),
    ],
    "gross_profit": [
        re.compile(r"resultado\s+bruto", re.IGNORECASE),
        re.compile(r"lucro\s+bruto", re.IGNORECASE),
    ],
    "ebitda": [
        re.compile(r"\bebitda\b", re.IGNORECASE),
        re.compile(r"\blajida\b", re.IGNORECASE),
    ],
    "net_income": [
        # "Consolidado" may appear between the label parts (e.g. Petrobras)
        re.compile(r"lucro\s*/\s*preju[íi]zo\s+(?:consolidado\s+)?do\s+per[íi]odo", re.IGNORECASE),
        re.compile(r"lucro\s+l[íi]quido\s+(?:consolidado\s+)?do\s+per[íi]odo", re.IGNORECASE),
        re.compile(r"resultado\s+l[íi]quido\s+do\s+per[íi]odo", re.IGNORECASE),
        re.compile(r"lucro\s+\(?preju[íi]zo\)?\s+l[íi]quido", re.IGNORECASE),
    ],
    "total_assets": [
        re.compile(r"^ativo\s+total$", re.IGNORECASE),
    ],
    "total_equity": [
        re.compile(r"patrim[oô]nio\s+l[íi]quido\b", re.IGNORECASE),
    ],
    "net_debt": [
        re.compile(r"d[íi]vida\s+l[íi]quida", re.IGNORECASE),
    ],
    "operating_cash_flow": [
        re.compile(r"caixa\s+l[íi]quido\s+atividades\s+operacionais", re.IGNORECASE),
        re.compile(r"caixa\s+das?\s+atividades\s+operacionais", re.IGNORECASE),
        re.compile(r"atividades\s+operacionais\b", re.IGNORECASE),
    ],
}

# CVM account code prefixes that identify each statement section
_SECTION_CODE_PREFIX: dict[str, tuple[str, ...]] = {
    "balance_sheet_assets":  ("1",),        # Ativo
    "balance_sheet_equity":  ("2",),        # Passivo + PL
    "income_statement":      ("3",),        # DRE
    "cash_flow":             ("6",),        # DFC
}

# Pattern matching consolidated page headers
_CONSOLIDATED_PAGE_RE = re.compile(r"DFs\s+Consolidadas", re.IGNORECASE)

# Pattern to detect whether a string starts with a CVM account code
_ACCOUNT_CODE_RE = re.compile(r"^\d+(?:\.\d+)*\n")

# A value token looks like an optional sign + digits with . separators
_VALUE_TOKEN_RE = re.compile(r"^[+-]?[\d]+(?:\.[\d]{3})*(?:,[\d]+)?$")


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class ExtractedMetric:
    """A single extracted financial metric with provenance."""

    metric_name: str
    raw_value_str: str
    parsed_value: float | None
    source: str  # "block" or "table"
    page_number: int
    row_label: str
    confidence: float = 1.0


@dataclass
class ExtractionResult:
    """All metrics extracted from one filing."""

    filing_id: str
    metrics: list[ExtractedMetric] = field(default_factory=list)
    failures: list[dict] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Main extraction entry point
# ---------------------------------------------------------------------------

def extract_metrics(
    parsed_pdf: "ParsedPDF",
    filing_id: str,
) -> ExtractionResult:
    """Extract all target metrics from a parsed CVM filing PDF.

    Focuses exclusively on consolidated statement pages ("DFs Consolidadas")
    since the CSV ground truth uses consolidated figures.

    **ITR vs DFP value selection:**
    Annual DFP pages have two value columns (current year, prior year).
    ITR income-statement pages (account codes 3.xx, 4.xx) have four columns:
    current quarter, current YTD, prior quarter, prior YTD.  For ITR filings
    the second value (index 1) is the YTD accumulated figure, which is
    consistent with the DFP annual value and with how the CSV ground truth
    records the primary ÚLTIMO period. Balance-sheet and cash-flow pages
    always carry the current-period value in position 0.

    Args:
        parsed_pdf: :class:`~src.parsing.pdf_parser.ParsedPDF` from Phase 2.
        filing_id: Identifier for this filing (used in result provenance).

    Returns:
        :class:`ExtractionResult` with extracted metrics and a failure log.
    """
    from src.extraction.value_parser import normalize_cell, detect_scale  # noqa: PLC0415

    result = ExtractionResult(filing_id=filing_id)
    found_metrics: dict[str, ExtractedMetric] = {}
    is_itr = "_ITR_" in filing_id.upper()

    # Metrics from income-statement sections (accounts 3.xx, 4.xx) where ITR
    # pages have [quarterly, YTD, prior_quarterly, prior_YTD] layout.
    _INCOME_STMT_METRICS = frozenset({"revenue", "cogs", "gross_profit", "net_income"})

    # Group text blocks by page
    pages: dict[int, list] = {}
    for b in parsed_pdf.text_blocks:
        pages.setdefault(b.page_number, []).append(b)

    for page_num in sorted(pages.keys()):
        page_blocks = pages[page_num]
        section_label = parsed_pdf.page_section_map.get(page_num, "Unknown")

        # Only process financial table pages (not prose sections)
        if section_label not in ("Balanço Patrimonial", "DRE"):
            continue

        # Only process CONSOLIDATED statements — search all top-area blocks
        # (y0 < 100) because the "DFs Consolidadas" label sits at y≈60, while
        # the CVM doc banner at y≈15 would be returned by a simple "first block"
        # approach and does not contain the consolidation marker.
        if not _is_consolidated_page(page_blocks):
            continue

        # Determine scale from the unit header block (y ≈ 84)
        scale_text = _find_scale_block(page_blocks)
        scale = detect_scale(scale_text) if scale_text else 1.0

        # Parse the data rows from this page
        rows = _parse_page_rows(page_blocks)

        for code, description, values in rows:
            metric = _match_metric(description, code)
            if metric is None:
                continue
            if metric in found_metrics:
                continue  # take the first occurrence (top-level account)

            # For ITR income-statement rows with 4 value columns, use index 1
            # (current YTD) to match the CSV ÚLTIMO row with full-year-to-date
            # accumulation:
            #   Q2/Q3: [curr_qtr, curr_YTD, prior_qtr, prior_YTD] → index 1
            #   Q1:    [curr_qtr=YTD, prior_qtr=YTD]               → index 0
            #   DFP/BPA/BPP/DFC: always 2 values                   → index 0
            if is_itr and metric in _INCOME_STMT_METRICS and len(values) >= 4:
                value_str = values[1]
            else:
                value_str = values[0]

            parsed = normalize_cell(value_str)
            if parsed is None:
                result.failures.append({
                    "metric": metric,
                    "reason": "parse_error",
                    "raw": value_str,
                    "page": page_num,
                })
                continue

            # Values are stored at the PDF's declared scale; apply if non-trivial
            # (scale=1000 means "(Reais Mil)" which is the same scale as CSV, so
            # no multiplication — both PDF and CSV are already in R$ thousands)
            em = ExtractedMetric(
                metric_name=metric,
                raw_value_str=value_str,
                parsed_value=parsed,
                source="block",
                page_number=page_num,
                row_label=description,
            )
            found_metrics[metric] = em
            logger.debug(
                "Extracted %s = %s (page %d, scale=%.0f)",
                metric, parsed, page_num, scale,
            )

    result.metrics = list(found_metrics.values())

    # Log which metrics were not found
    for metric in METRIC_PATTERNS:
        if metric not in found_metrics:
            result.failures.append({
                "metric": metric,
                "reason": "row_not_matched",
                "filing_id": filing_id,
            })
            logger.debug("Metric not found: %s in %s", metric, filing_id)

    logger.info(
        "extract_metrics(%s): %d found, %d missing",
        filing_id, len(result.metrics), len([f for f in result.failures if f.get("reason") == "row_not_matched"]),
    )
    return result


# ---------------------------------------------------------------------------
# extract_from_table (for test compatibility and pdfplumber fallback)
# ---------------------------------------------------------------------------

def extract_from_table(
    table: list[list[str | None]],
    metric_name: str,
    patterns: list[re.Pattern[str]],
    page_number: int,
) -> ExtractedMetric | None:
    """Search a single pdfplumber-style table for a metric by label pattern.

    Matches row labels in the first column against the given patterns and
    returns the value from the first adjacent numeric column.

    Args:
        table: A 2-D list of cell strings (from pdfplumber).
        metric_name: Canonical metric name (e.g. ``"revenue"``).
        patterns: Compiled regex patterns for the metric's Portuguese labels.
        page_number: Page number where this table was found (for provenance).

    Returns:
        :class:`ExtractedMetric` if found, ``None`` otherwise.
    """
    from src.extraction.value_parser import normalize_cell  # noqa: PLC0415

    for row in table:
        if not row or not row[0]:
            continue
        label = str(row[0]).strip()
        if not any(p.search(label) for p in patterns):
            continue

        # Find the first numeric cell in the row (skip the label cell)
        for cell in row[1:]:
            if cell is None:
                continue
            cell_str = str(cell).strip()
            if not cell_str:
                continue
            parsed = normalize_cell(cell_str)
            if parsed is not None:
                return ExtractedMetric(
                    metric_name=metric_name,
                    raw_value_str=cell_str,
                    parsed_value=parsed,
                    source="table",
                    page_number=page_number,
                    row_label=label,
                )

    return None


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _is_consolidated_page(blocks: list) -> bool:
    """Return True if any top-area block (y0 < 100) marks this as a consolidated page.

    CVM structured pages have two header rows:
    - y≈15: the document banner ("DFP - Demonstrações …") — not useful here
    - y≈60: the statement label ("DFs Consolidadas / Balanço Patrimonial Ativo")

    We must scan *all* blocks in the header zone rather than stopping at the
    first one, because the doc banner always appears before the section label.
    """
    for b in blocks:
        if b.y0 < 100 and _CONSOLIDATED_PAGE_RE.search(b.text):
            return True
    return False


def _find_scale_block(blocks: list) -> str:
    """Find the scale indicator block (e.g. "(Reais Mil)") near the page top."""
    for b in blocks:
        if 75 < b.y0 < 115:
            text = b.text.strip()
            if text.startswith("(") or any(kw in text.lower() for kw in ("mil", "milh", "bilh", "reais", "r$")):
                return text
    return ""


def _parse_page_rows(blocks: list) -> list[tuple[str, str, list[str]]]:
    """Parse financial data rows from a CVM structured table page.

    Each data block has the format::

        {account_code}\\n{description}\\n{val1}\\n{val2}[\\n{val3}\\n{val4}]

    For annual DFPs: 2 values (current year, prior year).
    For ITR income statement pages: 4 values
        (current quarter, current YTD, prior quarter, prior YTD).
    For ITR balance sheet pages: 2 values (current quarter-end, prior year-end).

    When a description is long it wraps, putting code+description in one block
    and the values in a separate block at the same y-position.

    Returns:
        List of ``(code, description, values)`` tuples where ``values`` is a
        list of all numeric value strings found in the row (at least one).
    """
    # Only look at blocks below the column headers (y0 > 125)
    data_blocks = [b for b in blocks if b.y0 > 125]

    # Group blocks by rounded y position (same row = same y ±3)
    by_row: dict[int, list] = {}
    for b in data_blocks:
        row_key = round(b.y0 / 3) * 3  # group within 3pt vertical tolerance
        by_row.setdefault(row_key, []).append(b)

    rows: list[tuple[str, str, list[str]]] = []

    for _y, row_blocks in sorted(by_row.items()):
        # Sort left to right
        row_blocks.sort(key=lambda b: b.x0)

        combined_text = "\n".join(b.text.strip() for b in row_blocks if b.text.strip())
        parts = combined_text.split("\n")

        # Skip if no useful content
        if len(parts) < 2:
            continue

        # Check if first part looks like an account code
        if not _ACCOUNT_CODE_RE.match(combined_text):
            continue

        code = parts[0].strip()
        # Find where values start (parts that look like numeric values)
        value_start = len(parts)
        for i in range(1, len(parts)):
            candidate = parts[i].strip()
            if _VALUE_TOKEN_RE.match(candidate) or (
                candidate.startswith("-") and _VALUE_TOKEN_RE.match(candidate[1:])
            ) or (
                candidate.startswith("(") and candidate.endswith(")")
            ):
                value_start = i
                break

        description = " ".join(parts[1:value_start]).strip()
        values = [p.strip() for p in parts[value_start:] if p.strip()]

        if description and values:
            rows.append((code, description, values))

    return rows


def _match_metric(description: str, code: str) -> str | None:
    """Return the canonical metric name for a row description, or None."""
    # Prefer top-level accounts (short code, no dots or single level)
    # to avoid matching sub-accounts. E.g. "3.01" yes, "3.04.05.01" no.
    code_depth = code.count(".")
    if code_depth > 2:
        return None  # too deep a sub-account

    for metric_name, patterns in METRIC_PATTERNS.items():
        for pattern in patterns:
            if pattern.search(description):
                return metric_name

    return None
