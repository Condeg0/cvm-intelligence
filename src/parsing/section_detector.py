"""Regex-based section identification in CVM filing text.

Identifies the four key sections: Balanço Patrimonial, DRE,
Relatório da Administração, and Notas Explicativas.

CVM filing PDFs have a very consistent page-header structure. The first text
block on each structured-table page (e.g. "DFs Individuais / Balanço
Patrimonial Ativo") is the section label. ``pdf_parser.parse_pdf`` embeds
``[SECTION:xxx]`` markers in ``full_text`` based on those page headers, so
``detect_sections`` finds them without rescanning the raw text.

Patterns are calibrated against the following manually examined filings:
  PETR4 DFP 2022, PETR4 ITR 2023, VALE3 ITR 2023, VALE3 DFP 2022,
  ITUB4 ITR 2023, BBAS3 ITR 2023, ABEV3 ITR 2023, BRAP4 ITR 2023
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from enum import Enum

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Section taxonomy
# ---------------------------------------------------------------------------

class SectionType(str, Enum):
    """Known section types in CVM filings."""

    BALANCO_PATRIMONIAL = "Balanço Patrimonial"
    DRE = "DRE"
    RELATORIO_ADMINISTRACAO = "Relatório da Administração"
    NOTAS_EXPLICATIVAS = "Notas Explicativas"
    UNKNOWN = "Unknown"


# ---------------------------------------------------------------------------
# Regex patterns for classifying a block of text (page header or free text)
# ---------------------------------------------------------------------------

#: Patterns used to classify an individual text block (e.g. a page header).
#: Ordered so that more specific patterns are checked first.
SECTION_PATTERNS: dict[SectionType, re.Pattern[str]] = {
    SectionType.BALANCO_PATRIMONIAL: re.compile(
        r"balan[çc]o\s+patrimonial",
        re.IGNORECASE,
    ),
    # Explicit negative lookahead: avoid matching "Demonstração do Resultado Abrangente"
    SectionType.DRE: re.compile(
        r"demonstra[çc][aã]o\s+do\s+resultado(?!\s+abrangente)"
        r"|resultado\s+do\s+exerc[ií]cio",
        re.IGNORECASE,
    ),
    SectionType.RELATORIO_ADMINISTRACAO: re.compile(
        r"relat[oó]rio\s+da\s+(?:administra[çc][aã]o|diretoria)"
        r"|coment[aá]rio[s]?\s+(?:de|do)\s+desempenho"
        r"|relat[oó]rio\s+de\s+desempenho"
        r"|desempenho\s+financeiro",
        re.IGNORECASE,
    ),
    SectionType.NOTAS_EXPLICATIVAS: re.compile(
        r"notas?\s+explicativas?",
        re.IGNORECASE,
    ),
}

#: Pattern used to detect embedded section markers inserted by ``pdf_parser``.
_SECTION_MARKER_RE: re.Pattern[str] = re.compile(
    r"\[SECTION:([^\]]+)\]"
)


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class DetectedSection:
    """A section detected in the filing with its character span."""

    section_type: SectionType
    start_char: int
    end_char: int
    text: str


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def classify_block(text: str) -> SectionType:
    """Classify a single text block into a section type using regex.

    Checks each section pattern in priority order and returns the first match.
    Returns ``SectionType.UNKNOWN`` if no pattern matches.

    Args:
        text: Text of the block to classify (e.g. a PDF page header).

    Returns:
        Best-matching :class:`SectionType`, or ``SectionType.UNKNOWN``.
    """
    for section_type, pattern in SECTION_PATTERNS.items():
        if pattern.search(text):
            return section_type
    return SectionType.UNKNOWN


def detect_sections(full_text: str) -> list[DetectedSection]:
    """Identify and extract named sections from filing full text.

    **Primary path**: If ``full_text`` was produced by ``pdf_parser.parse_pdf``
    it contains ``[SECTION:xxx]`` markers.  This function parses those markers
    to determine section boundaries, then merges consecutive runs of the same
    section type into a single :class:`DetectedSection`.

    **Fallback path**: If no markers are found, scans for regex patterns from
    ``SECTION_PATTERNS`` and takes the *first* occurrence of each section type.

    Args:
        full_text: Full text of the filing (possibly with embedded markers).

    Returns:
        List of :class:`DetectedSection` objects in document order.
        Consecutive pages of the same section are merged into one entry.
    """
    marker_matches = list(_SECTION_MARKER_RE.finditer(full_text))

    if marker_matches:
        sections = _sections_from_markers(full_text, marker_matches)
    else:
        logger.debug("No [SECTION:] markers found — falling back to regex scan")
        sections = _sections_from_regex(full_text)

    # Merge consecutive entries of the same section type
    merged = _merge_consecutive(sections)
    logger.debug(
        "detect_sections → %d section(s): %s",
        len(merged),
        [s.section_type.value for s in merged],
    )
    return merged


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _sections_from_markers(
    full_text: str,
    marker_matches: list[re.Match[str]],
) -> list[DetectedSection]:
    """Build DetectedSection list from embedded [SECTION:xxx] markers."""
    sections: list[DetectedSection] = []
    for i, m in enumerate(marker_matches):
        raw_name = m.group(1)
        # Resolve the string back to a SectionType enum value
        section_type = _name_to_section_type(raw_name)
        if section_type == SectionType.UNKNOWN:
            continue

        # Text runs from the end of this marker to the start of the next
        text_start = m.end()
        text_end = marker_matches[i + 1].start() if i + 1 < len(marker_matches) else len(full_text)
        text = full_text[text_start:text_end].strip()

        if not text:
            continue

        sections.append(DetectedSection(
            section_type=section_type,
            start_char=text_start,
            end_char=text_end,
            text=text,
        ))
    return sections


def _sections_from_regex(full_text: str) -> list[DetectedSection]:
    """Fallback: find sections by scanning full_text with SECTION_PATTERNS.

    Takes only the **first** occurrence of each section type to avoid
    false positives from table-of-contents repetitions.
    """
    first_matches: list[tuple[int, int, SectionType]] = []
    seen: set[SectionType] = set()

    for section_type, pattern in SECTION_PATTERNS.items():
        m = pattern.search(full_text)
        if m and section_type not in seen:
            first_matches.append((m.start(), m.end(), section_type))
            seen.add(section_type)

    first_matches.sort(key=lambda x: x[0])
    sections: list[DetectedSection] = []

    for i, (start, end, section_type) in enumerate(first_matches):
        next_start = first_matches[i + 1][0] if i + 1 < len(first_matches) else len(full_text)
        text = full_text[start:next_start].strip()
        if not text:
            continue
        sections.append(DetectedSection(
            section_type=section_type,
            start_char=start,
            end_char=next_start,
            text=text,
        ))

    return sections


def _name_to_section_type(name: str) -> SectionType:
    """Resolve a section name string back to a SectionType enum member."""
    for st in SectionType:
        if st.value == name:
            return st
    return SectionType.UNKNOWN


def _merge_consecutive(sections: list[DetectedSection]) -> list[DetectedSection]:
    """Merge consecutive DetectedSection entries that share the same type."""
    if not sections:
        return []

    merged: list[DetectedSection] = []
    current = sections[0]

    for nxt in sections[1:]:
        if nxt.section_type == current.section_type:
            # Extend the current section
            current = DetectedSection(
                section_type=current.section_type,
                start_char=current.start_char,
                end_char=nxt.end_char,
                text=current.text + "\n\n" + nxt.text,
            )
        else:
            merged.append(current)
            current = nxt

    merged.append(current)
    return merged
