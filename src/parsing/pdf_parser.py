"""PyMuPDF-based text and table extraction from CVM filing PDFs.

Extracts text blocks with positional metadata using fitz (PyMuPDF) and
structured tables using pdfplumber. Errors are logged and never crash the
batch pipeline.

CVM filing PDFs have a consistent page structure:
- Page 1: Table of contents (Índice)
- Pages 2–20: Structured financial tables; each page starts with a bold header
  such as "DFs Individuais / Balanço Patrimonial Ativo"
- Later pages: Narrative prose (Comentário do Desempenho / Relatório da
  Administração) and Notas Explicativas (mixed prose + tables)
- Final pages: Auditor reports and director declarations

The first text block on each structured-table page is the section header. On
prose pages the header is absent or is a free-form title. This module classifies
each page and embeds ``[SECTION:xxx]`` markers into ``full_text`` so that
``section_detector.detect_sections`` can split sections without re-parsing.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from pathlib import Path

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------


@dataclass
class TextBlock:
    """A single text block with its bounding box on the page."""

    page_number: int
    text: str
    x0: float
    y0: float
    x1: float
    y1: float
    block_number: int


@dataclass
class ParsedPDF:
    """All extracted content from a single PDF filing."""

    pdf_path: Path
    text_blocks: list[TextBlock] = field(default_factory=list)
    tables: list[list[list[str | None]]] = field(default_factory=list)
    full_text: str = ""
    # page_number → SectionType.value string; populated by parse_pdf
    page_section_map: dict[int, str] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Text extraction (PyMuPDF)
# ---------------------------------------------------------------------------

def extract_text_blocks(
    pdf_path: Path,
    max_pages: int | None = None,
) -> list[TextBlock]:
    """Extract all text blocks with bounding boxes via PyMuPDF.

    Image blocks (type 1) are skipped. Empty blocks and isolated page-number
    artifacts (single short digit string near the bottom of the page) are
    filtered out.

    Args:
        pdf_path: Absolute path to the PDF file.
        max_pages: If given, stop after this many pages (useful for extraction
            where only structured financial table pages 1–25 are needed).

    Returns:
        List of :class:`TextBlock` objects ordered by page then vertical position.
    """
    import fitz  # PyMuPDF  # noqa: PLC0415

    doc = fitz.open(str(pdf_path))
    result: list[TextBlock] = []
    n_pages = len(doc) if max_pages is None else min(max_pages, len(doc))
    try:
        for page_idx in range(n_pages):
            page = doc[page_idx]
            # sort=True returns blocks in reading order (top-left to bottom-right)
            raw_blocks = page.get_text("blocks", sort=True)
            for block_num, b in enumerate(raw_blocks):
                # b = (x0, y0, x1, y1, text, block_no, block_type)
                if b[6] != 0:  # skip image blocks
                    continue
                text = b[4].strip()
                if not text:
                    continue
                # Drop isolated page-number artifacts: short numeric strings at
                # the very bottom of the page (y0 > 750 on a ~840pt page)
                if len(text) <= 3 and text.replace("de", "").strip().isdigit() and b[1] > 750:
                    continue
                result.append(TextBlock(
                    page_number=page_idx + 1,
                    text=text,
                    x0=b[0],
                    y0=b[1],
                    x1=b[2],
                    y1=b[3],
                    block_number=block_num,
                ))
    finally:
        doc.close()
    return result


# ---------------------------------------------------------------------------
# Table extraction (pdfplumber)
# ---------------------------------------------------------------------------

def extract_tables(pdf_path: Path) -> list[list[list[str | None]]]:
    """Extract structured tables from all pages via pdfplumber.

    Tables with only one row (typically just a header) are skipped.
    Extraction errors on individual pages are logged and skipped rather than
    crashing the whole document.

    Args:
        pdf_path: Absolute path to the PDF file.

    Returns:
        List of tables; each table is a list of rows; each row is a list of
        cell strings (or ``None`` for merged/empty cells).
    """
    import pdfplumber  # noqa: PLC0415

    tables: list[list[list[str | None]]] = []
    try:
        with pdfplumber.open(str(pdf_path)) as pdf:
            for page_num, page in enumerate(pdf.pages, start=1):
                try:
                    for table in page.extract_tables():
                        if table and len(table) > 1:
                            tables.append(table)
                except Exception as exc:
                    logger.debug(
                        "Table extraction failed on page %d of %s: %s",
                        page_num, pdf_path.name, exc,
                    )
    except Exception as exc:
        logger.warning(
            "pdfplumber could not open %s: %s — table extraction skipped",
            pdf_path.name, exc,
        )
    return tables


# ---------------------------------------------------------------------------
# Page header classification helper
# ---------------------------------------------------------------------------

_CVM_DOC_HEADER_RE = re.compile(
    r"^(ITR|DFP|DFI|FRE|DFPF|PISP)\s*[-–]\s*",
    re.IGNORECASE,
)

_METADATA_BLOCK_RE = re.compile(
    r"^(ITR|DFP|DFI|FRE|DFPF|PISP)\s*[-–]"  # document banner
    r"|^Vers[aã]o\s*:\s*\d+"                  # "Versão : 1"
    r"|^P[ÁA]GINA\s*:\s*\d+"                  # "PÁGINA: 20 de 101"
    r"|^P[ÚU]BLIC[AO]$"                        # "PÚBLICA"
    r"|^CONFIDENCIAL$"
    r"|^P[ÁA]G\.\s*\d+",
    re.IGNORECASE,
)


def _is_metadata_block(block: TextBlock) -> bool:
    """Return True if this block is a repeating page metadata artifact.

    Filters out the CVM document-type banner, version lines, page numbers,
    and watermarks ("PÚBLICA") that repeat on every prose page and add noise
    to the extracted text.
    """
    text = block.text.strip()
    return bool(_METADATA_BLOCK_RE.match(text))


def _is_section_header_block(block: TextBlock) -> bool:
    """Return True if this block is a repeating section-title header.

    On CVM prose pages, the section title (e.g. "Comentário do Desempenho",
    "Relatório da Administração") appears at a fixed y ≈ 42–75 on *every*
    page of the section as a running page-level label. We filter it out so it
    doesn't pollute the chunked content — it was already used for section
    classification.
    """
    from src.parsing.section_detector import classify_block, SectionType  # noqa: PLC0415

    if block.y0 > 80:
        return False  # only running headers appear near the top
    text = block.text.strip()
    return classify_block(text) != SectionType.UNKNOWN


def _get_page_header(blocks: list[TextBlock]) -> str:
    """Return the section-header block text for a page.

    CVM PDFs have two header layers:
    - y0 ≈ 15: Document-level metadata line ("ITR - Informações Trimestrais …")
    - y0 ≈ 35–75: Either a company-specific running header OR the section label

    Strategy:
    1. Collect all candidate blocks at y0 < 90, excluding CVM doc banners.
    2. Prefer a block that directly matches a known section pattern (e.g. Banco
       do Brasil pages put "Notas Explicativas" at y0=42 but their own company
       header at y0=35).
    3. Fall back to the first non-metadata block if no section match is found.
    """
    from src.parsing.section_detector import classify_block, SectionType  # noqa: PLC0415

    candidates = [
        b for b in sorted(blocks, key=lambda b: b.y0)
        if b.y0 < 90 and len(b.text.strip()) >= 4
        and not _CVM_DOC_HEADER_RE.match(b.text.strip())
    ]

    # First pass: look for a block that classifies as a known section
    for b in candidates:
        if classify_block(b.text.strip()) != SectionType.UNKNOWN:
            return b.text.strip()

    # Second pass: return first non-metadata block (might be a company header)
    for b in candidates:
        text = b.text.strip()
        if text and text not in ("#interna", "#confidencial", "#publico"):
            return text

    return ""


# ---------------------------------------------------------------------------
# Main parse function
# ---------------------------------------------------------------------------

def parse_pdf(
    pdf_path: Path,
    max_pages: int | None = None,
    skip_tables: bool = False,
) -> ParsedPDF | None:
    """Extract text blocks and tables from a CVM filing PDF.

    Uses PyMuPDF for text and positional metadata, pdfplumber for tables.
    Also determines each page's section type and embeds ``[SECTION:xxx]``
    markers in ``full_text`` so downstream consumers can split by section
    without re-opening the file.

    Returns ``None`` and logs the error if extraction fails entirely.

    Args:
        pdf_path: Absolute path to the PDF file.
        max_pages: If given, only process this many pages. Set to ~25 for
            metric extraction (financial tables are in pages 1–25); leave
            ``None`` for full parsing needed by chunking.
        skip_tables: If ``True``, skip pdfplumber table extraction entirely
            (metric extraction does not use tables; this is the slow step on
            large PDFs).

    Returns:
        :class:`ParsedPDF` on success, ``None`` on unrecoverable error.
    """
    # Deferred import avoids circular dependency (section_detector → pdf_parser)
    from src.parsing.section_detector import SectionType, classify_block  # noqa: PLC0415

    try:
        text_blocks = extract_text_blocks(pdf_path, max_pages=max_pages)
        tables = [] if skip_tables else extract_tables(pdf_path)
    except Exception as exc:
        logger.error("Failed to extract content from %s: %s", pdf_path.name, exc)
        return None

    # Group blocks by page
    pages: dict[int, list[TextBlock]] = {}
    for b in text_blocks:
        pages.setdefault(b.page_number, []).append(b)

    page_section_map: dict[int, str] = {}
    text_parts: list[str] = []
    current_section = SectionType.UNKNOWN
    last_marked_section = None

    for page_num in sorted(pages.keys()):
        page_blocks = pages[page_num]

        # Determine this page's section from its header
        header = _get_page_header(page_blocks)
        section = classify_block(header) if header else SectionType.UNKNOWN

        # If unclassified, carry forward the running section (prose pages
        # after the first page of a section have no explicit header)
        if section == SectionType.UNKNOWN:
            section = current_section
        else:
            current_section = section

        page_section_map[page_num] = section.value

        # Insert a [SECTION:xxx] marker whenever the section changes
        if section != last_marked_section and section != SectionType.UNKNOWN:
            text_parts.append(f"\n\n[SECTION:{section.value}]\n\n")
            last_marked_section = section

        # Build page text: filter metadata artifacts and repeating section
        # title headers, then join blocks with \n\n so that downstream
        # paragraph splitting works correctly (each block = one paragraph).
        content_blocks = [
            b for b in page_blocks
            if not _is_metadata_block(b) and not _is_section_header_block(b)
        ]
        page_text = "\n\n".join(b.text for b in content_blocks)
        if page_text.strip():
            text_parts.append(page_text)

    full_text = "\n\n".join(text_parts)

    logger.debug(
        "Parsed %s: %d blocks, %d tables, %d pages classified",
        pdf_path.name, len(text_blocks), len(tables), len(page_section_map),
    )
    return ParsedPDF(
        pdf_path=pdf_path,
        text_blocks=text_blocks,
        tables=tables,
        full_text=full_text,
        page_section_map=page_section_map,
    )
