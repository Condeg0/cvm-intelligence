"""Section-aware paragraph chunking for CVM filing text.

Public entry point for Phase 2 batch processing is ``chunk_filing``, which
orchestrates parse → detect_sections → chunk_sections into a single call and
returns a JSON-serialisable list of chunk dicts ready to be written to disk.

Splits filing text at paragraph boundaries first, then at sentence boundaries
if a chunk would exceed the 384-token limit. Never uses fixed-size windows.

Only prose sections (Relatório da Administração, Notas Explicativas) are
chunked — structured financial table pages produce very short "table-context"
chunks and are handled separately by the extraction pipeline.

Token counting uses a lightweight whitespace heuristic (word count × 1.33)
that is accurate enough for chunking decisions without requiring a tokenizer.
"""

from __future__ import annotations

import logging
import re
import unicodedata
from dataclasses import dataclass, field

from src.parsing.section_detector import DetectedSection, SectionType

logger = logging.getLogger(__name__)

MAX_CHUNK_TOKENS: int = 384
MIN_CHUNK_TOKENS: int = 50
OVERLAP_SENTENCES: int = 1  # carry one sentence of context into the next chunk

# Sections whose text should be chunked for RAG
_PROSE_SECTIONS: frozenset[SectionType] = frozenset({
    SectionType.RELATORIO_ADMINISTRACAO,
    SectionType.NOTAS_EXPLICATIVAS,
})


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class Chunk:
    """A single text chunk with provenance metadata."""

    chunk_id: str
    filing_id: str
    section_type: SectionType
    text: str
    token_count: int
    chunk_index: int
    source_page_range: tuple[int, int] = field(default_factory=lambda: (0, 0))


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def chunk_sections(
    sections: list[DetectedSection],
    filing_id: str,
    max_tokens: int = MAX_CHUNK_TOKENS,
) -> list[Chunk]:
    """Produce section-aware chunks from detected sections.

    Only processes prose sections (RELATORIO_ADMINISTRACAO, NOTAS_EXPLICATIVAS).
    Structured financial table sections are not chunked for RAG — they are
    parsed separately by the extraction module.

    Splits first at paragraph boundaries (double newline), then at sentence
    boundaries when a paragraph exceeds ``max_tokens``. Chunks shorter than
    ``MIN_CHUNK_TOKENS`` are discarded (e.g. isolated page headers).

    Args:
        sections: Detected sections from :func:`section_detector.detect_sections`.
        filing_id: Unique identifier for the source filing (used in chunk IDs).
        max_tokens: Maximum number of tokens allowed per chunk.

    Returns:
        Ordered list of :class:`Chunk` objects ready for embedding.
    """
    all_chunks: list[Chunk] = []
    global_index = 0

    for section in sections:
        if section.section_type not in _PROSE_SECTIONS:
            continue

        paragraphs = _split_paragraphs(section.text)
        pending: list[str] = []
        pending_tokens = 0
        overlap_buffer: list[str] = []  # last sentence(s) for overlap

        def _flush(pending: list[str]) -> Chunk | None:
            nonlocal global_index
            text = " ".join(pending).strip()
            tc = estimate_token_count(text)
            if tc < MIN_CHUNK_TOKENS:
                return None
            chunk = Chunk(
                chunk_id=f"{filing_id}_{global_index:04d}",
                filing_id=filing_id,
                section_type=section.section_type,
                text=text,
                token_count=tc,
                chunk_index=global_index,
            )
            global_index += 1
            return chunk

        for para in paragraphs:
            para = para.strip()
            if not para:
                continue

            para_tokens = estimate_token_count(para)

            if para_tokens > max_tokens:
                # Flush pending first
                if pending:
                    c = _flush(pending)
                    if c:
                        all_chunks.append(c)
                    pending = []
                    pending_tokens = 0

                # Split paragraph at sentence boundaries
                sentences = split_into_sentences(para)
                sent_buf: list[str] = list(overlap_buffer)
                sent_buf_tokens = sum(estimate_token_count(s) for s in sent_buf)

                for sent in sentences:
                    sent_tokens = estimate_token_count(sent)
                    if sent_buf_tokens + sent_tokens > max_tokens and sent_buf:
                        c = _flush(sent_buf)
                        if c:
                            all_chunks.append(c)
                        # Keep overlap
                        sent_buf = sent_buf[-OVERLAP_SENTENCES:] if OVERLAP_SENTENCES else []
                        sent_buf_tokens = sum(estimate_token_count(s) for s in sent_buf)
                    sent_buf.append(sent)
                    sent_buf_tokens += sent_tokens

                # Flush remainder of paragraph into pending
                if sent_buf:
                    pending = sent_buf
                    pending_tokens = sent_buf_tokens
                    # Update overlap buffer
                    overlap_buffer = pending[-OVERLAP_SENTENCES:] if OVERLAP_SENTENCES else []

            elif pending_tokens + para_tokens > max_tokens:
                # Current paragraph would overflow — flush pending first
                c = _flush(pending)
                if c:
                    all_chunks.append(c)
                overlap_buffer = split_into_sentences(pending[-1])[-OVERLAP_SENTENCES:] if pending else []
                pending = list(overlap_buffer) + [para]
                pending_tokens = sum(estimate_token_count(s) for s in pending)
            else:
                pending.append(para)
                pending_tokens += para_tokens

        # Flush any remaining content
        if pending:
            c = _flush(pending)
            if c:
                all_chunks.append(c)

    logger.debug(
        "chunk_sections(%s): %d prose sections → %d chunks",
        filing_id, len([s for s in sections if s.section_type in _PROSE_SECTIONS]),
        len(all_chunks),
    )
    return all_chunks


# ---------------------------------------------------------------------------
# Token counting and text splitting
# ---------------------------------------------------------------------------

def estimate_token_count(text: str) -> int:
    """Estimate the token count of a text string.

    Uses a whitespace-split word count multiplied by 1.33 (average tokens/word
    for Portuguese text with BPE tokenizers). Sufficient for chunking decisions
    where exact count is not required.

    Args:
        text: Input text.

    Returns:
        Estimated token count as an integer.
    """
    word_count = len(text.split())
    return max(1, int(word_count * 1.33))


def split_into_sentences(text: str) -> list[str]:
    """Split a paragraph into individual sentences.

    Handles common Portuguese abbreviations to avoid false splits on strings
    like "S.A.", "R$", "No.", "Dr.", "Cia.", "pag.", "ref.", "p.ex.".

    Splits on:
    - Period/exclamation/question mark followed by whitespace and an uppercase
      letter or digit.
    - Portuguese abbreviations are protected by temporarily replacing the dot.

    Args:
        text: A single paragraph string.

    Returns:
        List of sentence strings, each stripped of surrounding whitespace.
        Empty results are filtered out.
    """
    # Protect common abbreviations from triggering sentence splits
    _ABBR_PATTERNS = [
        (re.compile(r"\b(S)\.(A)\.", re.IGNORECASE), r"\1<<DOT>>\2<<DOT>>"),
        (re.compile(r"\b(Cia)\.", re.IGNORECASE), r"\1<<DOT>>"),
        (re.compile(r"\b(Dr|Dra|Sr|Sra|Prof|No|Art|Pag|Ref|p\.ex)\.", re.IGNORECASE), r"\1<<DOT>>"),
        (re.compile(r"\bR\$\s*(\d)"), r"R<<DOT>>$ \1"),
        (re.compile(r"(\d+)\.(\d{3})"), r"\1<<THOU>>\2"),  # thousands separator
    ]

    protected = text
    for pattern, replacement in _ABBR_PATTERNS:
        protected = pattern.sub(replacement, protected)

    # Split at sentence-ending punctuation followed by space + uppercase/digit
    raw_sentences = re.split(r"(?<=[.!?])\s+(?=[A-ZÁÉÍÓÚÀÈÌÒÙÂÊÎÔÛÃÕÇ\d])", protected)

    # Restore protected dots and return
    result = []
    for sent in raw_sentences:
        restored = sent.replace("<<DOT>>", ".").replace("<<THOU>>", ".")
        cleaned = restored.strip()
        if cleaned:
            result.append(cleaned)

    return result if result else [text.strip()]


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def chunk_filing(
    pdf_path: "Path",
    filing_id: str,
    max_tokens: int = MAX_CHUNK_TOKENS,
) -> list[dict]:
    """Orchestrate parse → detect_sections → chunk_sections for one filing.

    Convenience entry point for batch processing scripts.  Handles all errors
    internally; returns an empty list if parsing fails.

    Args:
        pdf_path: Absolute path to the PDF file.
        filing_id: Unique identifier for the filing (used in chunk IDs).
        max_tokens: Maximum tokens per chunk (passed to ``chunk_sections``).

    Returns:
        List of JSON-serialisable chunk dicts with keys:
        ``chunk_id``, ``filing_id``, ``section_type``, ``text``,
        ``token_count``, ``chunk_index``.
    """
    from pathlib import Path as _Path  # noqa: PLC0415
    from src.parsing.pdf_parser import parse_pdf  # noqa: PLC0415
    from src.parsing.section_detector import detect_sections  # noqa: PLC0415

    pdf_path = _Path(pdf_path)
    parsed = parse_pdf(pdf_path)
    if parsed is None:
        logger.error("chunk_filing: parse_pdf returned None for %s", pdf_path.name)
        return []

    sections = detect_sections(parsed.full_text)
    if not sections:
        logger.warning("chunk_filing: no sections detected in %s", pdf_path.name)
        return []

    chunks = chunk_sections(sections, filing_id, max_tokens=max_tokens)
    return [
        {
            "chunk_id": c.chunk_id,
            "filing_id": c.filing_id,
            "section_type": c.section_type.value,
            "text": c.text,
            "token_count": c.token_count,
            "chunk_index": c.chunk_index,
        }
        for c in chunks
    ]


def _split_paragraphs(text: str) -> list[str]:
    """Split text into paragraphs at double-newline boundaries.

    Also treats CVM-specific artifacts like section markers and page
    separator lines as paragraph boundaries.

    Args:
        text: Section text from a :class:`DetectedSection`.

    Returns:
        List of paragraph strings.
    """
    # Remove embedded [SECTION:...] markers (they've already been processed)
    cleaned = re.sub(r"\[SECTION:[^\]]+\]", "", text)

    # Normalise various line-ending styles
    cleaned = cleaned.replace("\r\n", "\n").replace("\r", "\n")

    # Split on double (or more) newlines
    paragraphs = re.split(r"\n{2,}", cleaned)

    # Filter out single-line strings that look like page headers / numbers
    result = []
    for para in paragraphs:
        para = para.strip()
        if not para:
            continue
        # Skip isolated short lines (page numbers, watermarks like "PÚBLICA")
        if len(para) < 20 and not any(c.isalpha() for c in para[10:] if len(para) > 10):
            if para.isdigit() or para in ("PÚBLICA", "PÚBLICO", "CONFIDENCIAL"):
                continue
        result.append(para)

    return result
