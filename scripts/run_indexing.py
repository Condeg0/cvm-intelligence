"""Batch embedding and ChromaDB indexing pipeline.

Reads parsed chunks from data/processed/chunks/, embeds them with the
fine-tuned sentence transformer, and stores them in ChromaDB with metadata.
Also builds the BM25 index, saves it to disk, and populates the SQLite
chunks table.

Usage:
    python scripts/run_indexing.py [--reindex]
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sqlite3
import sys
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
_VENV_PYTHON = _PROJECT_ROOT / ".venv" / "bin" / "python"
_EXPECTED_PREFIX = str(_PROJECT_ROOT / ".venv")
if sys.prefix != _EXPECTED_PREFIX and _VENV_PYTHON.exists():
    os.execv(str(_VENV_PYTHON), [str(_VENV_PYTHON)] + sys.argv)

sys.path.insert(0, str(_PROJECT_ROOT))

from src import config  # noqa: E402

logging.basicConfig(
    level=config.LOG_LEVEL,
    format="%(asctime)s %(levelname)s %(name)s — %(message)s",
)
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Lightweight proxy so indexer functions can work without importing Chunk
# ---------------------------------------------------------------------------

class _ChunkProxy:
    """Minimal duck-type for Chunk objects, built from a JSON dict."""

    __slots__ = ("chunk_id", "filing_id", "section_type", "text", "token_count", "chunk_index")

    def __init__(self, d: dict) -> None:
        self.chunk_id: str = d["chunk_id"]
        self.filing_id: str = d["filing_id"]
        self.text: str = d["text"]
        self.token_count: int = d.get("token_count", 0)
        self.chunk_index: int = d.get("chunk_index", 0)
        # Wrap the section string so .value works uniformly in the indexer
        raw_section = d.get("section_type", "")
        self.section_type = _StrValue(raw_section)


class _StrValue:
    """Wraps a string as an object with a ``.value`` attribute."""

    def __init__(self, value: str) -> None:
        self.value = value

    def __str__(self) -> str:
        return self.value


# ---------------------------------------------------------------------------
# I/O helpers
# ---------------------------------------------------------------------------

def load_all_chunks(chunks_dir: Path) -> list[_ChunkProxy]:
    """Load all chunk JSON files from *chunks_dir* and return proxy objects.

    Args:
        chunks_dir: Directory containing one JSON file per filing.

    Returns:
        Flat list of :class:`_ChunkProxy` objects across all filings.
    """
    all_chunks: list[_ChunkProxy] = []
    json_files = sorted(chunks_dir.glob("*.json"))
    for json_path in json_files:
        try:
            with open(json_path, encoding="utf-8") as f:
                records = json.load(f)
            all_chunks.extend(_ChunkProxy(r) for r in records)
        except Exception as exc:
            logger.warning("Skipping %s — %s", json_path.name, exc)

    logger.info(
        "Loaded %d chunks from %d JSON files in %s",
        len(all_chunks), len(json_files), chunks_dir,
    )
    return all_chunks


def populate_chunks_db(chunk_dicts: list[dict], db_path: Path) -> None:
    """Insert chunks into the SQLite ``chunks`` table.

    Clears existing rows and re-inserts from *chunk_dicts*.  Uses
    ``chromadb_id = chunk_id`` since that is how we store them in the
    vector store.

    Args:
        chunk_dicts: Raw dicts from the chunk JSON files (not proxy objects).
        db_path: Path to the SQLite database.
    """
    rows = [
        (
            d["chunk_id"],
            d["filing_id"],
            d.get("section_type", ""),
            d["text"],
            None,   # sentiment_label — not stored at index time
            None,   # sentiment_score
            d["chunk_id"],  # chromadb_id matches chunk_id
        )
        for d in chunk_dicts
    ]
    with sqlite3.connect(db_path) as conn:
        conn.execute("PRAGMA foreign_keys = ON;")
        conn.execute("DELETE FROM chunks;")
        conn.executemany(
            """
            INSERT OR REPLACE INTO chunks
                (chunk_id, filing_id, section_name, chunk_text,
                 sentiment_label, sentiment_score, chromadb_id)
            VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
            rows,
        )
    logger.info("Populated SQLite chunks table with %d rows", len(rows))


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    """Parse CLI arguments."""
    parser = argparse.ArgumentParser(description="Batch embedding and indexing pipeline")
    parser.add_argument(
        "--reindex",
        action="store_true",
        help="Drop and recreate the ChromaDB collection before indexing",
    )
    return parser.parse_args()


def main() -> None:
    """Entry point for batch indexing."""
    args = parse_args()
    logger.info("Starting indexing pipeline (reindex=%s)", args.reindex)

    # 1. Load all chunks from disk
    chunk_proxies = load_all_chunks(config.CHUNKS_DIR)
    if not chunk_proxies:
        logger.error("No chunks found in %s — aborting.", config.CHUNKS_DIR)
        sys.exit(1)

    # 2. Locate the fine-tuned sentence transformer
    model_dir = config.MODELS_DIR / "sentence_transformer"
    if not model_dir.exists():
        logger.warning(
            "Fine-tuned model not found at %s — falling back to base BERTimbau.", model_dir
        )
        model_dir = None  # type: ignore[assignment]

    # 3. Build ChromaDB index
    from src.rag.indexer import (
        build_bm25_index,
        build_chroma_index,
        save_bm25_index,
    )

    build_chroma_index(
        chunks=chunk_proxies,
        chroma_dir=config.CHROMADB_DIR,
        collection_name=config.CHROMA_COLLECTION_NAME,
        model_dir=model_dir,
        batch_size=config.EMBEDDING_BATCH_SIZE,
        reindex=args.reindex,
    )

    # 4. Build and persist BM25 index
    bm25_index = build_bm25_index(chunk_proxies)
    bm25_path = config.VECTORSTORE_DIR / "bm25_index.pkl"
    save_bm25_index(bm25_index, chunk_proxies, bm25_path)

    # 5. Populate SQLite chunks table from the raw dicts
    raw_dicts: list[dict] = []
    for json_path in sorted(config.CHUNKS_DIR.glob("*.json")):
        try:
            with open(json_path, encoding="utf-8") as f:
                raw_dicts.extend(json.load(f))
        except Exception as exc:
            logger.warning("Skipping %s for DB population — %s", json_path.name, exc)

    populate_chunks_db(raw_dicts, config.DB_PATH)

    logger.info("Indexing pipeline complete.")


if __name__ == "__main__":
    main()
