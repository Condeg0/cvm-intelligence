"""SQLite schema definition and CRUD operations for CVM metrics database.

All four tables (companies, filings, metrics, chunks) are created here.
The ``init_db`` function is idempotent — safe to call on an existing database.
"""

from __future__ import annotations

import logging
import sqlite3
from pathlib import Path

from src import config

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# DDL
# ---------------------------------------------------------------------------
CREATE_COMPANIES = """
CREATE TABLE IF NOT EXISTS companies (
    ticker TEXT PRIMARY KEY,
    name   TEXT,
    sector TEXT
);
"""

CREATE_FILINGS = """
CREATE TABLE IF NOT EXISTS filings (
    filing_id         TEXT PRIMARY KEY,
    ticker            TEXT REFERENCES companies(ticker),
    filing_type       TEXT CHECK(filing_type IN ('ITR', 'DFP')),
    reference_date    DATE,
    pdf_path          TEXT,
    extraction_status TEXT CHECK(extraction_status IN ('success', 'partial', 'failed'))
);
"""

CREATE_METRICS = """
CREATE TABLE IF NOT EXISTS metrics (
    metric_id        INTEGER PRIMARY KEY AUTOINCREMENT,
    filing_id        TEXT REFERENCES filings(filing_id),
    metric_name      TEXT,
    extracted_value  REAL,
    validated_value  REAL,
    match_status     TEXT CHECK(match_status IN ('exact', 'close', 'mismatch', 'missing')),
    percentage_error REAL
);
"""

CREATE_CHUNKS = """
CREATE TABLE IF NOT EXISTS chunks (
    chunk_id        TEXT PRIMARY KEY,
    filing_id       TEXT REFERENCES filings(filing_id),
    section_name    TEXT,
    chunk_text      TEXT,
    sentiment_label TEXT,
    sentiment_score REAL,
    chromadb_id     TEXT
);
"""

_ALL_DDL = [CREATE_COMPANIES, CREATE_FILINGS, CREATE_METRICS, CREATE_CHUNKS]


# ---------------------------------------------------------------------------
# Initialisation
# ---------------------------------------------------------------------------

def init_db(db_path: Path = config.DB_PATH) -> None:
    """Create all tables if they do not exist.

    Idempotent — safe to call on an already-initialised database.

    Args:
        db_path: Path to the SQLite database file (created if absent).
    """
    db_path.parent.mkdir(parents=True, exist_ok=True)
    with sqlite3.connect(db_path) as conn:
        conn.execute("PRAGMA foreign_keys = ON;")
        for ddl in _ALL_DDL:
            conn.execute(ddl)
        conn.commit()
    logger.info("Database initialised at %s", db_path)


def get_connection(db_path: Path = config.DB_PATH) -> sqlite3.Connection:
    """Return a SQLite connection with foreign keys enabled.

    Callers are responsible for closing or using as a context manager.

    Args:
        db_path: Path to the SQLite database file.

    Returns:
        Open :class:`sqlite3.Connection`.
    """
    conn = sqlite3.connect(db_path)
    conn.execute("PRAGMA foreign_keys = ON;")
    conn.row_factory = sqlite3.Row
    return conn


# ---------------------------------------------------------------------------
# CRUD — companies
# ---------------------------------------------------------------------------

def upsert_company(ticker: str, name: str, sector: str, db_path: Path = config.DB_PATH) -> None:
    """Insert or replace a company record.

    Args:
        ticker: B3 ticker symbol (primary key).
        name: Company display name.
        sector: B3 sector classification.
        db_path: Database path.
    """
    with get_connection(db_path) as conn:
        conn.execute(
            "INSERT OR REPLACE INTO companies (ticker, name, sector) VALUES (?, ?, ?)",
            (ticker, name, sector),
        )
        conn.commit()


# ---------------------------------------------------------------------------
# CRUD — filings
# ---------------------------------------------------------------------------

def upsert_filing(
    filing_id: str,
    ticker: str,
    filing_type: str,
    reference_date: str,
    pdf_path: str,
    extraction_status: str,
    db_path: Path = config.DB_PATH,
) -> None:
    """Insert or replace a filing record.

    Args:
        filing_id: Unique filing identifier.
        ticker: Company ticker (must exist in companies table).
        filing_type: ``"ITR"`` or ``"DFP"``.
        reference_date: ISO date string (``YYYY-MM-DD``).
        pdf_path: Relative or absolute path to the PDF file.
        extraction_status: ``"success"``, ``"partial"``, or ``"failed"``.
        db_path: Database path.
    """
    with get_connection(db_path) as conn:
        conn.execute(
            """INSERT OR REPLACE INTO filings
               (filing_id, ticker, filing_type, reference_date, pdf_path, extraction_status)
               VALUES (?, ?, ?, ?, ?, ?)""",
            (filing_id, ticker, filing_type, reference_date, pdf_path, extraction_status),
        )
        conn.commit()


# Keep the old name as an alias for backwards compatibility with tests
insert_filing = upsert_filing


# ---------------------------------------------------------------------------
# CRUD — metrics
# ---------------------------------------------------------------------------

def insert_metric(
    filing_id: str,
    metric_name: str,
    extracted_value: float | None,
    validated_value: float | None,
    match_status: str,
    percentage_error: float | None,
    db_path: Path = config.DB_PATH,
) -> None:
    """Insert a metric extraction record.

    Args:
        filing_id: Parent filing identifier.
        metric_name: Canonical metric name (e.g. ``"revenue"``).
        extracted_value: Value extracted from the PDF.
        validated_value: Ground-truth value from the CSV.
        match_status: One of ``"exact"``, ``"close"``, ``"mismatch"``, ``"missing"``.
        percentage_error: Absolute percentage error (or ``None`` if unavailable).
        db_path: Database path.
    """
    with get_connection(db_path) as conn:
        conn.execute(
            """INSERT INTO metrics
               (filing_id, metric_name, extracted_value, validated_value,
                match_status, percentage_error)
               VALUES (?, ?, ?, ?, ?, ?)""",
            (filing_id, metric_name, extracted_value, validated_value,
             match_status, percentage_error),
        )
        conn.commit()


def delete_filing_metrics(filing_id: str, db_path: Path = config.DB_PATH) -> None:
    """Delete all metric rows for a filing (used before re-extraction).

    Args:
        filing_id: Filing whose metrics should be deleted.
        db_path: Database path.
    """
    with get_connection(db_path) as conn:
        conn.execute("DELETE FROM metrics WHERE filing_id = ?", (filing_id,))
        conn.commit()


# ---------------------------------------------------------------------------
# CRUD — chunks
# ---------------------------------------------------------------------------

def insert_chunk(
    chunk_id: str,
    filing_id: str,
    section_name: str,
    chunk_text: str,
    sentiment_label: str | None,
    sentiment_score: float | None,
    chromadb_id: str | None,
    db_path: Path = config.DB_PATH,
) -> None:
    """Insert a chunk record.

    Args:
        chunk_id: Unique chunk identifier.
        filing_id: Parent filing identifier.
        section_name: Section the chunk came from.
        chunk_text: Raw chunk text.
        sentiment_label: Predicted sentiment label (or ``None``).
        sentiment_score: Model confidence (or ``None``).
        chromadb_id: Corresponding ChromaDB document ID (or ``None``).
        db_path: Database path.
    """
    with get_connection(db_path) as conn:
        conn.execute(
            """INSERT OR IGNORE INTO chunks
               (chunk_id, filing_id, section_name, chunk_text,
                sentiment_label, sentiment_score, chromadb_id)
               VALUES (?, ?, ?, ?, ?, ?, ?)""",
            (chunk_id, filing_id, section_name, chunk_text,
             sentiment_label, sentiment_score, chromadb_id),
        )
        conn.commit()
