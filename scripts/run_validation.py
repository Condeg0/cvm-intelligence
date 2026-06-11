"""Validation report generation.

Compares extracted PDF metrics against CSV ground truth for all filings in
the database and prints a summary report (exact match rate, MAPE, coverage,
failure taxonomy).

Usage:
    python scripts/run_validation.py [--output PATH]
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
_VENV_PYTHON = _PROJECT_ROOT / ".venv" / "bin" / "python"
_EXPECTED_PREFIX = str(_PROJECT_ROOT / ".venv")
if sys.prefix != _EXPECTED_PREFIX and _VENV_PYTHON.exists():
    os.execv(str(_VENV_PYTHON), [str(_VENV_PYTHON)] + sys.argv)

sys.path.insert(0, str(_PROJECT_ROOT))

from src import config  # noqa: E402

logging.basicConfig(level=config.LOG_LEVEL, format="%(asctime)s %(levelname)s %(name)s — %(message)s")
logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments."""
    parser = argparse.ArgumentParser(description="Validation report generation")
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Optional path to write the report CSV (default: print to stdout)",
    )
    return parser.parse_args()


def _section(title: str) -> None:
    """Print a section header."""
    print(f"\n{'='*60}")
    print(f"  {title}")
    print(f"{'='*60}")


def main() -> None:
    """Entry point for validation report."""
    args = parse_args()
    logger.info("Generating validation report (output=%s)", args.output)

    import sqlite3

    db_path = _PROJECT_ROOT / "data" / "cvm_metrics.db"
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row

    # ------------------------------------------------------------------
    # Top-line summary
    # ------------------------------------------------------------------
    totals = dict(
        conn.execute(
            "SELECT match_status, COUNT(*) FROM metrics GROUP BY match_status"
        ).fetchall()
    )
    total = sum(totals.values())
    exact = totals.get("exact", 0)
    close = totals.get("close", 0)
    mismatch = totals.get("mismatch", 0)
    missing = totals.get("missing", 0)

    exact_rate = exact / total * 100 if total else 0.0
    coverage = (total - missing) / total * 100 if total else 0.0

    mape_row = conn.execute(
        "SELECT AVG(percentage_error) FROM metrics WHERE match_status = 'close'"
    ).fetchone()
    mape = mape_row[0] if mape_row[0] is not None else 0.0

    _section("TOP-LINE VALIDATION SUMMARY")
    print(f"  Total metric rows  : {total:>7,}")
    print(f"  Exact match rate   : {exact_rate:>6.2f}%   (target ≥ 95.2%)")
    print(f"  Coverage           : {coverage:>6.2f}%   (target ≥ 98.5%)")
    print(f"  MAPE (close only)  : {mape:>6.2f}%   (target < 1.0%)")

    # ------------------------------------------------------------------
    # Failure taxonomy
    # ------------------------------------------------------------------
    _section("FAILURE TAXONOMY")
    print(f"  {'Status':<12} {'Count':>7} {'% of total':>12}")
    print(f"  {'-'*34}")
    for status in ("exact", "close", "mismatch", "missing"):
        count = totals.get(status, 0)
        pct = count / total * 100 if total else 0.0
        print(f"  {status:<12} {count:>7,} {pct:>11.2f}%")

    # ------------------------------------------------------------------
    # Per-metric exact match rate
    # ------------------------------------------------------------------
    _section("PER-METRIC EXACT MATCH RATE")
    metric_rows = conn.execute(
        """
        SELECT metric_name,
               COUNT(*) AS total,
               SUM(CASE WHEN match_status = 'exact' THEN 1 ELSE 0 END) AS exact_cnt
        FROM metrics
        GROUP BY metric_name
        ORDER BY metric_name
        """
    ).fetchall()
    print(f"  {'Metric':<28} {'Total':>7} {'Exact':>7} {'Rate':>8}")
    print(f"  {'-'*54}")
    for row in metric_rows:
        rate = row["exact_cnt"] / row["total"] * 100 if row["total"] else 0.0
        print(f"  {row['metric_name']:<28} {row['total']:>7,} {row['exact_cnt']:>7,} {rate:>7.2f}%")

    # ------------------------------------------------------------------
    # Per-company exact match rate — top 10 and bottom 10
    # ------------------------------------------------------------------
    company_rows = conn.execute(
        """
        SELECT f.ticker,
               COUNT(*) AS total,
               SUM(CASE WHEN m.match_status = 'exact' THEN 1 ELSE 0 END) AS exact_cnt
        FROM metrics m
        JOIN filings f ON m.filing_id = f.filing_id
        GROUP BY f.ticker
        HAVING total >= 5
        ORDER BY CAST(exact_cnt AS REAL) / total DESC
        """
    ).fetchall()

    def _print_company_table(rows: list) -> None:
        print(f"  {'Ticker':<12} {'Total':>7} {'Exact':>7} {'Rate':>8}")
        print(f"  {'-'*38}")
        for row in rows:
            rate = row["exact_cnt"] / row["total"] * 100 if row["total"] else 0.0
            print(f"  {row['ticker']:<12} {row['total']:>7,} {row['exact_cnt']:>7,} {rate:>7.2f}%")

    _section("TOP 10 COMPANIES BY EXACT MATCH RATE")
    _print_company_table(company_rows[:10])

    _section("BOTTOM 10 COMPANIES BY EXACT MATCH RATE")
    _print_company_table(company_rows[-10:])

    conn.close()
    logger.info("Validation report complete.")


if __name__ == "__main__":
    main()
