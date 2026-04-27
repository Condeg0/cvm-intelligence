"""Batch metric extraction pipeline.

Iterates over all PDFs listed in the manifest, runs the full
parse → extract → validate → store pipeline on each, and writes results
to the SQLite database at data/cvm_metrics.db.

Usage:
    python scripts/run_extraction.py [--limit N] [--ticker TICKER] [--force]

Examples:
    # Quick test: 5 Petrobras filings
    python scripts/run_extraction.py --ticker PETR4 --limit 5

    # Full run (all 686 PDFs; may take 30–60 minutes)
    python scripts/run_extraction.py
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
_VENV_PYTHON = _PROJECT_ROOT / ".venv" / "bin" / "python"
_EXPECTED_PREFIX = str(_PROJECT_ROOT / ".venv")
if sys.prefix != _EXPECTED_PREFIX and _VENV_PYTHON.exists():
    os.execv(str(_VENV_PYTHON), [str(_VENV_PYTHON)] + sys.argv)

sys.path.insert(0, str(_PROJECT_ROOT))

from src import config  # noqa: E402
from src.db import schema  # noqa: E402
from src.extraction.metric_extractor import extract_metrics  # noqa: E402
from src.extraction.validator import (  # noqa: E402
    compute_report,
    load_ground_truth,
    validate_extraction,
)
from src.parsing.pdf_parser import parse_pdf  # noqa: E402

logging.basicConfig(
    level=config.LOG_LEVEL,
    format="%(asctime)s %(levelname)s %(name)s — %(message)s",
)
logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments."""
    parser = argparse.ArgumentParser(description="Batch metric extraction pipeline")
    parser.add_argument("--limit", type=int, default=None, help="Max PDFs to process")
    parser.add_argument("--ticker", type=str, default=None, help="Only this ticker")
    parser.add_argument("--workers", type=int, default=4, help="Parallel workers")
    parser.add_argument(
        "--force",
        action="store_true",
        help="Re-extract filings already in the database",
    )
    return parser.parse_args()


def _load_existing_filing_ids(db_path: Path) -> set[str]:
    """Return set of filing_ids already stored in the database."""
    if not db_path.exists():
        return set()
    try:
        conn = schema.get_connection(db_path)
        rows = conn.execute("SELECT filing_id FROM filings").fetchall()
        conn.close()
        return {r[0] for r in rows}
    except Exception:
        return set()


def _process_one(
    record: dict,
    csvs_dir: Path,
    resolved_companies: dict[str, dict],
    force: bool,
    existing_ids: set[str],
) -> dict:
    """Extract and validate metrics for one filing.

    Returns a status dict with keys: filing_id, status, metrics_found,
    match_rate, failures.
    """
    filing_id = record["filing_id"]
    ticker = record["ticker"]
    pdf_path_str = record.get("pdf_path")

    if not pdf_path_str or not Path(pdf_path_str).exists():
        return {"filing_id": filing_id, "status": "no_pdf", "metrics_found": 0}

    if not force and filing_id in existing_ids:
        return {"filing_id": filing_id, "status": "skipped", "metrics_found": 0}

    try:
        # Structured financial tables are always in pages 1–25; skip the
        # prose sections (Relatório, Notas Explicativas) and table extraction
        # to avoid spending 30+ minutes on 64MB bank PDFs.
        parsed = parse_pdf(Path(pdf_path_str), max_pages=30, skip_tables=True)
        result = extract_metrics(parsed, filing_id)
    except Exception as exc:
        logger.error("Parse/extract failed for %s: %s", filing_id, exc)
        return {"filing_id": filing_id, "status": "failed", "metrics_found": 0}

    # Load CSV ground truth
    try:
        gt = load_ground_truth(record, csvs_dir)
    except Exception as exc:
        logger.warning("Ground truth load failed for %s: %s", filing_id, exc)
        gt = {}

    # Validate
    records_val = validate_extraction(result, gt) if gt else []
    report = compute_report(records_val) if records_val else None

    # Determine extraction status
    n_found = len(result.metrics)
    n_missing = sum(1 for f in result.failures if f.get("reason") == "row_not_matched")
    if n_found == 0:
        extraction_status = "failed"
    elif n_missing > 2:  # ebitda + net_debt are always missing → allow 2
        extraction_status = "partial"
    else:
        extraction_status = "success"

    # Upsert company
    company = resolved_companies.get(ticker, {})
    schema.upsert_company(
        ticker=ticker,
        name=company.get("name", record.get("company_name", ticker)),
        sector=company.get("sector", ""),
    )

    # Upsert filing
    schema.upsert_filing(
        filing_id=filing_id,
        ticker=ticker,
        filing_type=record["filing_type"],
        reference_date=record.get("reference_date", ""),
        pdf_path=pdf_path_str,
        extraction_status=extraction_status,
    )

    # Delete old metrics if re-running
    if force and filing_id in existing_ids:
        schema.delete_filing_metrics(filing_id)

    # Insert metrics
    for val_rec in records_val:
        schema.insert_metric(
            filing_id=filing_id,
            metric_name=val_rec.metric_name,
            extracted_value=val_rec.extracted_value,
            validated_value=val_rec.ground_truth_value,
            match_status=val_rec.match_status,
            percentage_error=val_rec.percentage_error,
        )

    match_rate = report.exact_match_rate if report else None
    logger.debug(
        "%s: %s — %d metrics, match_rate=%.0f%%",
        filing_id,
        extraction_status,
        n_found,
        (match_rate or 0) * 100,
    )

    return {
        "filing_id": filing_id,
        "status": extraction_status,
        "metrics_found": n_found,
        "match_rate": match_rate,
        "n_validation_records": len(records_val),
    }


def main() -> None:
    """Run the full batch extraction pipeline."""
    args = parse_args()

    manifest_path = config.RAW_DIR / "manifest.json"
    if not manifest_path.exists():
        logger.error("Manifest not found: %s", manifest_path)
        sys.exit(1)

    manifest = json.loads(manifest_path.read_text())

    # Filter records
    records = [r for r in manifest if r.get("pdf_path")]
    if args.ticker:
        ticker = args.ticker.upper()
        records = [r for r in records if r["ticker"] == ticker]
        logger.info("Filtered to %s: %d filings", ticker, len(records))
    if args.limit:
        records = records[: args.limit]

    # Load resolved companies for sector info
    resolved_path = config.RAW_DIR / "resolved_companies.json"
    resolved_companies: dict[str, dict] = {}
    if resolved_path.exists():
        for co in json.loads(resolved_path.read_text()):
            resolved_companies[co["ticker"]] = co

    # Initialise database
    schema.init_db()
    existing_ids = _load_existing_filing_ids(config.DB_PATH)
    logger.info(
        "Processing %d filings (%d already in DB, force=%s)",
        len(records), len(existing_ids), args.force,
    )

    counts: dict[str, int] = {
        "success": 0, "partial": 0, "failed": 0, "skipped": 0, "no_pdf": 0,
    }
    total_match_rates: list[float] = []
    done = 0
    total = len(records)

    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        futures = {
            executor.submit(
                _process_one,
                r,
                config.CSVS_DIR,
                resolved_companies,
                args.force,
                existing_ids,
            ): r["filing_id"]
            for r in records
        }
        for future in as_completed(futures):
            res = future.result()
            status = res.get("status", "failed")
            counts[status] = counts.get(status, 0) + 1
            if res.get("match_rate") is not None:
                total_match_rates.append(res["match_rate"])
            done += 1
            if done % 20 == 0 or done == total:
                avg_match = (
                    sum(total_match_rates) / len(total_match_rates)
                    if total_match_rates else 0.0
                )
                logger.info(
                    "Progress: %d/%d  success=%d  partial=%d  failed=%d  "
                    "skipped=%d  avg_match=%.1f%%",
                    done, total,
                    counts["success"], counts["partial"], counts["failed"],
                    counts["skipped"], avg_match * 100,
                )

    avg_match = (
        sum(total_match_rates) / len(total_match_rates) if total_match_rates else 0.0
    )
    logger.info(
        "Done. success=%d  partial=%d  failed=%d  skipped=%d  no_pdf=%d  "
        "avg_match_rate=%.1f%%",
        counts["success"], counts["partial"], counts["failed"],
        counts["skipped"], counts["no_pdf"], avg_match * 100,
    )


if __name__ == "__main__":
    main()
