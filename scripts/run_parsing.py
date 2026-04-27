"""Batch PDF parsing and chunking pipeline.

Iterates over all PDFs listed in the manifest, runs the full
parse → section-detect → chunk pipeline on each, and writes one JSON file
per filing to data/processed/chunks/.

Usage:
    python scripts/run_parsing.py [--limit N] [--ticker TICKER] [--workers N]

Examples:
    # Parse all 682 PDFs (may take 2–4 hours)
    python scripts/run_parsing.py

    # Quick test: 5 Petrobras filings
    python scripts/run_parsing.py --ticker PETR4 --limit 5
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
from src.parsing.chunker import chunk_filing  # noqa: E402

logging.basicConfig(
    level=config.LOG_LEVEL,
    format="%(asctime)s %(levelname)s %(name)s — %(message)s",
)
logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Batch PDF parsing pipeline")
    parser.add_argument("--limit", type=int, default=None, help="Max PDFs to process")
    parser.add_argument("--ticker", type=str, default=None, help="Only this ticker")
    parser.add_argument("--workers", type=int, default=4, help="Parallel worker processes")
    parser.add_argument("--force", action="store_true", help="Re-parse already-processed files")
    return parser.parse_args()


def _process_record(record: dict, output_dir: Path, force: bool) -> tuple[str, int, str]:
    """Parse one filing and write chunks to disk.

    Returns:
        Tuple of (filing_id, chunk_count, status) where status is one of
        'ok', 'skipped', 'no_pdf', 'failed', 'empty'.
    """
    filing_id = record["filing_id"]
    pdf_path_str = record.get("pdf_path")

    if not pdf_path_str:
        return filing_id, 0, "no_pdf"

    pdf_path = Path(pdf_path_str)
    if not pdf_path.exists():
        return filing_id, 0, "no_pdf"

    out_path = output_dir / f"{filing_id}.json"
    if out_path.exists() and not force:
        # Count existing chunks without re-parsing
        try:
            existing = json.loads(out_path.read_text())
            return filing_id, len(existing), "skipped"
        except Exception:
            pass  # corrupted file → re-parse

    try:
        chunks = chunk_filing(pdf_path, filing_id)
    except Exception as exc:
        logger.error("Failed to parse %s: %s", filing_id, exc)
        return filing_id, 0, "failed"

    if not chunks:
        return filing_id, 0, "empty"

    out_path.write_text(json.dumps(chunks, ensure_ascii=False, indent=2))
    return filing_id, len(chunks), "ok"


def main() -> None:
    args = parse_args()
    manifest_path = config.RAW_DIR / "manifest.json"
    if not manifest_path.exists():
        logger.error("Manifest not found. Run: python scripts/run_pdfs.py first.")
        sys.exit(1)

    manifest = json.loads(manifest_path.read_text())

    # Filter to records that have a PDF on disk
    records = [r for r in manifest if r.get("pdf_path")]
    if args.ticker:
        ticker = args.ticker.upper()
        records = [r for r in records if r["ticker"] == ticker]
        logger.info("Filtered to ticker %s: %d filings", ticker, len(records))
    if args.limit:
        records = records[: args.limit]

    config.CHUNKS_DIR.mkdir(parents=True, exist_ok=True)
    total = len(records)
    logger.info("Processing %d PDFs with %d workers…", total, args.workers)

    counts = {"ok": 0, "skipped": 0, "no_pdf": 0, "failed": 0, "empty": 0}
    done = 0

    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        futures = {
            executor.submit(_process_record, r, config.CHUNKS_DIR, args.force): r["filing_id"]
            for r in records
        }
        for future in as_completed(futures):
            filing_id, n_chunks, status = future.result()
            counts[status] += 1
            done += 1
            if done % 20 == 0 or done == total:
                logger.info(
                    "Progress: %d/%d  ok=%d  skipped=%d  failed=%d  empty=%d",
                    done, total,
                    counts["ok"], counts["skipped"], counts["failed"], counts["empty"],
                )

    logger.info(
        "Done. %d parsed, %d skipped, %d failed, %d empty",
        counts["ok"], counts["skipped"], counts["failed"], counts["empty"],
    )


if __name__ == "__main__":
    main()
