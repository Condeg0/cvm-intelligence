"""Batch PDF download from CVM RAD portal.

Downloads all PDFs referenced in the manifest. Must be run from a residential
or office internet connection — rad.cvm.gov.br blocks cloud/VPN IP ranges.

Usage:
    python scripts/run_pdfs.py [--limit N] [--ticker TICKER] [--workers N]

Examples:
    # Download all 686 PDFs (takes 1–3 hours depending on connection)
    python scripts/run_pdfs.py

    # Test with 5 Petrobras filings first
    python scripts/run_pdfs.py --ticker PETR4 --limit 5
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from pathlib import Path

# Re-exec with the project venv Python if not already running inside it
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
_VENV_PYTHON = _PROJECT_ROOT / ".venv" / "bin" / "python"
_EXPECTED_PREFIX = str(_PROJECT_ROOT / ".venv")
if sys.prefix != _EXPECTED_PREFIX and _VENV_PYTHON.exists():
    os.execv(str(_VENV_PYTHON), [str(_VENV_PYTHON)] + sys.argv)

sys.path.insert(0, str(_PROJECT_ROOT))

from src import config
from src.acquisition.download_pdfs import download_pdfs

logging.basicConfig(
    level=config.LOG_LEVEL,
    format="%(asctime)s %(levelname)s %(name)s — %(message)s",
)
logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Batch PDF download from CVM RAD portal")
    parser.add_argument("--limit", type=int, default=None, help="Max PDFs to download")
    parser.add_argument("--ticker", type=str, default=None, help="Only this ticker")
    parser.add_argument("--workers", type=int, default=4, help="Parallel download threads")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    manifest_path = config.RAW_DIR / "manifest.json"
    if not manifest_path.exists():
        logger.error("Manifest not found. Run: python -m src.acquisition.download_csvs first.")
        sys.exit(1)

    full_manifest = json.loads(manifest_path.read_text())

    # Determine the subset to download; always write the full manifest back.
    if args.ticker:
        ticker = args.ticker.upper()
        subset = [m for m in full_manifest if m["ticker"] == ticker]
        logger.info("Filtered to ticker %s: %d filings", ticker, len(subset))
    else:
        subset = full_manifest

    updated_subset = download_pdfs(subset, max_workers=args.workers, limit=args.limit)

    # Merge updated pdf_path values back into the full manifest
    updated_by_id = {m["filing_id"]: m for m in updated_subset}
    for record in full_manifest:
        if record["filing_id"] in updated_by_id:
            record["pdf_path"] = updated_by_id[record["filing_id"]].get("pdf_path")

    manifest_path.write_text(json.dumps(full_manifest, ensure_ascii=False, indent=2))
    on_disk = sum(1 for m in full_manifest if m.get("pdf_path"))
    logger.info("Done. %d/%d total PDFs on disk. Manifest updated.", on_disk, len(full_manifest))


if __name__ == "__main__":
    main()
