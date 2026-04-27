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


def main() -> None:
    """Entry point for validation report."""
    args = parse_args()
    logger.info("Generating validation report (output=%s)", args.output)
    raise NotImplementedError("Implement after Phase 3 (extraction + validation) is done.")


if __name__ == "__main__":
    main()
