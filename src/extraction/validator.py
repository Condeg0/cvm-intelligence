"""Validation of extracted PDF metrics against CSV ground truth.

Computes exact match rate, MAPE, and coverage. Classifies each comparison
result into one of four match statuses:
- ``exact``   — values equal within floating-point tolerance (1e-6)
- ``close``   — percentage error ≤ tolerance (default 1%)
- ``mismatch``— percentage error > tolerance
- ``missing`` — metric was not found in the PDF extraction
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any

import pandas as pd

from src.extraction.metric_extractor import ExtractionResult

logger = logging.getLogger(__name__)

# Floating-point equality threshold (values are large integers in R$ thousands)
_EXACT_ABS_TOLERANCE = 0.01


class FailureType(str, Enum):
    """Taxonomy of extraction failure reasons."""

    TABLE_NOT_FOUND = "table_not_found"
    WRONG_ROW = "wrong_row"
    VALUE_PARSING_ERROR = "value_parsing_error"
    WRONG_COLUMN = "wrong_column"
    SECTION_MISIDENTIFIED = "section_misidentified"
    UNKNOWN = "unknown"


@dataclass
class ValidationRecord:
    """Comparison of one extracted value against its CSV ground truth."""

    filing_id: str
    metric_name: str
    extracted_value: float | None
    ground_truth_value: float | None
    match_status: str  # "exact", "close", "mismatch", "missing"
    percentage_error: float | None
    failure_type: FailureType | None = None


@dataclass
class ValidationReport:
    """Aggregate validation statistics for a batch of filings."""

    records: list[ValidationRecord] = field(default_factory=list)
    exact_match_rate: float = 0.0
    mape: float = 0.0
    coverage: float = 0.0
    failure_counts: dict[str, int] = field(default_factory=dict)


def validate_extraction(
    extraction_result: ExtractionResult,
    ground_truth: dict[str, float],
    tolerance: float = 0.01,
) -> list[ValidationRecord]:
    """Compare extracted metrics to CSV ground truth.

    Args:
        extraction_result: Output of :func:`metric_extractor.extract_metrics`.
        ground_truth: Mapping of metric name → validated float value from CSV.
        tolerance: Relative tolerance for "close" match classification (default 1%).

    Returns:
        List of :class:`ValidationRecord` objects, one per target metric.
    """
    extracted_map = {m.metric_name: m.parsed_value for m in extraction_result.metrics}
    records: list[ValidationRecord] = []

    all_metrics = set(ground_truth) | set(extracted_map)

    for metric_name in sorted(all_metrics):
        extracted = extracted_map.get(metric_name)
        truth = ground_truth.get(metric_name)

        if truth is None:
            # No CSV ground truth for this metric — skip (not applicable)
            continue

        if extracted is None:
            records.append(ValidationRecord(
                filing_id=extraction_result.filing_id,
                metric_name=metric_name,
                extracted_value=None,
                ground_truth_value=truth,
                match_status="missing",
                percentage_error=None,
                failure_type=FailureType.UNKNOWN,
            ))
            continue

        # Compute relative error (guard against zero denominator)
        if abs(truth) < _EXACT_ABS_TOLERANCE:
            pct_error = 0.0 if abs(extracted) < _EXACT_ABS_TOLERANCE else None
        else:
            pct_error = abs(extracted - truth) / abs(truth)

        if pct_error is None:
            # Truth is zero but extracted is non-zero
            match_status = "mismatch"
        elif pct_error <= 1e-6:
            match_status = "exact"
        elif pct_error <= tolerance:
            match_status = "close"
        else:
            match_status = "mismatch"

        records.append(ValidationRecord(
            filing_id=extraction_result.filing_id,
            metric_name=metric_name,
            extracted_value=extracted,
            ground_truth_value=truth,
            match_status=match_status,
            percentage_error=pct_error,
        ))

    return records


def compute_report(records: list[ValidationRecord]) -> ValidationReport:
    """Aggregate per-record validation results into summary statistics.

    Args:
        records: List of :class:`ValidationRecord` from :func:`validate_extraction`.

    Returns:
        :class:`ValidationReport` with exact_match_rate, MAPE, and coverage.
    """
    report = ValidationReport(records=records)

    if not records:
        return report

    total = len(records)
    exact = sum(1 for r in records if r.match_status == "exact")
    close = sum(1 for r in records if r.match_status == "close")
    found = sum(1 for r in records if r.match_status != "missing")

    report.exact_match_rate = (exact + close) / total
    report.coverage = found / total

    # MAPE over records that have a valid percentage error and non-zero truth
    mape_errors = [
        r.percentage_error
        for r in records
        if r.percentage_error is not None and r.match_status != "missing"
    ]
    report.mape = (sum(mape_errors) / len(mape_errors)) if mape_errors else 0.0

    # Failure counts by match status
    for r in records:
        report.failure_counts[r.match_status] = (
            report.failure_counts.get(r.match_status, 0) + 1
        )

    logger.debug(
        "ValidationReport: exact+close=%.1f%% coverage=%.1f%% MAPE=%.4f",
        report.exact_match_rate * 100,
        report.coverage * 100,
        report.mape,
    )
    return report


# ---------------------------------------------------------------------------
# CSV ground-truth loader
# ---------------------------------------------------------------------------

# CVM account codes for each target metric (consolidated, current period only).
# Multiple codes are tried in order; the first non-empty match wins.
# Some sectors (banks) use different codes for the same economic concept.
_METRIC_ACCOUNT_CODES: dict[str, list[str]] = {
    "revenue":             ["3.01"],
    "cogs":                ["3.02"],
    "gross_profit":        ["3.03"],
    # Net income is the deepest single-segment Lucro/Prejuízo in the DRE.
    # Standard companies: 3.11; banks: 3.09 or 3.10; utilities: 3.10.
    "net_income":          ["3.11", "3.10", "3.09"],
    "total_assets":        ["1"],
    # Equity code varies across sectors — skip code-based lookup entirely and
    # rely on DS_CONTA description pattern (see _METRIC_DESC_PATTERNS below).
    # Non-banks: 2.03; banks: 2.08; we can't hardcode this reliably.
    "total_equity":        [],
    "operating_cash_flow": ["6.01"],
    # ebitda and net_debt are not standard CVM line items — skipped
}

# DS_CONTA patterns used as fallback when no account-code match is found
_METRIC_DESC_PATTERNS: dict[str, re.Pattern[str]] = {
    "net_income":  re.compile(
        r"lucro.{0,10}preju[íi]zo.{0,20}(per[íi]odo|exerc[íi]cio)|"
        r"lucro\s+l[íi]quido",
        re.IGNORECASE,
    ),
    "total_equity": re.compile(
        r"patrim[oô]nio\s+l[íi]quido",
        re.IGNORECASE,
    ),
}

# Maps metric → CVM statement abbreviation (file suffix)
_METRIC_CSV_FILE: dict[str, str] = {
    "revenue":             "DRE",
    "cogs":                "DRE",
    "gross_profit":        "DRE",
    "net_income":          "DRE",
    "total_assets":        "BPA",
    "total_equity":        "BPP",
    "operating_cash_flow": "DFC_MI",
}

_CSV_SEP = ";"
_CSV_ENC = "utf-8"


def load_ground_truth(
    manifest_record: dict[str, Any],
    csvs_dir: Path,
) -> dict[str, float]:
    """Load CSV ground-truth values for one filing.

    Reads the relevant CVM structured CSV files (DRE, BPA, BPP, DFC_MI) for
    the company/period in *manifest_record* and returns a metric → value dict.

    Only metrics present in :data:`_METRIC_ACCOUNT_CODES` are returned
    (``ebitda`` and ``net_debt`` are omitted — they are not CVM line items).

    Args:
        manifest_record: A single entry from ``manifest.json``.
        csvs_dir: Root directory containing ``itr_{year}/`` and ``dfp_{year}/``
            sub-directories with the filtered CSV files.

    Returns:
        Dict mapping canonical metric name → float value in R$ thousands
        (the same unit scale used by the PDFs).
    """
    cnpj = manifest_record["cnpj"]
    filing_type = manifest_record["filing_type"].lower()  # "itr" or "dfp"
    ref_date = manifest_record["reference_date"]          # "YYYY-MM-DD"
    year = ref_date[:4]
    csv_subdir = csvs_dir / f"{filing_type}_{year}"

    # ORDEM_EXERC label for the current period in CVM CSVs
    order_label = "ÚLTIMO"

    ground_truth: dict[str, float] = {}
    loaded_frames: dict[str, pd.DataFrame | None] = {}

    for metric, codes in _METRIC_ACCOUNT_CODES.items():
        file_key = _METRIC_CSV_FILE[metric]
        # file key → actual filename pattern
        filename = f"{filing_type}_cia_aberta_{file_key}_con_{year}.csv"
        csv_path = csv_subdir / filename

        if file_key not in loaded_frames:
            if csv_path.exists():
                try:
                    df = pd.read_csv(csv_path, sep=_CSV_SEP, encoding=_CSV_ENC, dtype=str)
                    df.columns = df.columns.str.strip()
                    # Normalise ORDEM_EXERC (strip accents in case of encoding drift)
                    if "ORDEM_EXERC" in df.columns:
                        df["_ordem_n"] = (
                            df["ORDEM_EXERC"]
                            .fillna("")
                            .str.upper()
                            .str.normalize("NFKD")
                            .str.encode("ascii", "ignore")
                            .str.decode("ascii")
                        )
                    loaded_frames[file_key] = df
                except Exception as exc:
                    logger.warning("Failed to read %s: %s", csv_path, exc)
                    loaded_frames[file_key] = None
            else:
                logger.debug("CSV not found: %s", csv_path)
                loaded_frames[file_key] = None

        df = loaded_frames.get(file_key)
        if df is None:
            continue

        # Filter to this company and current period
        mask = df["CNPJ_CIA"] == cnpj
        if "DT_REFER" in df.columns:
            mask &= df["DT_REFER"] == ref_date
        if "_ordem_n" in df.columns:
            mask &= df["_ordem_n"] == "ULTIMO"  # "ÚLTIMO" after accent-stripping → "ULTIMO"

        # Try codes one-by-one and pick the first non-empty match.
        # For metrics with multiple candidate codes (e.g. net_income: 3.11/3.09,
        # total_equity: 2.03/2.08) we take the first code that has data, which
        # is consistent with CVM standardisation within a single sector.
        rows = pd.DataFrame()
        for code in codes:
            candidate = df[mask & (df["CD_CONTA"] == code)]
            if not candidate.empty:
                rows = candidate
                break

        if rows.empty:
            # Fallback: search by DS_CONTA description pattern (handles banks
            # and other sectors with non-standard account codes)
            pattern = _METRIC_DESC_PATTERNS.get(metric)
            if pattern is not None and "DS_CONTA" in df.columns:
                # Only look at top-level accounts (code depth ≤ 1)
                def _code_depth(s: str) -> int:
                    return s.count(".")

                desc_mask = df[mask]["DS_CONTA"].str.contains(
                    pattern, regex=True, na=False
                )
                candidates = df[mask][desc_mask].copy()
                # Limit depth to avoid sub-accounts
                if "CD_CONTA" in candidates.columns:
                    candidates = candidates[
                        candidates["CD_CONTA"].apply(_code_depth) <= 2
                    ]
                if not candidates.empty:
                    rows = candidates

        if rows.empty:
            logger.debug("No CSV row for %s / %s in %s", metric, cnpj, filename)
            continue

        # ITR income-statement CSVs have two ÚLTIMO rows per account: quarterly
        # (DT_INI ≈ quarter start) and YTD (DT_INI = Jan 1).  We extract the
        # YTD value from the PDF, so select the row with the earliest DT_INI.
        if "DT_INI_EXERC" in rows.columns and len(rows) > 1:
            rows = rows.sort_values("DT_INI_EXERC")

        try:
            raw = rows.iloc[0]["VL_CONTA"]
            value = float(str(raw).replace(",", "."))
            ground_truth[metric] = value
        except (ValueError, KeyError) as exc:
            logger.warning("Could not parse CSV value for %s: %s", metric, exc)

    return ground_truth
