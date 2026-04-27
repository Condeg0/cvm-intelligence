"""Download CVM DADOS_ABERTOS CSVs for the 50 largest B3 companies.

Downloads:
  - cad_cia_aberta.csv  — company registration (CNPJ / CD_CVM lookup)
  - itr_cia_aberta_{year}.zip  — quarterly filing metadata + financial data (2023-2025)
  - dfp_cia_aberta_{year}.zip  — annual filing metadata + financial data (2022-2024)

Extracts and filters only rows for the 50 target companies, then builds a
manifest JSON mapping each filing to its CSV rows and PDF link.
"""

from __future__ import annotations

import io
import json
import logging
import time
import zipfile
from pathlib import Path
from typing import Any

import pandas as pd
import requests

from src import config

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
BASE_URL = "https://dados.cvm.gov.br/dados/CIA_ABERTA"
CAD_URL = f"{BASE_URL}/CAD/DADOS/cad_cia_aberta.csv"
ITR_URL_TEMPLATE = f"{BASE_URL}/DOC/ITR/DADOS/itr_cia_aberta_{{year}}.zip"
DFP_URL_TEMPLATE = f"{BASE_URL}/DOC/DFP/DADOS/dfp_cia_aberta_{{year}}.zip"

# CSV files we care about inside each ZIP (consolidated statements)
ITR_STATEMENT_FILES = [
    "itr_cia_aberta_{year}.csv",        # filing metadata (has LINK_DOC)
    "itr_cia_aberta_DRE_con_{year}.csv",
    "itr_cia_aberta_BPA_con_{year}.csv",
    "itr_cia_aberta_BPP_con_{year}.csv",
    "itr_cia_aberta_DFC_MI_con_{year}.csv",
    "itr_cia_aberta_DVA_con_{year}.csv",
]

DFP_STATEMENT_FILES = [
    "dfp_cia_aberta_{year}.csv",
    "dfp_cia_aberta_DRE_con_{year}.csv",
    "dfp_cia_aberta_BPA_con_{year}.csv",
    "dfp_cia_aberta_BPP_con_{year}.csv",
    "dfp_cia_aberta_DFC_MI_con_{year}.csv",
    "dfp_cia_aberta_DVA_con_{year}.csv",
]

CSV_ENCODING = "utf-8"  # CVM CSVs are UTF-8 (confirmed by file inspection)
CSV_SEP = ";"

ITR_YEARS = [2023, 2024, 2025]
DFP_YEARS = [2022, 2023, 2024]

TARGET_COMPANIES_PATH = config.RAW_DIR / "target_companies.json"
MANIFEST_PATH = config.RAW_DIR / "manifest.json"
CAD_PATH = config.CSVS_DIR / "cad_cia_aberta.csv"


# ---------------------------------------------------------------------------
# HTTP helpers
# ---------------------------------------------------------------------------

def _download_bytes(url: str, retries: int = 3, timeout: int = 60) -> bytes:
    """Download URL contents with retries.

    Args:
        url: Target URL.
        retries: Number of retry attempts on failure.
        timeout: Request timeout in seconds.

    Returns:
        Raw response bytes.

    Raises:
        requests.HTTPError: If all retries are exhausted.
    """
    for attempt in range(1, retries + 1):
        try:
            resp = requests.get(url, timeout=timeout, stream=True)
            resp.raise_for_status()
            data = resp.content
            logger.info("Downloaded %s (%.1f KB)", url.split("/")[-1], len(data) / 1024)
            return data
        except requests.RequestException as exc:
            logger.warning("Attempt %d/%d failed for %s: %s", attempt, retries, url, exc)
            if attempt == retries:
                raise
            time.sleep(2 ** attempt)
    raise RuntimeError("Unreachable")


# ---------------------------------------------------------------------------
# Step 1 — cadaster
# ---------------------------------------------------------------------------

def download_cad(output_dir: Path = config.CSVS_DIR) -> Path:
    """Download the CVM company registration CSV.

    Args:
        output_dir: Directory to save the file.

    Returns:
        Path to the saved CSV file.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    out_path = output_dir / "cad_cia_aberta.csv"
    if out_path.exists():
        logger.info("Cadaster already present at %s, skipping download.", out_path)
        return out_path
    data = _download_bytes(CAD_URL)
    out_path.write_bytes(data)
    logger.info("Saved cadaster to %s", out_path)
    return out_path


# ---------------------------------------------------------------------------
# Step 2 — resolve target companies from cadaster
# ---------------------------------------------------------------------------

def resolve_companies(
    target_path: Path = TARGET_COMPANIES_PATH,
    cad_path: Path = CAD_PATH,
) -> dict[str, dict[str, str]]:
    """Match target companies against the CVM cadaster to get CNPJs and CD_CVM.

    Performs case-insensitive substring matching of each company's ``name_search``
    pattern against the ``DENOM_SOCIAL`` and ``DENOM_COMERC`` columns. When
    multiple matches exist, prefers companies with SIT == 'ATIVO'.

    Args:
        target_path: JSON file with list of ``{ticker, name_search, sector}`` dicts.
        cad_path: Path to the downloaded ``cad_cia_aberta.csv``.

    Returns:
        Dict mapping ticker → ``{ticker, cnpj, cd_cvm, name, sector}``.
    """
    import unicodedata

    def _normalize(s: pd.Series) -> pd.Series:
        """Strip accents and upper-case for fuzzy ASCII matching."""
        return (
            s.fillna("")
            .str.upper()
            .str.normalize("NFKD")
            .str.encode("ascii", "ignore")
            .str.decode("ascii")
        )

    targets: list[dict] = json.loads(target_path.read_text(encoding="utf-8"))
    cad = pd.read_csv(cad_path, sep=CSV_SEP, encoding=CSV_ENCODING, dtype=str)
    cad.columns = cad.columns.str.strip()

    # Build normalised search columns once
    cad["_social_n"] = _normalize(cad["DENOM_SOCIAL"])
    cad["_comerc_n"] = _normalize(cad["DENOM_COMERC"])

    resolved: dict[str, dict[str, str]] = {}
    unmatched: list[str] = []

    for company in targets:
        ticker = company["ticker"]
        # Normalise the search pattern the same way
        pattern = (
            unicodedata.normalize("NFKD", company["name_search"].upper())
            .encode("ascii", "ignore")
            .decode("ascii")
        )

        mask = (
            cad["_social_n"].str.contains(pattern, na=False, regex=False)
            | cad["_comerc_n"].str.contains(pattern, na=False, regex=False)
        )
        matches = cad[mask].copy()

        if matches.empty:
            logger.warning("No cadaster match for %s (pattern: %r)", ticker, pattern)
            unmatched.append(ticker)
            continue

        # Prefer active companies; within that, take the first match
        active = matches[matches["SIT"] == "ATIVO"]
        row = active.iloc[0] if not active.empty else matches.iloc[0]

        resolved[ticker] = {
            "ticker": ticker,
            "cnpj": row["CNPJ_CIA"],
            "cd_cvm": row["CD_CVM"],
            "name": row["DENOM_SOCIAL"],
            "sector": company["sector"],
        }
        logger.debug("Resolved %s → CNPJ %s", ticker, row["CNPJ_CIA"])

    logger.info(
        "Resolved %d/%d target companies (%d unmatched)",
        len(resolved), len(targets), len(unmatched),
    )
    if unmatched:
        logger.warning("Unmatched tickers: %s", unmatched)
    return resolved


# ---------------------------------------------------------------------------
# Step 3 — download and extract ZIP files
# ---------------------------------------------------------------------------

def download_and_extract_zip(
    url: str,
    target_files: list[str],
    output_dir: Path,
    cnpj_set: set[str],
    year: int,
) -> list[Path]:
    """Download a CVM ZIP, extract relevant CSVs, and filter for target companies.

    Args:
        url: URL of the ZIP file to download.
        target_files: List of filename templates (with ``{year}`` placeholder)
            to extract from the ZIP.
        output_dir: Directory to save filtered CSVs.
        cnpj_set: Set of CNPJ strings for the target companies.
        year: Calendar year (used to format ``target_files`` templates).

    Returns:
        List of paths to saved filtered CSV files.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    raw_bytes = _download_bytes(url)
    saved: list[Path] = []

    with zipfile.ZipFile(io.BytesIO(raw_bytes)) as zf:
        available = {name.lower(): name for name in zf.namelist()}
        for template in target_files:
            filename = template.format(year=year)
            actual_name = available.get(filename.lower())
            if actual_name is None:
                logger.warning("File %r not found in ZIP %s", filename, url.split("/")[-1])
                continue

            with zf.open(actual_name) as f:
                raw_csv = f.read()

            df = pd.read_csv(
                io.BytesIO(raw_csv),
                sep=CSV_SEP,
                encoding=CSV_ENCODING,
                dtype=str,
                low_memory=False,
            )
            df.columns = df.columns.str.strip()

            if "CNPJ_CIA" in df.columns:
                df = df[df["CNPJ_CIA"].isin(cnpj_set)]

            out_path = output_dir / filename
            df.to_csv(out_path, sep=CSV_SEP, index=False, encoding="utf-8")
            logger.info("Saved %s (%d rows)", out_path.name, len(df))
            saved.append(out_path)

    return saved


def download_all_financial_data(
    companies: dict[str, dict[str, str]],
    output_dir: Path = config.CSVS_DIR,
    itr_years: list[int] = ITR_YEARS,
    dfp_years: list[int] = DFP_YEARS,
) -> None:
    """Download ITR and DFP ZIPs for all target years and filter for target companies.

    Args:
        companies: Resolved companies dict from :func:`resolve_companies`.
        output_dir: Directory where filtered CSVs will be saved.
        itr_years: List of years to download for ITR (quarterly) data.
        dfp_years: List of years to download for DFP (annual) data.
    """
    cnpj_set = {c["cnpj"] for c in companies.values()}
    logger.info("Downloading financial data for %d companies", len(cnpj_set))

    for year in itr_years:
        url = ITR_URL_TEMPLATE.format(year=year)
        out_dir = output_dir / f"itr_{year}"
        logger.info("Downloading ITR %d…", year)
        try:
            download_and_extract_zip(url, ITR_STATEMENT_FILES, out_dir, cnpj_set, year)
        except Exception as exc:
            logger.error("Failed to download ITR %d: %s", year, exc)

    for year in dfp_years:
        url = DFP_URL_TEMPLATE.format(year=year)
        out_dir = output_dir / f"dfp_{year}"
        logger.info("Downloading DFP %d…", year)
        try:
            download_and_extract_zip(url, DFP_STATEMENT_FILES, out_dir, cnpj_set, year)
        except Exception as exc:
            logger.error("Failed to download DFP %d: %s", year, exc)


# ---------------------------------------------------------------------------
# Step 4 — build manifest
# ---------------------------------------------------------------------------

def build_manifest(
    companies: dict[str, dict[str, str]],
    csvs_dir: Path = config.CSVS_DIR,
    output_path: Path = MANIFEST_PATH,
) -> list[dict[str, Any]]:
    """Build a filing manifest mapping each company/period to its CSV rows and PDF link.

    Reads the main filing metadata CSVs (``itr_cia_aberta_{year}.csv`` and
    ``dfp_cia_aberta_{year}.csv``) to extract filing IDs, reference dates, and
    PDF download links.

    Args:
        companies: Resolved companies dict from :func:`resolve_companies`.
        csvs_dir: Root of the downloaded/filtered CSV directory.
        output_path: Where to write the manifest JSON.

    Returns:
        List of manifest record dicts.
    """
    cnpj_to_ticker = {v["cnpj"]: k for k, v in companies.items()}
    manifest: list[dict[str, Any]] = []

    meta_files: list[tuple[str, str]] = []
    for year in ITR_YEARS:
        meta_files.append(("ITR", str(csvs_dir / f"itr_{year}" / f"itr_cia_aberta_{year}.csv")))
    for year in DFP_YEARS:
        meta_files.append(("DFP", str(csvs_dir / f"dfp_{year}" / f"dfp_cia_aberta_{year}.csv")))

    for doc_type, csv_path_str in meta_files:
        csv_path = Path(csv_path_str)
        if not csv_path.exists():
            logger.warning("Metadata CSV not found: %s", csv_path)
            continue

        df = pd.read_csv(csv_path, sep=CSV_SEP, encoding="utf-8", dtype=str)
        df.columns = df.columns.str.strip()

        for _, row in df.iterrows():
            cnpj = row.get("CNPJ_CIA", "")
            ticker = cnpj_to_ticker.get(cnpj)
            if ticker is None:
                continue

            manifest.append({
                "filing_id": f"{ticker}_{doc_type}_{row.get('DT_REFER', '')}_{row.get('VERSAO', '1')}",
                "ticker": ticker,
                "cnpj": cnpj,
                "cd_cvm": row.get("CD_CVM", ""),
                "company_name": row.get("DENOM_CIA", ""),
                "filing_type": doc_type,
                "reference_date": row.get("DT_REFER", ""),
                "received_date": row.get("DT_RECEB", ""),
                "version": row.get("VERSAO", "1"),
                "id_doc": row.get("ID_DOC", ""),
                "pdf_url": row.get("LINK_DOC", ""),
                "pdf_path": None,  # filled in by download_pdfs
                "csv_dir": str(csv_path.parent),
            })

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    logger.info("Manifest written: %d filings → %s", len(manifest), output_path)
    return manifest


# ---------------------------------------------------------------------------
# Orchestrator
# ---------------------------------------------------------------------------

def run(
    itr_years: list[int] = ITR_YEARS,
    dfp_years: list[int] = DFP_YEARS,
) -> list[dict[str, Any]]:
    """Run the full CSV acquisition pipeline.

    Downloads cadaster, resolves companies, downloads financial data, and
    builds the filing manifest.

    Args:
        itr_years: Years to download for quarterly (ITR) data.
        dfp_years: Years to download for annual (DFP) data.

    Returns:
        The filing manifest as a list of record dicts.
    """
    download_cad()
    companies = resolve_companies()
    # Save resolved companies for reference
    resolved_path = config.RAW_DIR / "resolved_companies.json"
    resolved_path.write_text(
        json.dumps(list(companies.values()), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    logger.info("Saved resolved companies to %s", resolved_path)

    download_all_financial_data(companies, itr_years=itr_years, dfp_years=dfp_years)
    return build_manifest(companies)


