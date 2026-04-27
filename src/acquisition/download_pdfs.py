"""Download CVM PDF filings referenced in the manifest.

CVM's RAD portal (rad.cvm.gov.br) returns filing packages as ZIP archives
over HTTPS. The ZIP contains the PDF plus XML/XLSX metadata. This module:
  1. Warms up an HTTPS session to obtain portal cookies.
  2. Downloads each filing ZIP.
  3. Extracts the PDF from the ZIP and saves it to disk.

Must be run from a standard internet connection — the portal blocks
cloud/VPN IP ranges on HTTP port 80.
"""

from __future__ import annotations

import io
import logging
import threading
import time
import zipfile
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import requests

from src import config

logger = logging.getLogger(__name__)

DEFAULT_MAX_WORKERS: int = 2   # CVM rate-limits; 2 workers = safe sustained throughput
TIMEOUT: int = 120
MAX_RETRIES: int = 3
INTER_REQUEST_DELAY: float = 1.5  # seconds between downloads per worker

# Thread-local storage: one warm session per worker thread, created once.
_thread_local = threading.local()

_PORTAL_URL = "https://www.rad.cvm.gov.br/ENETCONSULTA/frmConsultaGeral.aspx"
_DOWNLOAD_URL_TEMPLATE = (
    "https://www.rad.cvm.gov.br/ENETCONSULTA/frmDownloadDocumento.aspx"
    "?CodigoInstituicao=1&NumeroSequencialDocumento={id_doc}"
)
_BROWSER_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 "
        "(KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
    ),
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
    "Accept-Language": "pt-BR,pt;q=0.9,en;q=0.8",
}


# ---------------------------------------------------------------------------
# Session management
# ---------------------------------------------------------------------------

def _make_session() -> requests.Session:
    """Create an HTTPS session pre-loaded with CVM portal cookies.

    Returns:
        A :class:`requests.Session` ready to download filing ZIPs.
    """
    s = requests.Session()
    s.headers.update(_BROWSER_HEADERS)
    try:
        s.get(_PORTAL_URL, timeout=30)
        logger.debug("Portal session initialised (%d cookies)", len(s.cookies))
    except requests.RequestException as exc:
        logger.warning("Could not warm up portal session: %s — proceeding anyway", exc)
    return s


def _get_thread_session() -> requests.Session:
    """Return the cached session for the current worker thread.

    Creates and warms up a new session on first call per thread. Subsequent
    calls on the same thread reuse the existing session — one portal warm-up
    per worker, not per download.

    Returns:
        Thread-local :class:`requests.Session`.
    """
    if not hasattr(_thread_local, "session"):
        _thread_local.session = _make_session()
    return _thread_local.session


# ---------------------------------------------------------------------------
# Single filing download
# ---------------------------------------------------------------------------

def _extract_pdf_from_zip(zip_bytes: bytes) -> bytes | None:
    """Extract the PDF file from a CVM filing ZIP archive.

    The CVM filing ZIP contains: one PDF, one XML, one XLSX, and optional
    additional XMLs. We return the bytes of the first `.pdf` entry found.

    Args:
        zip_bytes: Raw bytes of the downloaded ZIP archive.

    Returns:
        PDF bytes, or ``None`` if no PDF entry is found.
    """
    try:
        with zipfile.ZipFile(io.BytesIO(zip_bytes)) as zf:
            pdf_names = [n for n in zf.namelist() if n.lower().endswith(".pdf")]
            if not pdf_names:
                logger.warning("No PDF found in filing ZIP (files: %s)", zf.namelist())
                return None
            return zf.read(pdf_names[0])
    except zipfile.BadZipFile as exc:
        logger.warning("Response is not a valid ZIP: %s", exc)
        return None


def download_pdf(
    record: dict,
    output_path: Path,
    session: requests.Session | None = None,
    retries: int = MAX_RETRIES,
    timeout: int = TIMEOUT,
) -> bool:
    """Download and save the PDF for a single filing.

    Downloads the CVM filing ZIP via HTTPS, extracts the embedded PDF, and
    saves it to ``output_path``. Skips if the file already exists.

    Args:
        record: A manifest record with at least an ``id_doc`` field.
        output_path: Destination path for the PDF file.
        session: An existing :class:`requests.Session` (re-used for keep-alive).
            Creates a new one if ``None``.
        retries: Number of retry attempts on failure.
        timeout: HTTP request timeout in seconds.

    Returns:
        ``True`` on success (including skip), ``False`` if all retries fail.
    """
    if output_path.exists() and output_path.stat().st_size > 0:
        logger.debug("Already exists, skipping: %s", output_path.name)
        return True

    output_path.parent.mkdir(parents=True, exist_ok=True)
    id_doc = record.get("id_doc", "")
    if not id_doc:
        logger.warning("No id_doc in record %s — skipping", record.get("filing_id"))
        return False

    url = _DOWNLOAD_URL_TEMPLATE.format(id_doc=id_doc)
    sess = session or _make_session()

    for attempt in range(1, retries + 1):
        try:
            resp = sess.get(url, timeout=timeout, stream=False)
            resp.raise_for_status()

            pdf_bytes = _extract_pdf_from_zip(resp.content)
            if pdf_bytes is None:
                return False

            output_path.write_bytes(pdf_bytes)
            logger.info("Saved %s (%.1f MB)", output_path.name, len(pdf_bytes) / 1_048_576)
            return True

        except requests.RequestException as exc:
            logger.warning(
                "Attempt %d/%d failed for %s: %s",
                attempt, retries, output_path.name, exc,
            )
            if attempt < retries:
                time.sleep(2 ** attempt)

    logger.error("Gave up on %s after %d attempts", output_path.name, retries)
    return False


# ---------------------------------------------------------------------------
# Filename helper
# ---------------------------------------------------------------------------

def _make_pdf_path(record: dict, output_dir: Path) -> Path:
    """Construct a deterministic local PDF path from a manifest record.

    Args:
        record: Manifest record with ticker, filing_type, reference_date, version.
        output_dir: Root directory for PDF storage.

    Returns:
        Absolute :class:`~pathlib.Path` for this filing's PDF.
    """
    ticker = record["ticker"]
    doc_type = record["filing_type"].lower()
    date = record["reference_date"].replace("-", "")
    version = record.get("version", "1")
    filename = f"{ticker}_{doc_type}_{date}_v{version}.pdf"
    return output_dir / ticker / filename


# ---------------------------------------------------------------------------
# Batch download
# ---------------------------------------------------------------------------

def download_pdfs(
    manifest: list[dict],
    output_dir: Path = config.PDFS_DIR,
    max_workers: int = DEFAULT_MAX_WORKERS,
    limit: int | None = None,
) -> list[dict]:
    """Download all PDFs referenced in the manifest, in parallel.

    Each worker shares one HTTPS session (per-thread) to benefit from
    persistent cookies and TCP keep-alive.

    Args:
        manifest: Filing manifest from :func:`download_csvs.build_manifest`.
        output_dir: Root directory under which PDFs are saved per ticker.
        max_workers: Number of parallel download threads.
        limit: If set, download at most this many PDFs.

    Returns:
        Updated manifest records with ``pdf_path`` filled in.
    """
    to_download = [r for r in manifest if r.get("id_doc")]
    if limit is not None:
        to_download = to_download[:limit]

    logger.info("Downloading %d PDFs with %d workers…", len(to_download), max_workers)
    output_dir.mkdir(parents=True, exist_ok=True)

    results = {r["filing_id"]: r for r in manifest}

    def _task(record: dict) -> tuple[str, Path | None]:
        sess = _get_thread_session()   # one session per thread, reused across downloads
        pdf_path = _make_pdf_path(record, output_dir)
        success = download_pdf(record, pdf_path, session=sess)
        if INTER_REQUEST_DELAY > 0:
            time.sleep(INTER_REQUEST_DELAY)
        return record["filing_id"], pdf_path if success else None

    total = len(to_download)
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {executor.submit(_task, r): r["filing_id"] for r in to_download}
        succeeded = failed = done = 0
        for future in as_completed(futures):
            filing_id, pdf_path = future.result()
            done += 1
            if pdf_path:
                results[filing_id]["pdf_path"] = str(pdf_path)
                succeeded += 1
            else:
                results[filing_id]["pdf_path"] = None
                failed += 1
            if done % 10 == 0 or done == total:
                logger.info("Progress: %d/%d  (✓%d  ✗%d)", done, total, succeeded, failed)

    logger.info("PDF download complete: %d succeeded, %d failed", succeeded, failed)
    return list(results.values())


def resolve_pdf_url(csv_row: dict) -> str | None:
    """Extract the PDF download URL from a CVM CSV row.

    Args:
        csv_row: A single row from a CVM ITR or DFP structured CSV.

    Returns:
        PDF URL string, or ``None`` if not present.
    """
    return csv_row.get("LINK_DOC") or None
