"""Streamlit entry point for the CVM Filing Intelligence System.

Multi-page app structure — Streamlit discovers pages automatically from
the ``pages/`` directory. This file configures shared page settings and
renders the landing / home page.

Startup responsibilities:
- Bridge st.secrets → os.environ so that config.py's os.getenv() calls
  work on Streamlit Cloud where secrets are NOT in the environment.
- Check for required data files and display clear error messages if
  they are missing, rather than letting pages crash silently.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import streamlit as st

# ---------------------------------------------------------------------------
# Bridge st.secrets → os.environ (must happen before any src/ import)
# On Streamlit Cloud, API keys live in st.secrets, not os.environ.
# ---------------------------------------------------------------------------
try:
    for _k, _v in st.secrets.items():
        if isinstance(_v, str):
            os.environ.setdefault(_k, _v)
except Exception:
    pass  # no secrets file locally — fine, .env is loaded by config.py

# ---------------------------------------------------------------------------
# Path setup and data availability checks
# ---------------------------------------------------------------------------
_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT))

from src import config  # noqa: E402 — must come after sys.path insert

_DB_OK = config.DASHBOARD_DB_PATH.exists()
_CHROMA_OK = (config.CHROMADB_DIR / "chroma.sqlite3").exists()

# ---------------------------------------------------------------------------
# Page config
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="CVM Filing Intelligence",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------------------------------------------------------------------------
# Data availability banner
# ---------------------------------------------------------------------------
if not _DB_OK:
    st.error(
        "**Database not found:** `data/cvm_metrics_dashboard.db` is missing.\n\n"
        "The Financial Dashboard and Evaluation pages require this file. "
        "It should be committed to the repository — please ensure you have "
        "the latest `main` branch checked out."
    )

# ---------------------------------------------------------------------------
# Home page
# ---------------------------------------------------------------------------
st.title("CVM Filing Intelligence System")
st.markdown(
    """
    An end-to-end pipeline that transforms Brazilian public company filings
    (CVM ITR and DFP documents) into a queryable intelligence system.

    **Select a page from the sidebar to get started.**

    | Page | Description | Requires |
    |---|---|---|
    | 🔍 Query | Ask natural-language questions about filings (RAG) | ChromaDB + BM25 index (local only) |
    | 📈 Dashboard | Financial metrics over time | SQLite DB |
    | 💬 Sentiment | Management commentary sentiment timeline | SQLite DB |
    | 🧪 Evaluation | System evaluation metrics | SQLite DB + pre-computed JSONs |
    """
)

if _DB_OK and not _CHROMA_OK:
    st.info(
        "**RAG query page is in read-only mode.** "
        "The ChromaDB vector index (1.8 GB) is not available in this environment — "
        "the Query page will show a setup message. "
        "All other pages are fully functional."
    )

st.sidebar.success("Select a page above.")
