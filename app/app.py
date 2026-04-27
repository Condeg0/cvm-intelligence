"""Streamlit entry point for the CVM Filing Intelligence System.

Multi-page app structure — Streamlit discovers pages automatically from
the ``pages/`` directory. This file configures shared page settings and
renders the landing / home page.
"""

from __future__ import annotations

import streamlit as st

st.set_page_config(
    page_title="CVM Filing Intelligence",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.title("CVM Filing Intelligence System")
st.markdown(
    """
    An end-to-end pipeline that transforms Brazilian public company filings
    (CVM ITR and DFP documents) into a queryable intelligence system.

    **Select a page from the sidebar to get started.**

    | Page | Description |
    |---|---|
    | 🔍 Query | Ask natural-language questions about filings (RAG) |
    | 📈 Dashboard | Financial metrics over time |
    | 💬 Sentiment | Management commentary sentiment timeline |
    | 🧪 Evaluation | System evaluation metrics (retrieval, extraction) |
    """
)

st.sidebar.success("Select a page above.")
