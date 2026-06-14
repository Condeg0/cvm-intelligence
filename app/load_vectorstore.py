"""ChromaDB availability helper for the Streamlit app.

On local dev: the vectorstore/chromadb/ directory is built by run_indexing.py
and used directly. Nothing is downloaded.

On a high-memory deployment (>=4 GB RAM): call ensure_chromadb() once at
startup to pull the pre-built index from HuggingFace Hub dataset
condeg/cvm-chromadb if it is not already present on disk.

NOTE: Streamlit Community Cloud is capped at 1 GB RAM. The ChromaDB index is
1.8 GB and cannot fit in that environment. The RAG query page degrades
gracefully when ChromaDB is absent, so this function is effectively a no-op on
Community Cloud — it just returns False and the query page shows its
"not available" message.
"""

from __future__ import annotations

import logging
from pathlib import Path

logger = logging.getLogger(__name__)

# Resolved at import time so we can use it without importing all of src/
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
_CHROMADB_DIR = _PROJECT_ROOT / "vectorstore" / "chromadb"
_HF_REPO_ID = "condeg/cvm-chromadb"


def _chromadb_present() -> bool:
    """Return True if a non-empty ChromaDB directory exists on disk."""
    if not _CHROMADB_DIR.exists():
        return False
    children = list(_CHROMADB_DIR.iterdir())
    return len(children) > 0


def ensure_chromadb(show_progress: bool = True) -> bool:
    """Download ChromaDB from HuggingFace Hub if it is not present locally.

    Args:
        show_progress: If True, log progress messages visible in Streamlit logs.

    Returns:
        True if ChromaDB is available after this call, False otherwise.
    """
    if _chromadb_present():
        logger.debug("ChromaDB already present at %s — skipping download.", _CHROMADB_DIR)
        return True

    try:
        from huggingface_hub import snapshot_download
    except ImportError:
        logger.warning("huggingface_hub not installed — cannot auto-download ChromaDB.")
        return False

    logger.info(
        "ChromaDB not found locally. Attempting download from HuggingFace Hub: %s",
        _HF_REPO_ID,
    )

    if show_progress:
        try:
            import streamlit as st
            st.info(
                f"Downloading ChromaDB vector index from `{_HF_REPO_ID}` (~1.8 GB). "
                "This runs once and may take several minutes…"
            )
        except Exception:
            pass

    try:
        snapshot_download(
            repo_id=_HF_REPO_ID,
            repo_type="dataset",
            local_dir=str(_CHROMADB_DIR),
            local_dir_use_symlinks=False,
        )
        logger.info("ChromaDB downloaded successfully to %s", _CHROMADB_DIR)
        return True
    except Exception as exc:
        logger.error("ChromaDB download failed: %s", exc)
        return False
