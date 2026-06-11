"""Centralized configuration: paths, constants, and API setup.

All path constants and environment-dependent settings live here.
Import from this module instead of hardcoding paths anywhere else.
"""

from __future__ import annotations

import logging
import os
from pathlib import Path

from dotenv import load_dotenv

# ---------------------------------------------------------------------------
# Project root (two levels up from this file: src/ → project root)
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent

# ---------------------------------------------------------------------------
# Load .env (GEMINI_API_KEY, etc.)
# ---------------------------------------------------------------------------
load_dotenv(PROJECT_ROOT / ".env")

# ---------------------------------------------------------------------------
# Data directories
# ---------------------------------------------------------------------------
DATA_DIR = PROJECT_ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
CSVS_DIR = RAW_DIR / "csvs"
PDFS_DIR = RAW_DIR / "pdfs"
PROCESSED_DIR = DATA_DIR / "processed"
CHUNKS_DIR = PROCESSED_DIR / "chunks"
METRICS_DIR = PROCESSED_DIR / "metrics"
EVALUATION_DIR = DATA_DIR / "evaluation"

# ---------------------------------------------------------------------------
# Database
# ---------------------------------------------------------------------------
DB_PATH = DATA_DIR / "cvm_metrics.db"

# ---------------------------------------------------------------------------
# Models
# ---------------------------------------------------------------------------
MODELS_DIR = PROJECT_ROOT / "models"
SENTIMENT_MODEL_DIR = MODELS_DIR / "sentiment_classifier"

# HuggingFace Hub ID for the fine-tuned sentence transformer
SENTENCE_TRANSFORMER_HUB_ID: str = "condeg/cvm-bertimbau-sentence-transformer"

# Resolved to local weights during development, Hub ID on a fresh clone / Streamlit Cloud
_local_st = MODELS_DIR / "sentence_transformer"
SENTENCE_TRANSFORMER_PATH: Path | str = _local_st if _local_st.is_dir() else SENTENCE_TRANSFORMER_HUB_ID

# ---------------------------------------------------------------------------
# Vector store
# ---------------------------------------------------------------------------
VECTORSTORE_DIR = PROJECT_ROOT / "vectorstore"
CHROMADB_DIR = VECTORSTORE_DIR / "chromadb"

# ---------------------------------------------------------------------------
# Evaluation artefacts
# ---------------------------------------------------------------------------
RETRIEVAL_TEST_SET_PATH = EVALUATION_DIR / "retrieval_test_set.json"
SENTIMENT_LABELS_PATH = EVALUATION_DIR / "sentiment_labels.json"

# ---------------------------------------------------------------------------
# NLP / embedding constants
# ---------------------------------------------------------------------------
BERT_MODEL_NAME: str = "neuralmind/bert-base-portuguese-cased"
RERANKER_MODEL_NAME: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"
MAX_SEQ_LENGTH: int = 256
EMBEDDING_BATCH_SIZE: int = 32

# ---------------------------------------------------------------------------
# Training hyper-parameters
# ---------------------------------------------------------------------------
FINETUNE_BATCH_SIZE: int = 16
GRADIENT_ACCUMULATION_STEPS: int = 4  # effective batch = 64
USE_FP16: bool = True

# ---------------------------------------------------------------------------
# ChromaDB collection name
# ---------------------------------------------------------------------------
CHROMA_COLLECTION_NAME: str = "cvm_chunks"

# ---------------------------------------------------------------------------
# Retrieval constants
# ---------------------------------------------------------------------------
TOP_K_DENSE: int = 20
TOP_K_SPARSE: int = 20
TOP_K_RRF: int = 10
TOP_K_RERANKED: int = 5
RRF_K: int = 60  # standard RRF constant

# ---------------------------------------------------------------------------
# Gemini API setup
# ---------------------------------------------------------------------------
GEMINI_API_KEY: str | None = os.getenv("GEMINI_API_KEY")
GEMINI_MODEL: str = "models/gemini-2.5-flash"

try:
    import google.genai as genai  # type: ignore  # new google-genai SDK

    _GENAI_CLIENT: object | None = genai.Client(api_key=GEMINI_API_KEY) if GEMINI_API_KEY else None
    if not GEMINI_API_KEY:
        logging.getLogger(__name__).warning(
            "GEMINI_API_KEY not set — Gemini calls will fail."
        )
except ImportError:
    _GENAI_CLIENT = None
    try:
        # Fallback: old google-generativeai package (deprecated)
        import google.generativeai as _old_genai  # type: ignore

        if GEMINI_API_KEY:
            _old_genai.configure(api_key=GEMINI_API_KEY)
        logging.getLogger(__name__).warning(
            "Using deprecated google-generativeai SDK. "
            "Please upgrade to google-genai: pip install google-genai"
        )
    except ImportError:
        logging.getLogger(__name__).warning(
            "Neither google-genai nor google-generativeai is installed; "
            "Gemini features unavailable."
        )

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
LOG_LEVEL: int = logging.INFO
