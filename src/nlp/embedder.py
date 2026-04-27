"""Embedding generation for indexing and querying.

Wraps the fine-tuned BERTimbau sentence transformer for batch embedding of
chunks (at index time) and single-query embedding (at inference time).
Pre-computes all embeddings so Streamlit Cloud (CPU-only) can serve queries.
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np

logger = logging.getLogger(__name__)

_model_cache: dict[str, object] = {}


def load_model(model_dir: Path | None = None, device: str | None = None):
    """Load the sentence transformer model for embedding.

    Falls back to the base ``neuralmind/bert-base-portuguese-cased`` model if
    no fine-tuned model is found at ``model_dir``. Results are cached in-process
    so repeated calls do not reload the model.

    Args:
        model_dir: Optional path to a fine-tuned model directory. If ``None``
            or missing, loads the base model from HuggingFace Hub.
        device: PyTorch device string (e.g. ``"cpu"``, ``"cuda"``). ``None``
            lets sentence-transformers pick automatically (GPU if available).

    Returns:
        A ``sentence_transformers.SentenceTransformer`` instance.
    """
    from sentence_transformers import SentenceTransformer
    from src import config

    cache_key = f"{model_dir}:{device}"
    if cache_key in _model_cache:
        return _model_cache[cache_key]

    # A valid SentenceTransformer directory must have modules.json or config.json
    # with a model_type key. Partial training output dirs (only checkpoints) are invalid.
    _valid = (
        model_dir is not None
        and model_dir.is_dir()
        and (
            (model_dir / "modules.json").exists()
            or (model_dir / "config.json").exists()
        )
    )

    kwargs: dict = {}
    if device is not None:
        kwargs["device"] = device

    if _valid:
        logger.info("Loading fine-tuned model from %s (device=%s)", model_dir, device or "auto")
        model = SentenceTransformer(str(model_dir), **kwargs)
    else:
        if model_dir:
            logger.info(
                "Fine-tuned model not found at %s — loading base model %s (device=%s)",
                model_dir, config.BERT_MODEL_NAME, device or "auto",
            )
        else:
            logger.info("Loading base model %s (device=%s)", config.BERT_MODEL_NAME, device or "auto")
        model = SentenceTransformer(config.BERT_MODEL_NAME, **kwargs)

    model.max_seq_length = config.MAX_SEQ_LENGTH
    _model_cache[cache_key] = model
    return model


def embed_chunks(
    texts: list[str],
    model_dir: Path | None = None,
    batch_size: int = 32,
    show_progress: bool = True,
    device: str | None = None,
) -> np.ndarray:
    """Embed a list of text chunks in batches.

    Args:
        texts: List of chunk strings to embed.
        model_dir: Path to fine-tuned model; falls back to base if missing.
        batch_size: Number of texts per forward pass.
        show_progress: Show a tqdm progress bar.
        device: Force a specific device (e.g. ``"cpu"`` when GPU is occupied).

    Returns:
        Float32 array of shape ``(len(texts), embedding_dim)``.
    """
    model = load_model(model_dir, device=device)
    embeddings = model.encode(
        texts,
        batch_size=batch_size,
        show_progress_bar=show_progress,
        convert_to_numpy=True,
        normalize_embeddings=True,
    )
    return embeddings.astype(np.float32)


def embed_query(
    query: str,
    model_dir: Path | None = None,
) -> np.ndarray:
    """Embed a single query string.

    Args:
        query: User query text.
        model_dir: Path to fine-tuned model.

    Returns:
        1-D float32 array of shape ``(embedding_dim,)``.
    """
    model = load_model(model_dir)
    embedding = model.encode(
        [query],
        batch_size=1,
        show_progress_bar=False,
        convert_to_numpy=True,
        normalize_embeddings=True,
    )
    return embedding[0].astype(np.float32)
