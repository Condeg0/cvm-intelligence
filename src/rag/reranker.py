"""Cross-encoder reranking of hybrid retrieval results.

Takes the top-10 RRF candidates and reranks them with
``cross-encoder/ms-marco-MiniLM-L-6-v2``, returning the top-5 final results
for context assembly.
"""

from __future__ import annotations

import logging

from src.rag.retriever import RetrievedChunk

logger = logging.getLogger(__name__)

RERANKER_MODEL: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"

_reranker_cache: dict[str, object] = {}


def rerank(
    query: str,
    candidates: list[RetrievedChunk],
    top_k: int = 5,
    model_name: str = RERANKER_MODEL,
) -> list[RetrievedChunk]:
    """Rerank retrieval candidates with a cross-encoder model.

    Scores each (query, chunk_text) pair and returns the top-K by cross-encoder
    relevance score. The model is cached in-process after the first load.

    Args:
        query: The user's query string.
        candidates: Up to 10 candidates from
            :func:`~src.rag.retriever.reciprocal_rank_fusion`.
        top_k: Number of results to keep after reranking.
        model_name: HuggingFace model identifier for the cross-encoder.

    Returns:
        Top-K :class:`~src.rag.retriever.RetrievedChunk` objects ordered by
        descending cross-encoder relevance score.
    """
    if not candidates:
        return []

    from sentence_transformers import CrossEncoder

    if model_name not in _reranker_cache:
        logger.info("Loading cross-encoder %r …", model_name)
        _reranker_cache[model_name] = CrossEncoder(model_name)

    model = _reranker_cache[model_name]

    pairs = [(query, chunk.text) for chunk in candidates]
    scores = model.predict(pairs)

    scored = sorted(zip(candidates, scores), key=lambda x: x[1], reverse=True)
    top = [chunk for chunk, _ in scored[:top_k]]
    logger.debug(
        "Reranker: %d candidates → top %d (model=%s)", len(candidates), len(top), model_name
    )
    return top
