"""Hybrid retrieval combining dense (ChromaDB) and sparse (BM25) search.

Merges results using Reciprocal Rank Fusion (RRF) to produce a unified
top-K candidate list for reranking.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class RetrievedChunk:
    """A retrieved chunk with its hybrid RRF score."""

    chunk_id: str
    filing_id: str
    text: str
    section: str
    ticker: str
    reference_date: str
    dense_rank: int | None = None
    sparse_rank: int | None = None
    rrf_score: float = 0.0
    metadata: dict = field(default_factory=dict)


def retrieve_dense(
    query_embedding,
    collection,
    top_k: int = 20,
    filters: dict | None = None,
) -> list[RetrievedChunk]:
    """Retrieve top-K chunks from ChromaDB by vector similarity.

    Args:
        query_embedding: 1-D float array from :func:`~src.nlp.embedder.embed_query`.
        collection: A ``chromadb.Collection`` instance.
        top_k: Number of results to return.
        filters: Optional ChromaDB metadata filter dict (``where`` clause).

    Returns:
        List of :class:`RetrievedChunk` ordered by descending cosine similarity.
    """
    emb = query_embedding
    if hasattr(emb, "tolist"):
        emb = emb.tolist()

    query_params: dict = {
        "query_embeddings": [emb],
        "n_results": min(top_k, collection.count()),
        "include": ["documents", "metadatas", "distances"],
    }
    if filters:
        query_params["where"] = filters

    results = collection.query(**query_params)

    chunks: list[RetrievedChunk] = []
    for rank, (doc_id, doc, meta, _dist) in enumerate(zip(
        results["ids"][0],
        results["documents"][0],
        results["metadatas"][0],
        results["distances"][0],
    )):
        chunks.append(RetrievedChunk(
            chunk_id=doc_id,
            filing_id=meta.get("filing_id", ""),
            text=doc,
            section=meta.get("section_name", ""),
            ticker=meta.get("ticker", ""),
            reference_date=meta.get("reference_date", ""),
            dense_rank=rank,
            metadata=dict(meta),
        ))

    logger.debug("Dense retrieval returned %d results", len(chunks))
    return chunks


def retrieve_sparse(
    query: str,
    bm25_index,
    chunks: list,
    top_k: int = 20,
) -> list[RetrievedChunk]:
    """Retrieve top-K chunks from the BM25 index by keyword match.

    Args:
        query: Raw query string (tokenised by whitespace + lowercased).
        bm25_index: A ``rank_bm25.BM25Okapi`` instance.
        chunks: Ordered list of chunk records used to build the index. Each
            element may be a :class:`~src.parsing.chunker.Chunk` object or a
            plain dict with keys ``chunk_id``, ``filing_id``, ``section_type``,
            and ``text``.
        top_k: Number of results to return.

    Returns:
        List of :class:`RetrievedChunk` ordered by descending BM25 score.
    """
    tokenized_query = query.lower().split()
    scores = bm25_index.get_scores(tokenized_query)
    top_indices = np.argsort(scores)[::-1][:top_k]

    results: list[RetrievedChunk] = []
    for rank, idx in enumerate(top_indices):
        chunk = chunks[int(idx)]
        if isinstance(chunk, dict):
            chunk_id = chunk.get("chunk_id", "")
            filing_id = chunk.get("filing_id", "")
            section = chunk.get("section_type", "")
            text = chunk.get("text", "")
        else:
            chunk_id = chunk.chunk_id
            filing_id = chunk.filing_id
            section = chunk.section_type.value if hasattr(chunk.section_type, "value") else str(chunk.section_type)
            text = chunk.text

        parts = filing_id.split("_")
        ticker = parts[0] if parts else ""
        reference_date = parts[2] if len(parts) > 2 else ""

        results.append(RetrievedChunk(
            chunk_id=chunk_id,
            filing_id=filing_id,
            text=text,
            section=section,
            ticker=ticker,
            reference_date=reference_date,
            sparse_rank=rank,
        ))

    logger.debug("Sparse retrieval returned %d results", len(results))
    return results


def reciprocal_rank_fusion(
    dense_results: list[RetrievedChunk],
    sparse_results: list[RetrievedChunk],
    k: int = 60,
    top_n: int = 10,
) -> list[RetrievedChunk]:
    """Merge dense and sparse rankings using Reciprocal Rank Fusion.

    RRF score for chunk *d*: ``sum(1 / (k + rank_i(d) + 1))`` across all
    result lists (1-indexed rank).

    Args:
        dense_results: Ranked dense retrieval results.
        sparse_results: Ranked sparse retrieval results.
        k: RRF smoothing constant (default 60 per the original paper).
        top_n: Number of fused results to return.

    Returns:
        Top-N :class:`RetrievedChunk` objects ordered by descending RRF score.
    """
    merged: dict[str, RetrievedChunk] = {}
    rrf_scores: dict[str, float] = {}

    for rank, chunk in enumerate(dense_results):
        rrf_scores[chunk.chunk_id] = rrf_scores.get(chunk.chunk_id, 0.0) + 1.0 / (k + rank + 1)
        if chunk.chunk_id not in merged:
            merged[chunk.chunk_id] = chunk

    for rank, chunk in enumerate(sparse_results):
        rrf_scores[chunk.chunk_id] = rrf_scores.get(chunk.chunk_id, 0.0) + 1.0 / (k + rank + 1)
        if chunk.chunk_id not in merged:
            merged[chunk.chunk_id] = chunk
        else:
            # Carry sparse rank onto the existing entry
            merged[chunk.chunk_id].sparse_rank = chunk.sparse_rank

    for cid, score in rrf_scores.items():
        merged[cid].rrf_score = score

    fused = sorted(merged.values(), key=lambda c: c.rrf_score, reverse=True)[:top_n]
    logger.debug(
        "RRF fusion: %d dense + %d sparse → %d unique → top %d",
        len(dense_results), len(sparse_results), len(merged), len(fused),
    )
    return fused
