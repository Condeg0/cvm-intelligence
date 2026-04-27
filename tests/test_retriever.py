"""Tests for src/rag/retriever.py — RRF fusion logic."""

from __future__ import annotations

import pytest

from src.rag.retriever import RetrievedChunk, reciprocal_rank_fusion


def make_chunk(chunk_id: str) -> RetrievedChunk:
    """Create a minimal RetrievedChunk for testing."""
    return RetrievedChunk(
        chunk_id=chunk_id,
        filing_id="filing_001",
        text=f"Text for chunk {chunk_id}",
        section="DRE",
        ticker="PETR4",
        reference_date="2023-09-30",
    )


class TestRRF:
    def test_merges_disjoint_lists(self):
        dense = [make_chunk(f"d{i}") for i in range(3)]
        sparse = [make_chunk(f"s{i}") for i in range(3)]
        for i, c in enumerate(dense):
            c.dense_rank = i + 1
        for i, c in enumerate(sparse):
            c.sparse_rank = i + 1

        result = reciprocal_rank_fusion(dense, sparse, top_n=5)
        ids = [c.chunk_id for c in result]
        # Top dense and sparse chunks should both appear
        assert "d0" in ids
        assert "s0" in ids

    def test_chunk_appearing_in_both_lists_ranks_higher(self):
        shared = make_chunk("shared")
        shared.dense_rank = 1
        shared.sparse_rank = 1

        unique_dense = make_chunk("unique_dense")
        unique_dense.dense_rank = 2

        result = reciprocal_rank_fusion([shared, unique_dense], [shared], top_n=3)
        # Shared chunk appears in both lists → higher RRF score → first
        assert result[0].chunk_id == "shared"

    def test_respects_top_n(self):
        dense = [make_chunk(f"d{i}") for i in range(10)]
        sparse = [make_chunk(f"s{i}") for i in range(10)]
        for i, c in enumerate(dense):
            c.dense_rank = i + 1
        for i, c in enumerate(sparse):
            c.sparse_rank = i + 1

        result = reciprocal_rank_fusion(dense, sparse, top_n=5)
        assert len(result) == 5

    def test_rrf_scores_are_positive(self):
        dense = [make_chunk("a")]
        dense[0].dense_rank = 1
        sparse = [make_chunk("b")]
        sparse[0].sparse_rank = 1

        result = reciprocal_rank_fusion(dense, sparse, top_n=2)
        for chunk in result:
            assert chunk.rrf_score > 0
