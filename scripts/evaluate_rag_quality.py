"""End-to-end RAG quality evaluation on 20 representative queries.

Runs the full pipeline (hybrid retrieval → cross-encoder reranking →
Gemini generation) for 5 queries of each type (factual, thematic,
comparative, temporal) selected from the retrieval test set.

Results saved to data/evaluation/rag_quality_eval.json with null
scoring fields ready for manual annotation.

Usage:
    python scripts/evaluate_rag_quality.py
"""

from __future__ import annotations

import datetime
import json
import logging
import os
import random
import sys
import time
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
_VENV_PYTHON = _PROJECT_ROOT / ".venv" / "bin" / "python"
_EXPECTED_PREFIX = str(_PROJECT_ROOT / ".venv")
if sys.prefix != _EXPECTED_PREFIX and _VENV_PYTHON.exists():
    os.execv(str(_VENV_PYTHON), [str(_VENV_PYTHON)] + sys.argv)

sys.path.insert(0, str(_PROJECT_ROOT))

from src import config  # noqa: E402
from src.nlp.embedder import embed_query  # noqa: E402
from src.rag.generator import generate_answer  # noqa: E402
from src.rag.indexer import load_bm25_index, load_chroma_collection  # noqa: E402
from src.rag.reranker import rerank  # noqa: E402
from src.rag.retriever import reciprocal_rank_fusion, retrieve_dense, retrieve_sparse  # noqa: E402

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s — %(message)s",
)
logger = logging.getLogger(__name__)

QUERIES_PER_TYPE = 5
QUERY_TYPES = ["factual", "thematic", "comparative", "temporal"]
DENSE_TOP_K = 20
SPARSE_TOP_K = 20
RRF_TOP_N = 10
RERANK_TOP_K = 5

SCORING_INSTRUCTIONS = """\
╔══════════════════════════════════════════════════════════════════╗
║              RAG QUALITY EVALUATION — SCORING GUIDE             ║
╠══════════════════════════════════════════════════════════════════╣
║  faithfulness  1 = answer uses ONLY information from retrieved  ║
║                    chunks (no hallucination)                     ║
║                0 = answer contains information NOT in chunks     ║
║                                                                  ║
║  relevance     1–5: does the answer actually address the query?  ║
║                1 = completely off-topic                          ║
║                3 = partially addresses the query                 ║
║                5 = fully and directly answers the query          ║
║                                                                  ║
║  completeness  1–5: does the answer use all relevant info from   ║
║                     the retrieved chunks?                        ║
║                1 = ignores most relevant retrieved content       ║
║                3 = uses some relevant chunks                     ║
║                5 = synthesises all relevant retrieved content    ║
╚══════════════════════════════════════════════════════════════════╝
"""


def select_queries(all_queries: list[dict], seed: int = 42) -> tuple[list[dict], list[str]]:
    """Select 5 queries per type; note gaps if fewer than 5 exist.

    Args:
        all_queries: Full list of query dicts from the test set.
        seed: Random seed for reproducible selection.

    Returns:
        Tuple of (selected_queries, gap_notes) where gap_notes lists any
        types that had fewer than QUERIES_PER_TYPE available.
    """
    rng = random.Random(seed)
    by_type: dict[str, list[dict]] = {t: [] for t in QUERY_TYPES}
    for q in all_queries:
        qt = q.get("query_type", "")
        if qt in by_type:
            by_type[qt].append(q)

    selected: list[dict] = []
    gaps: list[str] = []

    for qtype in QUERY_TYPES:
        pool = by_type[qtype]
        n_available = len(pool)
        if n_available < QUERIES_PER_TYPE:
            gaps.append(
                f"{qtype}: only {n_available} available (wanted {QUERIES_PER_TYPE})"
            )
            chosen = pool[:]
        else:
            chosen = rng.sample(pool, QUERIES_PER_TYPE)
        selected.extend(chosen)

    return selected, gaps


def run_pipeline(
    query: str,
    collection,
    bm25_index,
    bm25_chunks: list[dict],
    model_dir,
) -> tuple[list, str]:
    """Run hybrid retrieval → rerank → generate for a single query.

    Args:
        query: Natural language query string.
        collection: Loaded ChromaDB collection.
        bm25_index: Loaded BM25Okapi instance.
        bm25_chunks: Ordered chunk dicts matching the BM25 index.
        model_dir: Sentence transformer model path or Hub ID.

    Returns:
        Tuple of (reranked_chunks, generated_answer).
    """
    query_emb = embed_query(query, model_dir=model_dir)

    dense = retrieve_dense(query_emb, collection, top_k=DENSE_TOP_K)
    sparse = retrieve_sparse(query, bm25_index, bm25_chunks, top_k=SPARSE_TOP_K)
    fused = reciprocal_rank_fusion(dense, sparse, top_n=RRF_TOP_N)
    reranked = rerank(query, fused, top_k=RERANK_TOP_K)

    answer = _generate_with_retry(query, reranked)
    return reranked, answer


def _generate_with_retry(query: str, chunks, max_attempts: int = 5) -> str:
    """Call generate_answer with exponential backoff on 503 / ServerError."""
    delay = 10
    for attempt in range(1, max_attempts + 1):
        try:
            return generate_answer(query, chunks)
        except Exception as exc:
            is_503 = "503" in str(exc) or "UNAVAILABLE" in str(exc)
            if is_503 and attempt < max_attempts:
                logger.warning(
                    "Gemini 503 on attempt %d/%d — retrying in %ds …",
                    attempt, max_attempts, delay,
                )
                time.sleep(delay)
                delay = min(delay * 2, 60)
            else:
                raise


def main() -> None:
    """Select queries, run full RAG pipeline, save results, print for scoring."""
    # Load test set
    with open(config.RETRIEVAL_TEST_SET_PATH, encoding="utf-8") as f:
        data = json.load(f)
    all_queries = data["queries"]
    logger.info("Test set: %d queries", len(all_queries))

    selected, gaps = select_queries(all_queries)
    logger.info("Selected %d queries (%d per type)", len(selected), QUERIES_PER_TYPE)
    for note in gaps:
        logger.warning("Type gap — %s", note)

    # Load indexes
    bm25_path = config.VECTORSTORE_DIR / "bm25_index.pkl"
    logger.info("Loading BM25 index from %s …", bm25_path)
    bm25_index, bm25_chunks = load_bm25_index(bm25_path)

    logger.info("Loading ChromaDB collection %r …", config.CHROMA_COLLECTION_NAME)
    collection = load_chroma_collection(config.CHROMADB_DIR, config.CHROMA_COLLECTION_NAME)

    model_dir = config.SENTENCE_TRANSFORMER_PATH

    # Run pipeline for each query
    results: list[dict] = []
    n = len(selected)
    for i, q in enumerate(selected, start=1):
        query_text = q["query"]
        qtype = q["query_type"]
        logger.info("[%d/%d] %s — %s", i, n, qtype, query_text[:80])

        reranked, answer = run_pipeline(
            query_text, collection, bm25_index, bm25_chunks, model_dir
        )

        retrieved_chunks = [
            {
                "chunk_id": c.chunk_id,
                "company": c.ticker,
                "date": c.reference_date,
                "section": c.section,
                "text": c.text,
            }
            for c in reranked
        ]

        results.append({
            "id": i,
            "query": query_text,
            "query_type": qtype,
            "retrieved_chunks": retrieved_chunks,
            "generated_answer": answer,
            "faithfulness": None,
            "relevance": None,
            "completeness": None,
        })

    # Save JSON
    output = {
        "metadata": {
            "n_queries": len(results),
            "queries_per_type": QUERIES_PER_TYPE,
            "gaps": gaps,
            "pipeline": {
                "dense_top_k": DENSE_TOP_K,
                "sparse_top_k": SPARSE_TOP_K,
                "rrf_top_n": RRF_TOP_N,
                "rerank_top_k": RERANK_TOP_K,
            },
            "generated_at": datetime.datetime.now().isoformat(),
        },
        "results": results,
    }

    out_path = config.EVALUATION_DIR / "rag_quality_eval.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)
    logger.info("Results saved to %s", out_path)

    # Print for manual scoring
    print(SCORING_INSTRUCTIONS)
    if gaps:
        print("⚠  TYPE GAPS:")
        for g in gaps:
            print(f"   {g}")
        print()

    for r in results:
        print("═" * 72)
        print(f"[{r['id']:02d}/{n}]  TYPE: {r['query_type'].upper()}")
        print(f"QUERY: {r['query']}")
        print()
        print("RETRIEVED CHUNKS:")
        for j, ch in enumerate(r["retrieved_chunks"], start=1):
            print(f"  [{j}] {ch['company']} | {ch['date']} | {ch['section']}")
            print(f"       {ch['text'][:120].replace(chr(10), ' ')}…")
        print()
        print("ANSWER:")
        print(r["generated_answer"])
        print()
        print("SCORES (fill in rag_quality_eval.json):")
        print("  faithfulness: __   relevance: __   completeness: __")
        print()

    print("═" * 72)
    print(f"\nSaved to: {out_path}")
    print(f"Fill in faithfulness/relevance/completeness for all {n} entries.")


if __name__ == "__main__":
    main()
