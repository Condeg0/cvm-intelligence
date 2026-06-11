"""Embedding comparison: fine-tuned BERTimbau vs base vs multilingual.

Evaluates three sentence transformer models on the retrieval test set
(94 queries, 97,138-chunk corpus) using dense-only retrieval to isolate
embedding quality from BM25 and reranking effects.

Fine-tuned model: uses pre-built ChromaDB embeddings (full 97k corpus).
Base models: use a 5,000-chunk subsample with in-memory cosine search.
All relevant chunks are guaranteed present in the subsample so Recall is
computable. All three models also run on the same 5k subsample for a strict
apples-to-apples comparison.

Results saved to data/evaluation/embedding_comparison.json.

Usage:
    python scripts/run_embedding_comparison.py [--subsample-size N] [--seed S]
"""

from __future__ import annotations

import argparse
import datetime
import json
import logging
import math
import os
import random
import sqlite3
import sys
from pathlib import Path

import numpy as np

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
_VENV_PYTHON = _PROJECT_ROOT / ".venv" / "bin" / "python"
_EXPECTED_PREFIX = str(_PROJECT_ROOT / ".venv")
if sys.prefix != _EXPECTED_PREFIX and _VENV_PYTHON.exists():
    os.execv(str(_VENV_PYTHON), [str(_VENV_PYTHON)] + sys.argv)

sys.path.insert(0, str(_PROJECT_ROOT))

from src import config  # noqa: E402
from src.rag.indexer import load_chroma_collection  # noqa: E402
from src.rag.retriever import retrieve_dense  # noqa: E402

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s — %(message)s",
)
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# IR metric functions (copied verbatim from notebook 05 cells)
# ---------------------------------------------------------------------------

def recall_at_k(retrieved_ids: list[str], relevant_ids: set[str], k: int) -> float:
    """Proportion of relevant docs found in the top-k retrieved."""
    if not relevant_ids:
        return 0.0
    return len(set(retrieved_ids[:k]) & relevant_ids) / len(relevant_ids)


def reciprocal_rank(retrieved_ids: list[str], relevant_ids: set[str]) -> float:
    """Reciprocal of the rank of the first relevant document."""
    for rank, doc_id in enumerate(retrieved_ids, start=1):
        if doc_id in relevant_ids:
            return 1.0 / rank
    return 0.0


def ndcg_at_k(retrieved_ids: list[str], relevant_ids: set[str], k: int) -> float:
    """Normalised Discounted Cumulative Gain at k."""
    dcg = sum(
        1.0 / math.log2(r + 2)
        for r, doc_id in enumerate(retrieved_ids[:k])
        if doc_id in relevant_ids
    )
    ideal = sum(1.0 / math.log2(r + 2) for r in range(min(len(relevant_ids), k)))
    return dcg / ideal if ideal else 0.0


def aggregate_metrics(per_query: list[dict]) -> dict[str, float]:
    """Mean Recall@5, Recall@10, MRR, NDCG@10 over all queries."""
    keys = ["Recall@5", "Recall@10", "MRR", "NDCG@10"]
    return {k: float(np.mean([q[k] for q in per_query])) for k in keys}


# ---------------------------------------------------------------------------
# Evaluation helpers
# ---------------------------------------------------------------------------

def eval_per_query(top_ids: list[str], relevant: set[str]) -> dict[str, float]:
    """Compute all four metrics for a single query result."""
    return {
        "Recall@5":  recall_at_k(top_ids, relevant, 5),
        "Recall@10": recall_at_k(top_ids, relevant, 10),
        "MRR":       reciprocal_rank(top_ids, relevant),
        "NDCG@10":   ndcg_at_k(top_ids, relevant, 10),
    }


def evaluate_in_memory(
    queries: list[dict],
    chunk_ids: list[str],
    chunk_embs: np.ndarray,
    query_embs: np.ndarray,
) -> dict[str, float]:
    """Dense retrieval over an in-memory embedding matrix.

    Args:
        queries: List of test-set query dicts (must have ``relevant_chunk_ids``).
        chunk_ids: Ordered list of chunk IDs corresponding to rows in chunk_embs.
        chunk_embs: Float32 array of shape ``(N, dim)``, L2-normalised.
        query_embs: Float32 array of shape ``(Q, dim)``, L2-normalised.

    Returns:
        Dict with mean Recall@5, Recall@10, MRR, NDCG@10.
    """
    chunk_ids_arr = np.array(chunk_ids)
    per_query = []
    for i, q in enumerate(queries):
        relevant = set(q["relevant_chunk_ids"])
        scores = chunk_embs @ query_embs[i]          # (N,) dot products
        top_idx = np.argsort(scores)[::-1][:10]
        top_ids = chunk_ids_arr[top_idx].tolist()
        per_query.append(eval_per_query(top_ids, relevant))
    return aggregate_metrics(per_query)


def evaluate_on_chromadb(
    queries: list[dict],
    collection,
    query_embs: np.ndarray,
) -> dict[str, float]:
    """Dense retrieval using the pre-built ChromaDB index.

    Args:
        queries: List of test-set query dicts.
        collection: ChromaDB collection with the fine-tuned model's embeddings.
        query_embs: Pre-encoded query embeddings (fine-tuned model, L2-normalised).

    Returns:
        Dict with mean Recall@5, Recall@10, MRR, NDCG@10.
    """
    per_query = []
    for i, q in enumerate(queries):
        relevant = set(q["relevant_chunk_ids"])
        chunks = retrieve_dense(query_embs[i], collection, top_k=10)
        top_ids = [c.chunk_id for c in chunks]
        per_query.append(eval_per_query(top_ids, relevant))
    return aggregate_metrics(per_query)


# ---------------------------------------------------------------------------
# Subsample construction
# ---------------------------------------------------------------------------

def build_subsample(
    queries: list[dict],
    db_path: Path,
    target_size: int,
    seed: int,
) -> tuple[list[str], list[str]]:
    """Build a subsample of chunk_ids and texts for in-memory evaluation.

    All relevant chunks from every query are guaranteed to be present.
    The remainder is filled with random chunks from the SQLite database.

    Args:
        queries: Test-set query list.
        db_path: Path to cvm_metrics.db.
        target_size: Total number of chunks in subsample.
        seed: Random seed for reproducibility.

    Returns:
        Tuple of (chunk_ids, chunk_texts) both length target_size.
    """
    random.seed(seed)

    required_ids = set()
    for q in queries:
        required_ids.update(q["relevant_chunk_ids"])
    logger.info("Required (relevant) chunks: %d", len(required_ids))

    n_extra = max(0, target_size - len(required_ids))
    conn = sqlite3.connect(db_path)

    req_rows = conn.execute(
        f"SELECT chunk_id, chunk_text FROM chunks "
        f"WHERE chunk_id IN ({','.join('?' * len(required_ids))})",
        tuple(required_ids),
    ).fetchall()

    extra_rows = conn.execute(
        f"SELECT chunk_id, chunk_text FROM chunks "
        f"WHERE chunk_text IS NOT NULL AND chunk_text != '' "
        f"AND chunk_id NOT IN ({','.join('?' * len(required_ids))}) "
        f"ORDER BY RANDOM() LIMIT ?",
        (*required_ids, n_extra),
    ).fetchall()
    conn.close()

    all_rows = req_rows + extra_rows
    ids   = [r[0] for r in all_rows]
    texts = [r[1] for r in all_rows]
    logger.info("Subsample built: %d chunks (%d required + %d random)", len(ids), len(req_rows), len(extra_rows))
    return ids, texts


# ---------------------------------------------------------------------------
# Per-model evaluation
# ---------------------------------------------------------------------------

def run_one_model(
    model_name: str,
    label: str,
    queries: list[dict],
    subsample_ids: list[str],
    subsample_texts: list[str],
    device: str,
) -> dict:
    """Load a model, embed the subsample and queries, evaluate, free VRAM.

    Args:
        model_name: HuggingFace model ID or local path string.
        label: Human-readable label for output JSON.
        queries: Test-set query list.
        subsample_ids: Chunk IDs for the subsample.
        subsample_texts: Corresponding chunk texts.
        device: ``"cuda"`` or ``"cpu"``.

    Returns:
        Dict with ``label`` and the four metric keys.
    """
    import torch
    from sentence_transformers import SentenceTransformer

    logger.info("Loading model: %s", model_name)
    model = SentenceTransformer(model_name, device=device)
    model.max_seq_length = config.MAX_SEQ_LENGTH

    logger.info("Embedding %d subsample chunks…", len(subsample_texts))
    chunk_embs = model.encode(
        subsample_texts,
        batch_size=64,
        show_progress_bar=True,
        normalize_embeddings=True,
        convert_to_numpy=True,
    ).astype(np.float32)

    logger.info("Embedding %d queries…", len(queries))
    query_embs = model.encode(
        [q["query"] for q in queries],
        batch_size=64,
        show_progress_bar=True,
        normalize_embeddings=True,
        convert_to_numpy=True,
    ).astype(np.float32)

    metrics = evaluate_in_memory(queries, subsample_ids, chunk_embs, query_embs)
    logger.info("%s — %s", label, metrics)

    del model, chunk_embs, query_embs
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        logger.info("VRAM freed after %s", model_name)

    return {"label": label, **metrics}


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    """Parse CLI arguments."""
    parser = argparse.ArgumentParser(description="Embedding comparison evaluation")
    parser.add_argument("--subsample-size", type=int, default=5_000,
                        help="Total subsample size including required relevant chunks (default: 5000)")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed for subsample construction (default: 42)")
    parser.add_argument("--device", default="cuda",
                        help="PyTorch device (default: cuda, falls back to cpu if unavailable)")
    return parser.parse_args()


def main() -> None:
    """Run embedding comparison and save results to JSON."""
    import torch

    args = parse_args()

    device = args.device
    if device == "cuda" and not torch.cuda.is_available():
        logger.warning("CUDA not available — falling back to CPU (will be slow)")
        device = "cpu"

    # Load test set
    with open(config.RETRIEVAL_TEST_SET_PATH) as f:
        data = json.load(f)
    queries = data["queries"]
    logger.info("Test set: %d queries", len(queries))

    # Build subsample
    subsample_ids, subsample_texts = build_subsample(
        queries, config.DB_PATH, args.subsample_size, args.seed
    )

    required_ids = set()
    for q in queries:
        required_ids.update(q["relevant_chunk_ids"])

    results: dict[str, dict] = {}

    # ── 1. Fine-tuned model: subsample + full ChromaDB ──────────────────────
    from sentence_transformers import SentenceTransformer

    ft_model_path = str(config.SENTENCE_TRANSFORMER_PATH)
    logger.info("Loading fine-tuned model: %s", ft_model_path)
    ft_model = SentenceTransformer(ft_model_path, device=device)
    ft_model.max_seq_length = config.MAX_SEQ_LENGTH

    logger.info("Embedding subsample with fine-tuned model…")
    ft_chunk_embs = ft_model.encode(
        subsample_texts, batch_size=64, show_progress_bar=True,
        normalize_embeddings=True, convert_to_numpy=True,
    ).astype(np.float32)

    logger.info("Embedding queries with fine-tuned model…")
    ft_query_embs = ft_model.encode(
        [q["query"] for q in queries], batch_size=64, show_progress_bar=True,
        normalize_embeddings=True, convert_to_numpy=True,
    ).astype(np.float32)

    results["fine_tuned_subsample"] = {
        "label": "Fine-tuned BERTimbau (5k subsample)",
        **evaluate_in_memory(queries, subsample_ids, ft_chunk_embs, ft_query_embs),
    }
    logger.info("Fine-tuned subsample: %s", results["fine_tuned_subsample"])

    # Full ChromaDB evaluation (reuse already-computed query embeddings)
    logger.info("Evaluating fine-tuned on full ChromaDB corpus…")
    collection = load_chroma_collection(config.CHROMADB_DIR, config.CHROMA_COLLECTION_NAME)
    results["fine_tuned_full"] = {
        "label": "Fine-tuned BERTimbau (97k corpus)",
        **evaluate_on_chromadb(queries, collection, ft_query_embs),
    }
    logger.info("Fine-tuned full corpus: %s", results["fine_tuned_full"])

    del ft_model, ft_chunk_embs, ft_query_embs
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    logger.info("VRAM freed after fine-tuned model")

    # ── 2. Base BERTimbau ────────────────────────────────────────────────────
    results["base_bertimbau"] = run_one_model(
        model_name=config.BERT_MODEL_NAME,
        label="Base BERTimbau (5k subsample)",
        queries=queries,
        subsample_ids=subsample_ids,
        subsample_texts=subsample_texts,
        device=device,
    )

    # ── 3. Multilingual MiniLM ───────────────────────────────────────────────
    results["multilingual_minilm"] = run_one_model(
        model_name="paraphrase-multilingual-MiniLM-L12-v2",
        label="paraphrase-multilingual-MiniLM-L12-v2 (5k subsample)",
        queries=queries,
        subsample_ids=subsample_ids,
        subsample_texts=subsample_texts,
        device=device,
    )

    # ── Save results ─────────────────────────────────────────────────────────
    output = {
        "metadata": {
            "n_queries": len(queries),
            "subsample_size": len(subsample_ids),
            "n_relevant_chunks": len(required_ids),
            "subsample_seed": args.seed,
            "device": device,
            "generated_at": datetime.datetime.now().isoformat(),
        },
        "results": results,
    }

    out_path = config.EVALUATION_DIR / "embedding_comparison.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)
    logger.info("Results saved to %s", out_path)

    # Print summary table
    metric_keys = ["Recall@5", "Recall@10", "MRR", "NDCG@10"]
    header = f"{'Model':<52} " + "  ".join(f"{k:>10}" for k in metric_keys)
    print("\n" + header)
    print("-" * len(header))
    order = ["fine_tuned_full", "fine_tuned_subsample", "base_bertimbau", "multilingual_minilm"]
    for key in order:
        r = results[key]
        row = f"{r['label']:<52} " + "  ".join(f"{r[k]:>10.4f}" for k in metric_keys)
        print(row)


if __name__ == "__main__":
    main()
