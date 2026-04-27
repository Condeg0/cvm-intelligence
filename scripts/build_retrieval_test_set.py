"""Generate the retrieval test set for Phase 5 ablation evaluation.

Samples chunks from the processed corpus, sends them to Gemini to generate
natural-language queries, then writes the labeled test set to
data/evaluation/retrieval_test_set.json.

Usage:
    python scripts/build_retrieval_test_set.py [--n-queries 100]
"""

from __future__ import annotations

import argparse
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

logging.basicConfig(
    level=config.LOG_LEVEL,
    format="%(asctime)s %(levelname)s %(name)s — %(message)s",
)
logger = logging.getLogger(__name__)

# Query types to generate, with how many of each
QUERY_TYPE_COUNTS = {
    "factual": 40,
    "thematic": 25,
    "comparative": 20,
    "temporal": 15,
}

QUERY_GENERATION_PROMPT = """\
Você está construindo um conjunto de avaliação para um sistema RAG (Retrieval-Augmented Generation) \
sobre relatórios anuais e trimestrais de empresas brasileiras de capital aberto.

Dado o trecho abaixo de um relatório da {company} ({filing_type} de {date}), gere UMA pergunta \
do tipo "{query_type}" que seja respondida por esse trecho.

Tipos:
- factual: pergunta específica sobre um fato, número ou evento mencionado no trecho
- thematic: pergunta sobre um tema ou estratégia geral discutida
- comparative: pergunta que compare períodos, empresas ou métricas
- temporal: pergunta sobre evolução ou tendência ao longo do tempo

Responda SOMENTE com a pergunta em português. Sem explicações, sem aspas.

Trecho:
{chunk_text}

Pergunta ({query_type}):"""


def load_all_chunk_dicts(chunks_dir: Path) -> list[dict]:
    """Load all chunks as raw dicts from disk."""
    all_chunks: list[dict] = []
    for json_path in sorted(chunks_dir.glob("*.json")):
        try:
            with open(json_path, encoding="utf-8") as f:
                all_chunks.extend(json.load(f))
        except Exception as exc:
            logger.warning("Skipping %s — %s", json_path.name, exc)
    return all_chunks


def _filing_meta(filing_id: str) -> dict[str, str]:
    """Parse ticker, filing_type, date from a filing_id string."""
    parts = filing_id.split("_")
    return {
        "company": parts[0] if parts else "?",
        "filing_type": parts[1] if len(parts) > 1 else "?",
        "date": parts[2] if len(parts) > 2 else "?",
    }


def call_gemini(prompt: str, api_key: str) -> str:
    """Call Gemini 2.5 Flash and return the response text."""
    from google import genai  # type: ignore

    client = genai.Client(api_key=api_key)
    response = client.models.generate_content(
        model=config.GEMINI_MODEL,
        contents=prompt,
    )
    return response.text.strip()


def generate_test_set(
    all_chunks: list[dict],
    n_queries: int,
    api_key: str,
    seed: int = 42,
) -> dict:
    """Sample chunks and generate queries via Gemini.

    For each query, the ``relevant_chunk_ids`` list contains the chunk that
    was used to generate the query (guaranteed relevant) plus any adjacent
    chunks from the same filing/section (likely relevant).

    Args:
        all_chunks: All chunk dicts loaded from disk.
        n_queries: Total number of queries to generate.
        api_key: Gemini API key.
        seed: Random seed for reproducibility.

    Returns:
        Dict with key ``queries``, a list of query dicts.
    """
    rng = random.Random(seed)

    # Filter to prose chunks (non-trivially short)
    prose_chunks = [
        c for c in all_chunks
        if len(c.get("text", "")) >= 200
        and c.get("section_type", "") in {
            "Relatório da Administração",
            "Notas Explicativas",
        }
    ]
    logger.info("Prose chunk pool: %d chunks", len(prose_chunks))

    # Build index: filing_id+section → list of chunk_ids (ordered by chunk_index)
    from collections import defaultdict
    section_map: dict[str, list[str]] = defaultdict(list)
    chunk_by_id: dict[str, dict] = {}
    for c in all_chunks:
        key = f"{c['filing_id']}|{c.get('section_type', '')}"
        section_map[key].append(c["chunk_id"])
        chunk_by_id[c["chunk_id"]] = c
    # Sort each section's chunks by index
    for key in section_map:
        section_map[key].sort(key=lambda cid: chunk_by_id[cid].get("chunk_index", 0))

    # Distribute queries across types
    type_schedule: list[str] = []
    for qtype, count in QUERY_TYPE_COUNTS.items():
        adjusted = round(count * n_queries / 100)
        type_schedule.extend([qtype] * adjusted)
    rng.shuffle(type_schedule)
    type_schedule = type_schedule[:n_queries]

    queries: list[dict] = []
    selected_chunks = rng.sample(prose_chunks, min(n_queries, len(prose_chunks)))

    for i, (chunk, qtype) in enumerate(zip(selected_chunks, type_schedule)):
        meta = _filing_meta(chunk["filing_id"])
        prompt = QUERY_GENERATION_PROMPT.format(
            company=meta["company"],
            filing_type=meta["filing_type"],
            date=meta["date"],
            query_type=qtype,
            chunk_text=chunk["text"][:1500],  # cap to avoid token limits
        )
        try:
            query_text = call_gemini(prompt, api_key)
        except Exception as exc:
            logger.warning("Gemini call failed for chunk %s — %s", chunk["chunk_id"], exc)
            continue

        # Build relevant_chunk_ids: anchor chunk + adjacent chunks in same section
        section_key = f"{chunk['filing_id']}|{chunk.get('section_type', '')}"
        section_chunk_ids = section_map.get(section_key, [])
        try:
            pos = section_chunk_ids.index(chunk["chunk_id"])
        except ValueError:
            pos = -1

        relevant_ids = [chunk["chunk_id"]]
        if pos > 0:
            relevant_ids.append(section_chunk_ids[pos - 1])
        if 0 <= pos < len(section_chunk_ids) - 1:
            relevant_ids.append(section_chunk_ids[pos + 1])

        queries.append({
            "query": query_text,
            "query_type": qtype,
            "filing_id": chunk["filing_id"],
            "anchor_chunk_id": chunk["chunk_id"],
            "relevant_chunk_ids": relevant_ids,
        })

        if (i + 1) % 10 == 0:
            logger.info("Generated %d / %d queries …", i + 1, n_queries)

        # Rate-limit: ~2 req/s
        time.sleep(0.5)

    return {"queries": queries}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build retrieval test set")
    parser.add_argument("--n-queries", type=int, default=100)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    api_key = os.getenv("GEMINI_API_KEY") or config.GEMINI_API_KEY
    if not api_key:
        logger.error("GEMINI_API_KEY not set.")
        sys.exit(1)

    all_chunks = load_all_chunk_dicts(config.CHUNKS_DIR)
    logger.info("Total chunks: %d", len(all_chunks))

    test_set = generate_test_set(all_chunks, args.n_queries, api_key, seed=args.seed)
    logger.info("Generated %d queries", len(test_set["queries"]))

    out_path = config.RETRIEVAL_TEST_SET_PATH
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(test_set, f, indent=2, ensure_ascii=False)
    logger.info("Test set saved to %s", out_path)


if __name__ == "__main__":
    main()
