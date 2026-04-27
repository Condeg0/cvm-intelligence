# src/rag/ — Hybrid Retrieval + Generation Pipeline

## Module Purpose

Full RAG pipeline: hybrid retrieval (dense + sparse) with cross-encoder reranking and Gemini API generation. This is the user-facing intelligence layer.

## Architecture

```
User Query
    │
    ├──► Dense Search (fine-tuned BERTimbau via ChromaDB) → top 20
    │
    ├──► Sparse Search (BM25 via rank_bm25) → top 20
    │
    ▼
Reciprocal Rank Fusion (merge + deduplicate) → top 10
    │
    ▼
Cross-Encoder Reranker → top 5
    │
    ▼
Context Assembly + Gemini API Generation → Answer
```

### Files

- `indexer.py` — Embeds all chunks and stores in ChromaDB + builds BM25 index. Run once during setup.
- `retriever.py` — Core retrieval: takes a query, runs both dense and sparse search, fuses with RRF.
- `reranker.py` — Cross-encoder scoring of top-k candidates.
- `generator.py` — Assembles context from retrieved chunks + metadata, sends to Gemini API, returns generated answer.

## Indexer

### ChromaDB Setup
```python
import chromadb

client = chromadb.PersistentClient(path="vectorstore/chromadb")
collection = client.get_or_create_collection(
    name="cvm_filings",
    metadata={"hnsw:space": "cosine"}
)
```

### Metadata per chunk
```python
{
    "company_ticker": "PETR4",
    "company_name": "Petrobras",
    "filing_type": "ITR",          # or "DFP"
    "filing_date": "2024-09-30",   # reference date
    "section_name": "Relatório da Administração",
    "chunk_index": 3,
    "sentiment_label": "optimistic",
    "sentiment_score": 0.85,
}
```

### BM25 Index
```python
from rank_bm25 import BM25Okapi

# Tokenize chunks (simple whitespace + lowercase)
tokenized_chunks = [chunk.lower().split() for chunk in all_chunk_texts]
bm25_index = BM25Okapi(tokenized_chunks)

# Save chunk IDs alongside for mapping back to ChromaDB results
```

Persist BM25 index with `pickle` alongside the chunk ID mapping.

## Retriever — Hybrid Search + RRF

### Reciprocal Rank Fusion

```python
def reciprocal_rank_fusion(
    ranked_lists: list[list[str]],  # each list is chunk_ids ordered by rank
    k: int = 60,
) -> list[tuple[str, float]]:
    """Fuse multiple ranked lists using RRF. Returns (chunk_id, score) sorted by score desc."""
    scores = {}
    for ranked_list in ranked_lists:
        for rank, chunk_id in enumerate(ranked_list):
            scores[chunk_id] = scores.get(chunk_id, 0) + 1 / (k + rank + 1)
    return sorted(scores.items(), key=lambda x: x[1], reverse=True)
```

### Metadata Filtering

The retriever accepts optional filters passed through to ChromaDB:
- `company_ticker` — filter by specific company
- `filing_date_range` — filter by date range
- `section_name` — filter by section type
- `sentiment_label` — filter by sentiment

BM25 filtering is done post-retrieval (filter the BM25 results by the same metadata from the chunk mapping).

## Reranker

### Cross-Encoder Setup
```python
from sentence_transformers import CrossEncoder

reranker = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")

# Score (query, chunk) pairs
pairs = [(query, chunk_text) for chunk_text in candidate_texts]
scores = reranker.predict(pairs)

# Sort by score, take top 5
```

~80MB model, runs on CPU at ~200ms for 10 candidates. Acceptable latency.

## Generator

### Gemini API Call

```python
import google.generativeai as genai

SYSTEM_PROMPT = """You are a financial analyst assistant specializing in Brazilian public companies.
Answer the user's question based ONLY on the provided filing excerpts.
If the excerpts do not contain sufficient information, say so explicitly.
Always cite which company and filing period each claim comes from.
Respond in the same language as the user's question."""

def generate_answer(query: str, chunks: list[RetrievedChunk]) -> str:
    context = format_chunks_with_metadata(chunks)
    prompt = f"{SYSTEM_PROMPT}\n\nEXCERPTS:\n{context}\n\nUSER QUESTION: {query}"

    model = genai.GenerativeModel("models/gemini-2.5-flash")
    response = model.generate_content(prompt)
    return response.text
```

### Context Assembly

Format each chunk with its metadata so the model can cite sources:

```
[Source: PETR4 — Petrobras | ITR Q3 2024 | Relatório da Administração]
{chunk_text}
---
[Source: VALE3 — Vale | DFP 2023 | Notas Explicativas]
{chunk_text}
---
```

## Evaluation

### Retrieval Test Set
100 queries in `data/evaluation/retrieval_test_set.json`:

```json
{
    "queries": [
        {
            "query": "Qual foi o EBITDA da Petrobras no Q3 2024?",
            "query_type": "factual",
            "relevant_chunk_ids": ["chunk_001", "chunk_002", "chunk_003"]
        }
    ]
}
```

Query types: factual, thematic, comparative, temporal.

### Ablation Study

Run retrieval metrics for each configuration and report in a table:

| Configuration | Recall@5 | Recall@10 | MRR | NDCG@10 |
|---|---|---|---|---|
| Fine-tuned BERTimbau (dense only) | | | | |
| BM25 (sparse only) | | | | |
| Hybrid (dense + sparse + RRF) | | | | |
| Hybrid + cross-encoder reranker | | | | |

### End-to-End RAG Evaluation

For 20 representative queries, manually score generated answers:
- Faithfulness (binary): answer only uses information from chunks
- Relevance (1–5): answer addresses the query
- Completeness (1–5): answer uses all relevant chunks

## Performance Budget

| Component | Latency (CPU) |
|---|---|
| Query embedding (bi-encoder) | ~50ms |
| ChromaDB search (100K vectors) | ~10ms |
| BM25 search | ~5ms |
| RRF fusion | ~1ms |
| Cross-encoder reranking (10 candidates) | ~200ms |
| Gemini API generation | ~1–3s |
| **Total** | **~2–4s** |
