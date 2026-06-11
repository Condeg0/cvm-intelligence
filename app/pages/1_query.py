"""RAG query interface — ask natural-language questions about CVM filings.

Accepts a free-text query and optional metadata filters (ticker, section),
runs the full hybrid retrieval + reranking + generation pipeline, and
displays the answer with source citations.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import streamlit as st

_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(_ROOT))

st.set_page_config(page_title="Query Filings", page_icon="🔍", layout="wide")

# ---------------------------------------------------------------------------
# Cloud-safe availability check
# ---------------------------------------------------------------------------

def _rag_available() -> bool:
    """Return True only if ChromaDB and BM25 index exist on disk."""
    from src import config
    chroma_ok = (config.CHROMADB_DIR / "chroma.sqlite3").exists()
    bm25_ok   = (config.VECTORSTORE_DIR / "bm25_index.pkl").exists()
    return chroma_ok and bm25_ok


# ---------------------------------------------------------------------------
# Cached resource loaders (loaded once per session)
# ---------------------------------------------------------------------------

@st.cache_resource(show_spinner="Loading vector indexes…")
def _load_indexes():
    """Load ChromaDB collection and BM25 index from disk."""
    from src import config
    from src.rag.indexer import load_chroma_collection, load_bm25_index

    collection = load_chroma_collection(config.CHROMADB_DIR, config.CHROMA_COLLECTION_NAME)
    bm25_path = config.VECTORSTORE_DIR / "bm25_index.pkl"
    bm25_index, bm25_chunks = load_bm25_index(bm25_path)
    return collection, bm25_index, bm25_chunks


@st.cache_resource(show_spinner="Loading embedding model…")
def _load_model_dir() -> Path | str:
    from src import config
    return config.SENTENCE_TRANSFORMER_PATH


# ---------------------------------------------------------------------------
# Page layout
# ---------------------------------------------------------------------------

st.title("Query Filings")
st.markdown("Ask a question about Brazilian public company filings.")

if not _rag_available():
    st.info(
        "**RAG pipeline not available in this deployment.**\n\n"
        "The ChromaDB vector index (1.8 GB) and BM25 index require a high-memory "
        "environment and are not included in the Streamlit Community Cloud deployment.\n\n"
        "To run the full pipeline locally:\n"
        "```bash\n"
        "python scripts/run_indexing.py\n"
        "streamlit run app/app.py\n"
        "```\n\n"
        "All other pages (Financial Dashboard, Sentiment Timeline, Evaluation) "
        "are fully available and use only the SQLite database."
    )
    st.stop()

with st.sidebar:
    st.header("Filters")
    ticker_filter = st.text_input("Ticker (optional)", placeholder="PETR4").strip().upper()
    section_filter = st.selectbox(
        "Section",
        ["All", "Relatório da Administração", "Notas Explicativas"],
    )
    st.divider()
    st.caption("Retrieval: fine-tuned BERTimbau + BM25 + RRF + cross-encoder reranker")

query = st.text_area(
    "Your question",
    height=100,
    placeholder="Qual foi o EBITDA da Petrobras no 3T23?",
)

if st.button("Search", type="primary"):
    if not query.strip():
        st.warning("Please enter a question.")
        st.stop()

    try:
        collection, bm25_index, bm25_chunks = _load_indexes()
    except Exception as exc:
        st.error(f"Vector indexes not available: {exc}. Run `scripts/run_indexing.py` first.")
        st.stop()

    model_dir = _load_model_dir()

    with st.spinner("Retrieving relevant passages…"):
        from src.nlp.embedder import embed_query
        from src.rag.retriever import retrieve_dense, retrieve_sparse, reciprocal_rank_fusion
        from src.rag.reranker import rerank

        chroma_filter = None
        if ticker_filter:
            chroma_filter = {"ticker": ticker_filter}
        if section_filter != "All":
            section_where = {"section_name": section_filter}
            chroma_filter = {**chroma_filter, **section_where} if chroma_filter else section_where

        query_emb = embed_query(query, model_dir=model_dir)
        dense = retrieve_dense(query_emb, collection, top_k=20, filters=chroma_filter)
        sparse = retrieve_sparse(query, bm25_index, bm25_chunks, top_k=20)

        # Post-filter sparse results by ticker/section if needed
        if ticker_filter:
            sparse = [c for c in sparse if c.ticker == ticker_filter]
        if section_filter != "All":
            sparse = [c for c in sparse if c.section == section_filter]

        fused = reciprocal_rank_fusion(dense, sparse, k=60, top_n=10)
        reranked = rerank(query, fused, top_k=5)

    with st.spinner("Generating answer via Gemini…"):
        from src.rag.generator import generate_answer
        try:
            answer = generate_answer(query, reranked)
        except Exception as exc:
            st.error(f"Gemini API error: {exc}")
            answer = None

    # ---------------------------------------------------------------------------
    # Display answer
    # ---------------------------------------------------------------------------
    if answer:
        st.subheader("Answer")
        st.markdown(answer)

    st.divider()
    st.subheader(f"Sources ({len(reranked)} passages)")

    for i, chunk in enumerate(reranked, start=1):
        label = f"[{i}] {chunk.ticker} — {chunk.reference_date} | {chunk.section}"
        with st.expander(label):
            st.markdown(f"**Chunk ID:** `{chunk.chunk_id}`")
            st.markdown(f"**Filing:** `{chunk.filing_id}`")
            st.markdown(f"**RRF score:** {chunk.rrf_score:.4f}")
            extra = []
            if chunk.dense_rank is not None:
                extra.append(f"dense rank {chunk.dense_rank + 1}")
            if chunk.sparse_rank is not None:
                extra.append(f"sparse rank {chunk.sparse_rank + 1}")
            if extra:
                st.markdown(f"**Retrieved by:** {', '.join(extra)}")
            st.text_area("Text", chunk.text, height=200, disabled=True, key=f"chunk_{i}")
