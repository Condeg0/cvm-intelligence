"""ChromaDB and BM25 indexing for the hybrid retrieval pipeline.

Embeds all parsed chunks, stores them in a persisted ChromaDB collection
with metadata (ticker, date, section, sentiment), and builds an in-memory
BM25 index over the same corpus.
"""

from __future__ import annotations

import logging
import pickle
from pathlib import Path

logger = logging.getLogger(__name__)

# Separate batch size for ChromaDB upsert calls (independent of embedding batch)
_CHROMA_UPSERT_BATCH: int = 256


def _section_value(section_type: object) -> str:
    """Extract string value from a SectionType enum or plain string."""
    if hasattr(section_type, "value"):
        return str(section_type.value)
    return str(section_type)


def _parse_filing_id(filing_id: str) -> dict[str, str]:
    """Parse ticker, filing_type, and reference_date from a filing_id string.

    Filing IDs follow the pattern ``TICKER_TYPE_DATE_VERSION``,
    e.g. ``PETR4_DFP_2023-12-31_1``.
    """
    parts = filing_id.split("_")
    return {
        "ticker": parts[0] if len(parts) >= 1 else "",
        "filing_type": parts[1] if len(parts) >= 2 else "",
        "reference_date": parts[2] if len(parts) >= 3 else "",
    }


def build_chroma_index(
    chunks: list,
    chroma_dir: Path,
    collection_name: str,
    model_dir: Path | None = None,
    batch_size: int = 32,
    reindex: bool = True,
) -> None:
    """Embed chunks and store them in a persisted ChromaDB collection.

    Each chunk is stored with metadata fields: ``ticker``, ``filing_type``,
    ``reference_date``, ``section_name``, ``filing_id``, ``chunk_index``,
    and ``token_count``.

    Args:
        chunks: Parsed and chunked filing text. Each element must expose
            ``chunk_id``, ``filing_id``, ``section_type``, ``text``,
            ``token_count``, and ``chunk_index`` attributes. Accepts both
            :class:`~src.parsing.chunker.Chunk` objects and plain proxy objects.
        chroma_dir: Directory for ChromaDB persistence (created if absent).
        collection_name: Name of the ChromaDB collection to create/update.
        model_dir: Optional path to fine-tuned sentence-transformer model.
            Falls back to the base BERTimbau model if ``None`` or missing.
        batch_size: Embedding batch size forwarded to :func:`embedder.embed_chunks`.
        reindex: Drop and recreate the collection before inserting. Set to
            ``False`` to add new documents without clearing existing ones.
    """
    import chromadb
    from src.nlp.embedder import embed_chunks

    chroma_dir.mkdir(parents=True, exist_ok=True)
    client = chromadb.PersistentClient(path=str(chroma_dir))

    if reindex:
        try:
            client.delete_collection(collection_name)
            logger.info("Dropped existing collection %r", collection_name)
        except Exception:
            pass

    collection = client.get_or_create_collection(
        name=collection_name,
        metadata={"hnsw:space": "cosine"},
    )

    texts = [c.text for c in chunks]
    logger.info("Embedding %d chunks in batches of %d …", len(texts), batch_size)
    embeddings = embed_chunks(texts, model_dir=model_dir, batch_size=batch_size)

    # Upsert into ChromaDB in sub-batches to avoid memory spikes
    ids: list[str] = []
    emb_list: list[list[float]] = []
    docs: list[str] = []
    metas: list[dict] = []

    for i, (chunk, emb) in enumerate(zip(chunks, embeddings)):
        info = _parse_filing_id(chunk.filing_id)
        meta: dict = {
            "filing_id": chunk.filing_id,
            "ticker": info["ticker"],
            "filing_type": info["filing_type"],
            "reference_date": info["reference_date"],
            "section_name": _section_value(chunk.section_type),
            "chunk_index": int(chunk.chunk_index),
            "token_count": int(chunk.token_count),
        }
        ids.append(chunk.chunk_id)
        emb_list.append(emb.tolist())
        docs.append(chunk.text)
        metas.append(meta)

        flush = len(ids) >= _CHROMA_UPSERT_BATCH or i == len(chunks) - 1
        if flush and ids:
            collection.upsert(
                ids=ids,
                embeddings=emb_list,
                documents=docs,
                metadatas=metas,
            )
            logger.debug("Upserted %d chunks (up to index %d)", len(ids), i)
            ids, emb_list, docs, metas = [], [], [], []

    logger.info(
        "ChromaDB collection %r: %d total documents indexed",
        collection_name,
        collection.count(),
    )


def build_bm25_index(chunks: list) -> object:
    """Build an in-memory BM25 index over the chunk corpus.

    Args:
        chunks: Same corpus used for the dense index. Each element must
            expose a ``text`` attribute.

    Returns:
        A ``rank_bm25.BM25Okapi`` instance (typed as ``object`` to avoid
        a hard import at module level).
    """
    from rank_bm25 import BM25Okapi

    tokenized = [c.text.lower().split() for c in chunks]
    bm25 = BM25Okapi(tokenized)
    logger.info("BM25 index built over %d documents", len(tokenized))
    return bm25


def save_bm25_index(bm25_index: object, chunks: list, path: Path) -> None:
    """Persist a BM25 index and the accompanying chunk corpus to disk.

    Args:
        bm25_index: ``rank_bm25.BM25Okapi`` instance returned by
            :func:`build_bm25_index`.
        chunks: Ordered list of chunks **in the same order** as the index
            was built. Only lightweight fields are serialised.
        path: Destination path for the pickle file.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    chunk_records = [
        {
            "chunk_id": c.chunk_id,
            "filing_id": c.filing_id,
            "section_type": _section_value(c.section_type),
            "text": c.text,
            "chunk_index": int(c.chunk_index),
        }
        for c in chunks
    ]
    with open(path, "wb") as f:
        pickle.dump({"bm25": bm25_index, "chunks": chunk_records}, f)
    logger.info("BM25 index saved to %s (%d chunks)", path, len(chunk_records))


def load_bm25_index(path: Path) -> tuple[object, list[dict]]:
    """Load a persisted BM25 index and chunk corpus from disk.

    Args:
        path: Path to the pickle file written by :func:`save_bm25_index`.

    Returns:
        Tuple of ``(BM25Okapi instance, list of chunk dicts)``.  The chunk
        dicts have keys ``chunk_id``, ``filing_id``, ``section_type``,
        ``text``, and ``chunk_index``.
    """
    with open(path, "rb") as f:
        data = pickle.load(f)
    chunks: list[dict] = data["chunks"]
    logger.info("BM25 index loaded from %s (%d chunks)", path, len(chunks))
    return data["bm25"], chunks


def load_chroma_collection(chroma_dir: Path, collection_name: str) -> object:
    """Load an existing ChromaDB collection from disk.

    Args:
        chroma_dir: Directory where ChromaDB is persisted.
        collection_name: Collection to load.

    Returns:
        A ``chromadb.Collection`` instance.
    """
    import chromadb

    client = chromadb.PersistentClient(path=str(chroma_dir))
    collection = client.get_collection(name=collection_name)
    logger.info(
        "Loaded ChromaDB collection %r with %d documents",
        collection_name,
        collection.count(),
    )
    return collection
