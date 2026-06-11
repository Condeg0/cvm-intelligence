"""Score all chunks with the sentiment classifier and write labels to SQLite.

Reads every chunk from the DB, embeds the text with the fine-tuned sentence
transformer, runs the binary LogisticRegression classifier, and writes the
predicted label and probability back to the ``chunks`` table.

Runtime: ~10–20 min on GPU for 97k chunks (embedding is the bottleneck).

Usage:
    python scripts/run_sentiment_scoring.py [--batch-size 64] [--device cuda]
"""

from __future__ import annotations

import argparse
import logging
import os
import sqlite3
import sys
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

_CHUNK_FETCH_BATCH = 2000   # rows fetched from SQLite at a time
_DB_WRITE_BATCH   = 500    # rows updated per SQLite transaction


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Sentiment scoring pipeline")
    p.add_argument("--batch-size", type=int, default=64,
                   help="Embedding batch size (default: 64)")
    p.add_argument("--device", type=str, default=None,
                   help="Force device (cpu / cuda). Default: auto")
    p.add_argument("--overwrite", action="store_true",
                   help="Re-score chunks that already have a label")
    return p.parse_args()


def load_classifier():
    """Load the saved binary LogisticRegression and LabelEncoder."""
    import joblib
    clf = joblib.load(config.SENTIMENT_MODEL_DIR / "classifier.joblib")
    le  = joblib.load(config.SENTIMENT_MODEL_DIR / "label_encoder.joblib")
    return clf, le


def main() -> None:
    args = parse_args()

    logger.info("Loading sentiment classifier from %s", config.SENTIMENT_MODEL_DIR)
    clf, le = load_classifier()
    logger.info("Classes: %s", list(le.classes_))

    model_dir = config.SENTENCE_TRANSFORMER_PATH

    from src.nlp.embedder import embed_chunks

    conn = sqlite3.connect(config.DB_PATH)

    where_clause = "WHERE chunk_text IS NOT NULL AND chunk_text != ''"
    if not args.overwrite:
        where_clause += " AND sentiment_label IS NULL"

    total = conn.execute(f"SELECT COUNT(*) FROM chunks {where_clause}").fetchone()[0]
    logger.info("%d chunks to score", total)

    if total == 0:
        logger.info("Nothing to do — all chunks already scored. Use --overwrite to re-score.")
        conn.close()
        return

    offset = 0
    scored = 0

    while offset < total:
        rows = conn.execute(
            f"SELECT chunk_id, chunk_text FROM chunks {where_clause} LIMIT ? OFFSET ?",
            (_CHUNK_FETCH_BATCH, offset),
        ).fetchall()
        if not rows:
            break

        chunk_ids = [r[0] for r in rows]
        texts     = [r[1] for r in rows]

        # Embed
        embeddings = embed_chunks(
            texts,
            model_dir=model_dir,
            batch_size=args.batch_size,
            show_progress=False,
            device=args.device,
        )

        # Predict
        y_pred  = clf.predict(embeddings)
        y_proba = clf.predict_proba(embeddings)

        # Build (label, score, chunk_id) triples
        updates = []
        for i, chunk_id in enumerate(chunk_ids):
            label = le.inverse_transform([y_pred[i]])[0]
            # confidence = probability of predicted class
            score = float(y_proba[i][y_pred[i]])
            updates.append((label, score, chunk_id))

        # Write to DB in batches
        for i in range(0, len(updates), _DB_WRITE_BATCH):
            batch = updates[i : i + _DB_WRITE_BATCH]
            conn.executemany(
                "UPDATE chunks SET sentiment_label = ?, sentiment_score = ? WHERE chunk_id = ?",
                batch,
            )
        conn.commit()

        scored  += len(rows)
        offset  += _CHUNK_FETCH_BATCH
        logger.info("Scored %d / %d chunks (%.0f%%)", scored, total, 100 * scored / total)

    conn.close()

    # Quick distribution check
    conn2 = sqlite3.connect(config.DB_PATH)
    dist = conn2.execute(
        "SELECT sentiment_label, COUNT(*) FROM chunks WHERE sentiment_label IS NOT NULL "
        "GROUP BY sentiment_label ORDER BY COUNT(*) DESC"
    ).fetchall()
    conn2.close()

    logger.info("Done. Label distribution: %s", dict(dist))


if __name__ == "__main__":
    main()
