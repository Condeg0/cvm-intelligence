"""Train a 3-class sentiment classifier on top of frozen BERTimbau embeddings.

Labels (pessimistic / neutral / optimistic) are bootstrapped via Gemini API.
The classifier head (LogisticRegression) is trained on fixed-length sentence
embeddings so no GPU fine-tuning is needed at this stage.

Usage:
    # Step 1 — bootstrap labels (requires GEMINI_API_KEY in .env)
    python -m src.nlp.train_sentiment --bootstrap

    # Step 2 — train classifier on the saved labels
    python -m src.nlp.train_sentiment --train

    # Both steps
    python -m src.nlp.train_sentiment --bootstrap --train
"""

from __future__ import annotations

import json
import logging
import random
import time
from pathlib import Path

import numpy as np

logger = logging.getLogger(__name__)

SENTIMENT_CLASSES: list[str] = ["pessimistic", "neutral", "optimistic"]

_SENTIMENT_PROMPT = """\
Classify the tone of this paragraph from a Brazilian public company's management commentary.
Respond with ONLY one word: pessimistic, neutral, or optimistic.

Paragraph:
{text}"""


def _call_gemini(prompt: str) -> str:
    """Call the Gemini API and return the raw response text."""
    from src import config

    if config._GENAI_CLIENT is not None:
        response = config._GENAI_CLIENT.models.generate_content(
            model=config.GEMINI_MODEL,
            contents=prompt,
        )
        return response.text.strip()

    # Fallback: old google-generativeai SDK
    import google.generativeai as genai  # type: ignore
    model = genai.GenerativeModel(config.GEMINI_MODEL)
    response = model.generate_content(prompt)
    return response.text.strip()


def _collect_prose_chunks(chunks_dir: Path) -> list[str]:
    """Return all Relatório da Administração chunk texts from the chunks directory."""
    texts: list[str] = []
    for chunk_file in sorted(chunks_dir.glob("*.json")):
        try:
            chunks = json.loads(chunk_file.read_text())
        except Exception as exc:
            logger.warning("Failed to read %s: %s", chunk_file.name, exc)
            continue
        for chunk in chunks:
            section = chunk.get("section_type", "")
            if "Administração" not in section and "Comentário" not in section:
                continue
            text = chunk.get("text", "").strip()
            if len(text.split()) >= 30:  # min 30 words for meaningful sentiment
                texts.append(text)
    return texts


def bootstrap_labels_with_gemini(
    chunks: list[str],
    n_samples: int = 500,
) -> list[dict]:
    """Use Gemini API to generate initial sentiment labels for chunk text.

    Calls Gemini 2.5 Flash for each sampled chunk, parses the label, and
    handles API errors gracefully (defaults to 'neutral' on failure).
    Applies polite rate-limiting (~14 req/burst then 1 s pause).

    Args:
        chunks: List of text chunks to label.
        n_samples: Maximum number of samples to label.

    Returns:
        List of dicts with keys: ``text``, ``label``, ``raw_response``.
    """
    rng = random.Random(42)
    sample = rng.sample(chunks, min(n_samples, len(chunks)))

    results: list[dict] = []
    for i, text in enumerate(sample):
        if i > 0 and i % 50 == 0:
            logger.info("Labeling chunk %d/%d", i, len(sample))

        prompt = _SENTIMENT_PROMPT.format(text=text[:800])

        try:
            raw = _call_gemini(prompt).lower()
        except Exception as exc:
            logger.warning("Gemini call failed for chunk %d: %s", i, exc)
            raw = "error"

        label = next(
            (cls for cls in SENTIMENT_CLASSES if cls in raw),
            "neutral",  # fallback when response is unexpected
        )

        results.append({"text": text, "label": label, "raw_response": raw})

        # Rate-limit: Gemini free tier allows ~15 req/min → pause after every 14
        if (i + 1) % 14 == 0:
            time.sleep(1.0)

    dist = {cls: sum(1 for r in results if r["label"] == cls) for cls in SENTIMENT_CLASSES}
    logger.info("Labeled %d chunks. Distribution: %s", len(results), dist)
    return results


def train_classifier(
    embeddings: np.ndarray,
    labels: list[str],
    output_dir: Path,
) -> None:
    """Train a lightweight classifier head on frozen sentence embeddings.

    Fits a multinomial LogisticRegression on pre-computed embeddings.
    Uses balanced class weights to handle label imbalance.
    Saves the trained classifier and label encoder to ``output_dir``.

    Args:
        embeddings: 2-D float32 array of shape ``(n_samples, embedding_dim)``.
        labels: Sentiment label strings aligned with rows in ``embeddings``.
        output_dir: Directory where classifier weights will be saved.
    """
    import joblib
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import LabelEncoder

    output_dir.mkdir(parents=True, exist_ok=True)

    le = LabelEncoder()
    y = le.fit_transform(labels)

    clf = LogisticRegression(
        C=1.0,
        max_iter=1000,
        class_weight="balanced",
        solver="lbfgs",
        random_state=42,
    )
    clf.fit(embeddings, y)

    joblib.dump(clf, output_dir / "classifier.joblib")
    joblib.dump(le, output_dir / "label_encoder.joblib")

    train_acc = clf.score(embeddings, y)
    logger.info("Classifier trained. Train accuracy: %.3f. Saved to %s", train_acc, output_dir)


def evaluate_classifier(
    model_dir: Path,
    embeddings: np.ndarray,
    labels: list[str],
) -> dict[str, float]:
    """Evaluate saved classifier and return per-class and macro F1 scores.

    Args:
        model_dir: Directory of saved classifier (from :func:`train_classifier`).
        embeddings: Test-set embeddings.
        labels: Ground-truth label strings for the test set.

    Returns:
        Dict with keys: ``macro_f1``, ``pessimistic_f1``, ``neutral_f1``,
        ``optimistic_f1``.
    """
    import joblib
    from sklearn.metrics import classification_report, f1_score

    clf = joblib.load(model_dir / "classifier.joblib")
    le = joblib.load(model_dir / "label_encoder.joblib")

    y_true = le.transform(labels)
    y_pred = clf.predict(embeddings)

    macro_f1 = f1_score(y_true, y_pred, average="macro")
    report = classification_report(
        y_true, y_pred,
        target_names=le.classes_,
        output_dict=True,
    )

    logger.info(
        "Classification report:\n%s",
        classification_report(y_true, y_pred, target_names=le.classes_),
    )

    result: dict[str, float] = {"macro_f1": macro_f1}
    for cls in SENTIMENT_CLASSES:
        if cls in report:
            result[f"{cls}_f1"] = report[cls]["f1-score"]
    return result


def main() -> None:
    """CLI entry point: bootstrap labels and/or train classifier."""
    import argparse
    import sys

    _PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
    sys.path.insert(0, str(_PROJECT_ROOT))

    from src import config
    from src.nlp.embedder import embed_chunks

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s — %(message)s",
    )

    parser = argparse.ArgumentParser(description="Sentiment classifier pipeline")
    parser.add_argument("--bootstrap", action="store_true", help="Run Gemini labeling")
    parser.add_argument("--train", action="store_true", help="Train classifier on saved labels")
    parser.add_argument("--n-samples", type=int, default=500)
    parser.add_argument(
        "--model-dir",
        type=Path,
        default=config.MODELS_DIR / "sentence_transformer",
        help="Fine-tuned sentence transformer for embeddings",
    )
    parser.add_argument(
        "--device", type=str, default="cpu",
        help="Device for embedding (default: cpu — avoids OOM when GPU is training)",
    )
    parser.add_argument(
        "--binary", action="store_true",
        help="Collapse to binary: optimistic=positive vs neutral+pessimistic=negative",
    )
    args = parser.parse_args()

    if not args.bootstrap and not args.train:
        parser.error("Specify at least one of --bootstrap, --train")

    labels_path = config.SENTIMENT_LABELS_PATH

    if args.bootstrap:
        logger.info("Collecting prose chunks from %s", config.CHUNKS_DIR)
        prose_chunks = _collect_prose_chunks(config.CHUNKS_DIR)
        logger.info("Found %d prose chunks; sampling %d", len(prose_chunks), args.n_samples)

        labeled = bootstrap_labels_with_gemini(prose_chunks, n_samples=args.n_samples)

        labels_path.parent.mkdir(parents=True, exist_ok=True)
        labels_path.write_text(
            json.dumps(labeled, ensure_ascii=False, indent=2)
        )
        logger.info("Labels saved to %s", labels_path)

    if args.train:
        if not labels_path.exists():
            logger.error("Labels file not found at %s — run --bootstrap first", labels_path)
            sys.exit(1)

        labeled = json.loads(labels_path.read_text())
        texts = [r["text"] for r in labeled]
        labels = [r["label"] for r in labeled]

        if args.binary:
            # Binary fallback: optimistic → "positive", neutral/pessimistic → "negative"
            labels = ["positive" if l == "optimistic" else "negative" for l in labels]
            logger.info(
                "Binary mode: positive=%d  negative=%d",
                labels.count("positive"), labels.count("negative"),
            )

        logger.info("Embedding %d labeled chunks on device=%s…", len(texts), args.device)
        embeddings = embed_chunks(
            texts,
            model_dir=args.model_dir if args.model_dir.exists() else None,
            batch_size=config.EMBEDDING_BATCH_SIZE,
            device=args.device,
        )

        # 80/20 stratified split
        from sklearn.model_selection import train_test_split
        X_train, X_test, y_train, y_test = train_test_split(
            embeddings, labels,
            test_size=0.2,
            random_state=42,
            stratify=labels,
        )

        logger.info("Training on %d samples, evaluating on %d", len(X_train), len(X_test))
        out_dir = config.SENTIMENT_MODEL_DIR
        train_classifier(X_train, y_train, out_dir)

        # Record binary mode in metadata so consumers know the label schema
        if args.binary:
            (out_dir / "mode.json").write_text(
                json.dumps({"mode": "binary", "classes": ["negative", "positive"]})
            )

        metrics = evaluate_classifier(out_dir, X_test, y_test)
        logger.info("Final metrics: %s", metrics)

        target = 0.65
        target_met = metrics.get("macro_f1", 0) >= target
        logger.info(
            "Target macro F1 ≥ %.2f: %s (got %.3f)",
            target, "MET" if target_met else "NOT MET",
            metrics.get("macro_f1", 0),
        )


if __name__ == "__main__":
    main()
