"""Fine-tune BERTimbau as a sentence transformer for CVM filings.

Generates contrastive training pairs from parsed chunks and fine-tunes
``neuralmind/bert-base-portuguese-cased`` with ``MultipleNegativesRankingLoss``.
Runs on RTX A1000 (6GB VRAM) with fp16 mixed precision and gradient accumulation.

Usage:
    python -m src.nlp.train_sentence_transformer
"""

from __future__ import annotations

import json
import logging
import math
import random
from collections import defaultdict
from pathlib import Path

logger = logging.getLogger(__name__)

# Sections whose adjacent chunks make semantically coherent positive pairs.
# Financial-table sections are excluded — their rows are numeric, not prose.
_PROSE_SECTION_KEYWORDS = (
    "Administração",
    "Diretoria",
    "Comentário",
    "Desempenho",
    "Auditores",
    "Auditor",
    "Notas Explicativas",
)


def generate_training_pairs(chunks_dir: Path) -> list[tuple[str, str]]:
    """Generate contrastive (anchor, positive) pairs from chunked filing text.

    For each (filing, section) group, adjacent chunks are paired as positives.
    Non-adjacent pairs are added with 30% probability when the group has ≥ 4 chunks.
    Very short chunks (< 20 whitespace-separated tokens) are skipped.

    Args:
        chunks_dir: Directory containing JSON chunk files from the parsing phase.

    Returns:
        List of (anchor, positive) string pairs for contrastive training.
    """
    groups: dict[tuple[str, str], list[str]] = defaultdict(list)

    for chunk_file in sorted(chunks_dir.glob("*.json")):
        try:
            chunks = json.loads(chunk_file.read_text())
        except Exception as exc:
            logger.warning("Failed to read %s: %s", chunk_file.name, exc)
            continue

        for chunk in chunks:
            section = chunk.get("section_type", "")
            text = chunk.get("text", "").strip()
            # Only prose sections produce useful semantic pairs
            if not any(kw in section for kw in _PROSE_SECTION_KEYWORDS):
                continue
            if len(text.split()) < 20:
                continue
            groups[(chunk["filing_id"], section)].append(text)

    rng = random.Random(42)
    pairs: list[tuple[str, str]] = []

    for (_filing_id, _section), texts in groups.items():
        if len(texts) < 2:
            continue

        # Adjacent pairs (always included)
        for i in range(len(texts) - 1):
            pairs.append((texts[i], texts[i + 1]))

        # Non-adjacent pairs (30% sample, requires ≥ 4 chunks)
        if len(texts) >= 4:
            for i in range(len(texts) - 3):
                if rng.random() < 0.3:
                    j = rng.randint(i + 2, len(texts) - 1)
                    pairs.append((texts[i], texts[j]))

    rng.shuffle(pairs)
    logger.info(
        "Generated %d training pairs from %d (filing, section) groups",
        len(pairs), len(groups),
    )
    return pairs


def train(
    pairs: list[tuple[str, str]],
    output_dir: Path,
    epochs: int = 10,
    batch_size: int = 16,
    gradient_accumulation_steps: int = 4,
    use_fp16: bool = True,
) -> None:
    """Fine-tune BERTimbau sentence transformer on the provided pairs.

    Uses ``MultipleNegativesRankingLoss`` with in-batch negatives via the
    sentence-transformers v5 ``SentenceTransformerTrainer`` API.
    Saves the best checkpoint (by eval cosine similarity) to ``output_dir``.

    Args:
        pairs: (anchor, positive) text pairs for training.
        output_dir: Path where the fine-tuned model will be saved.
        epochs: Number of training epochs.
        batch_size: Per-device batch size (keep ≤ 16 for 6GB VRAM).
        gradient_accumulation_steps: Steps before a parameter update.
        use_fp16: Enable fp16 mixed precision.
    """
    from datasets import Dataset
    from sentence_transformers import SentenceTransformer, SentenceTransformerTrainer
    from sentence_transformers import SentenceTransformerTrainingArguments
    from sentence_transformers.sentence_transformer.losses import (
        MultipleNegativesRankingLoss,
    )

    from src import config

    logger.info("Loading base model: %s", config.BERT_MODEL_NAME)
    model = SentenceTransformer(config.BERT_MODEL_NAME)
    model.max_seq_length = config.MAX_SEQ_LENGTH

    # Hold out 5% for evaluation (capped at 500 pairs)
    rng = random.Random(42)
    shuffled = list(pairs)
    rng.shuffle(shuffled)
    n_eval = min(500, max(50, len(shuffled) // 20))
    eval_pairs = shuffled[:n_eval]
    train_pairs = shuffled[n_eval:]

    logger.info("Train pairs: %d  Eval pairs: %d", len(train_pairs), len(eval_pairs))

    train_dataset = Dataset.from_dict({
        "anchor": [a for a, _ in train_pairs],
        "positive": [p for _, p in train_pairs],
    })
    eval_dataset = Dataset.from_dict({
        "anchor": [a for a, _ in eval_pairs],
        "positive": [p for _, p in eval_pairs],
    })

    loss = MultipleNegativesRankingLoss(model)

    steps_per_epoch = math.ceil(len(train_pairs) / (batch_size * gradient_accumulation_steps))
    warmup_steps = math.ceil(steps_per_epoch * epochs * 0.1)

    output_dir.mkdir(parents=True, exist_ok=True)

    args = SentenceTransformerTrainingArguments(
        output_dir=str(output_dir),
        num_train_epochs=epochs,
        per_device_train_batch_size=batch_size,
        gradient_accumulation_steps=gradient_accumulation_steps,
        learning_rate=2e-5,
        warmup_steps=warmup_steps,
        fp16=use_fp16,
        bf16=False,
        eval_strategy="steps",
        eval_steps=steps_per_epoch,  # evaluate once per epoch
        save_strategy="steps",
        save_steps=steps_per_epoch,
        save_total_limit=2,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        logging_steps=50,
        report_to="none",
        dataloader_drop_last=True,
    )

    trainer = SentenceTransformerTrainer(
        model=model,
        args=args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        loss=loss,
    )

    logger.info(
        "Starting fine-tuning — epochs=%d  batch=%d  grad_accum=%d  fp16=%s  warmup=%d",
        epochs, batch_size, gradient_accumulation_steps, use_fp16, warmup_steps,
    )
    trainer.train()

    model.save_pretrained(str(output_dir))
    logger.info("Fine-tuned model saved to %s", output_dir)


def main() -> None:
    """CLI entry point: generate pairs → fine-tune → save model."""
    import argparse
    import sys

    _PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
    sys.path.insert(0, str(_PROJECT_ROOT))

    from src import config

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s — %(message)s",
    )

    parser = argparse.ArgumentParser(description="Fine-tune BERTimbau sentence transformer")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=config.FINETUNE_BATCH_SIZE)
    parser.add_argument("--output-dir", type=Path, default=config.MODELS_DIR / "sentence_transformer")
    parser.add_argument("--pairs-cache", type=Path, default=config.PROCESSED_DIR / "training_pairs.json",
                        help="Cache generated pairs here to avoid re-generating")
    parser.add_argument("--max-pairs", type=int, default=15000,
                        help="Sample this many pairs from the full set (default 15000; 0 = use all)")
    parser.add_argument("--no-fp16", action="store_true")
    args = parser.parse_args()

    # Load or generate pairs
    if args.pairs_cache.exists():
        logger.info("Loading cached pairs from %s", args.pairs_cache)
        raw = json.loads(args.pairs_cache.read_text())
        pairs = [tuple(p) for p in raw]
    else:
        logger.info("Generating training pairs from %s", config.CHUNKS_DIR)
        pairs = generate_training_pairs(config.CHUNKS_DIR)
        args.pairs_cache.parent.mkdir(parents=True, exist_ok=True)
        args.pairs_cache.write_text(json.dumps(pairs, ensure_ascii=False, indent=None))
        logger.info("Pairs cached to %s", args.pairs_cache)

    # Subsample if requested (keeps training time within 2–4 h on RTX A1000)
    if args.max_pairs and len(pairs) > args.max_pairs:
        rng_sample = random.Random(42)
        pairs = rng_sample.sample(pairs, args.max_pairs)
        logger.info("Sampled %d pairs from %d total", args.max_pairs, len(raw))

    logger.info("Total pairs: %d", len(pairs))

    train(
        pairs=pairs,
        output_dir=args.output_dir,
        epochs=args.epochs,
        batch_size=args.batch_size,
        gradient_accumulation_steps=config.GRADIENT_ACCUMULATION_STEPS,
        use_fp16=not args.no_fp16,
    )


if __name__ == "__main__":
    main()
