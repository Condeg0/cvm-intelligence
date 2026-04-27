# src/nlp/ — Embeddings & Sentiment Classification

## Module Purpose

Two ML artifacts: (1) a fine-tuned sentence transformer for Portuguese financial text retrieval, and (2) a 3-class sentiment classifier for management commentary tone.

## Architecture

- `train_sentence_transformer.py` — End-to-end fine-tuning pipeline: pair generation → training → evaluation → model saving.
- `train_sentiment.py` — Bootstrap labeling via Gemini API → classifier training → evaluation.
- `embedder.py` — Inference wrapper: loads fine-tuned model, encodes text to embeddings. Used by both indexing and query-time retrieval.

## Sentence Transformer Fine-Tuning

### Base Model
`neuralmind/bert-base-portuguese-cased` (BERTimbau) — ~440MB, pre-trained on brWaC (Brazilian Portuguese web corpus).

### Training Data — Contrastive Pairs

Generate from the parsed chunks (Phase 2 output). No manual labeling needed.

- **Positive pairs:** Two chunks from the same section of the same filing. They share context and should be embedded nearby.
- **Hard negatives:** Come from in-batch negatives via `MultipleNegativesRankingLoss` — the loss function treats all other positives in the batch as negatives automatically.
- **Target:** 10,000–20,000 positive pairs from ~750 documents.

Pair generation logic:
```python
# For each filing's each section:
#   chunks = [c1, c2, c3, c4, ...]
#   pairs = [(c1, c2), (c2, c3), (c3, c4), ...]  # adjacent chunks
#   Also sample non-adjacent: (c1, c4) with probability 0.3
```

### Training Configuration

Constrained by RTX A1000 (6GB VRAM):

```python
training_args = {
    "model_name": "neuralmind/bert-base-portuguese-cased",
    "max_seq_length": 256,          # BERTimbau limit for 6GB
    "batch_size": 16,               # Actual batch size
    "gradient_accumulation_steps": 4, # Effective batch size = 64
    "epochs": 10,
    "learning_rate": 2e-5,
    "warmup_ratio": 0.1,
    "fp16": True,                   # Mixed precision required
    "loss": "MultipleNegativesRankingLoss",
    "evaluation_steps": 500,
    "output_dir": "models/sentence_transformer",
}
```

Expected training time: 2–4 hours.

### Evaluation — Ablation Study

Compare retrieval metrics on a test set of 100 queries (created in Phase 5):

| Model | Type |
|---|---|
| Fine-tuned BERTimbau | Dense retrieval |
| Base BERTimbau (no fine-tuning) | Dense retrieval |
| `all-MiniLM-L6-v2` | Dense retrieval (English baseline) |
| `paraphrase-multilingual-MiniLM-L12-v2` | Dense retrieval (multilingual baseline) |

Metrics: Recall@5, Recall@10, MRR, NDCG@10.

### Publishing

Upload fine-tuned model to HuggingFace Hub with a model card describing:
- Base model and fine-tuning approach
- Training data characteristics (domain, language, pair count)
- Evaluation results (ablation table)
- Usage example

## Sentiment Classifier

### Labeling Pipeline

1. Select 500 paragraphs from "Relatório da Administração" sections (diverse companies and periods)
2. Send to Gemini API with this prompt:

```
Classify the tone of this management commentary paragraph from a Brazilian public company filing.
Respond with ONLY one of: pessimistic, neutral, optimistic

Paragraph:
{text}
```

3. Parse responses, store in `data/evaluation/sentiment_labels.json`
4. Manually review 100 labels to compute inter-annotator agreement (Cohen's κ)
5. If κ < 0.5 with the API labels, fall back to binary (positive/negative)

### Classifier Architecture

Simple classification head on frozen BERTimbau embeddings:

```python
# Freeze the sentence transformer
# Add: Linear(768, 256) → ReLU → Dropout(0.3) → Linear(256, 3)
# Train with CrossEntropyLoss
# Use class weights if label distribution is imbalanced
```

### Evaluation

- Train/test split: 80/20 stratified
- Metrics: macro F1 (target >0.65), per-class precision/recall
- Report confusion matrix
- Report Cohen's κ between your manual labels and API labels

## Hardware Notes

- Fine-tuning the sentence transformer will use most of the 6GB VRAM. Close other GPU processes.
- Sentiment classifier training is lightweight (frozen backbone, small head) — 5 minutes max.
- At inference time (`embedder.py`), the model runs on CPU for deployment. Pre-compute all chunk embeddings locally on GPU before deployment.
