"""Push the fine-tuned BERTimbau sentence transformer to HuggingFace Hub.

Writes a hand-crafted model card to models/sentence_transformer/README.md
before pushing so the Hub copy has accurate training metadata.

Usage:
    huggingface-cli login          # one-time, stores token in ~/.cache
    python scripts/push_model_to_hub.py
"""

from __future__ import annotations

import logging
import os
import sys
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
_VENV_PYTHON = _PROJECT_ROOT / ".venv" / "bin" / "python"
_EXPECTED_PREFIX = str(_PROJECT_ROOT / ".venv")
if sys.prefix != _EXPECTED_PREFIX and _VENV_PYTHON.exists():
    os.execv(str(_VENV_PYTHON), [str(_VENV_PYTHON)] + sys.argv)

sys.path.insert(0, str(_PROJECT_ROOT))

from src import config  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s — %(message)s")
logger = logging.getLogger(__name__)

HUB_REPO_ID: str = config.SENTENCE_TRANSFORMER_HUB_ID

MODEL_CARD: str = """\
---
language:
- pt
tags:
- sentence-transformers
- sentence-similarity
- feature-extraction
- portuguese
- financial
- cvm
- bertimbau
base_model: neuralmind/bert-base-portuguese-cased
license: mit
---

# cvm-bertimbau-sentence-transformer

Fine-tuned [BERTimbau](https://huggingface.co/neuralmind/bert-base-portuguese-cased)
sentence transformer for dense retrieval over Brazilian public company filings (CVM ITR/DFP).
Part of the [CVM Filing Intelligence System](https://github.com/conderafael/cvm-intelligence).

## Training

| Parameter | Value |
|---|---|
| Base model | `neuralmind/bert-base-portuguese-cased` |
| Loss | `MultipleNegativesRankingLoss` |
| Training pairs | 14,500 (adjacent same-section chunk pairs from 686 CVM filings) |
| Epochs | 10 |
| Batch size | 16 (effective 64 with gradient accumulation ×4) |
| Mixed precision | fp16 |
| Max sequence length | 256 tokens |
| Hardware | NVIDIA RTX A1000 (6 GB VRAM), ~2 hours |
| Initial loss | 2.201 (step 50) |
| Final loss | 0.115 (step 2,270) |

Training data: 97,138 management commentary chunks from 49 B3 large-cap companies
(Petrobras, Vale, Itaú, Bradesco, Ambev, etc.), 2022–2025. Pairs are adjacent
paragraphs within the same section of the same filing.

## Retrieval Results

Evaluated on 94 synthetic queries over the 97,138-chunk corpus
(dense-only configuration):

| Metric | Value |
|---|---|
| Recall@5 | 0.057 |
| Recall@10 | 0.071 |
| MRR | 0.100 |
| NDCG@10 | 0.063 |

**Note:** The model underperforms BM25 on query–document retrieval because it was
fine-tuned with doc–doc contrastive pairs. Query–doc performance improves significantly
with GPL (Generative Pseudo Labeling) fine-tuning using synthetic query–chunk pairs.

## Usage

```python
from sentence_transformers import SentenceTransformer

model = SentenceTransformer("condeg/cvm-bertimbau-sentence-transformer")

# Encode a single passage
embeddings = model.encode(["Receita líquida cresceu 12% no trimestre"])

# Encode a batch
texts = [
    "O EBITDA ajustado atingiu R$ 4,2 bilhões no 3T24.",
    "A Companhia mantém posição conservadora de hedge cambial.",
]
embeddings = model.encode(texts, normalize_embeddings=True)
print(embeddings.shape)  # (2, 768)
```

## Limitations

- Trained on Portuguese-language financial filings only; degrades on other domains.
- Max sequence length 256 tokens; longer passages are truncated.
- Query-time performance is below doc-time performance due to training objective mismatch
  (doc–doc pairs vs. query–doc retrieval).
"""


def main() -> None:
    """Write model card and push to HuggingFace Hub."""
    from sentence_transformers import SentenceTransformer

    model_dir = _PROJECT_ROOT / "models" / "sentence_transformer"
    if not model_dir.is_dir():
        logger.error(
            "Local model directory not found: %s\n"
            "Run training first or ensure models/sentence_transformer/ exists.",
            model_dir,
        )
        sys.exit(1)

    # Write the model card before pushing so the Hub copy is correct
    card_path = model_dir / "README.md"
    logger.info("Writing model card to %s", card_path)
    card_path.write_text(MODEL_CARD, encoding="utf-8")

    logger.info("Loading model from %s", model_dir)
    model = SentenceTransformer(str(model_dir))

    logger.info("Pushing to Hub: %s", HUB_REPO_ID)
    try:
        model.push_to_hub(HUB_REPO_ID, exist_ok=True)
    except Exception as exc:
        logger.error("Push failed: %s", exc)
        sys.exit(1)

    logger.info("Done. Model published at https://huggingface.co/%s", HUB_REPO_ID)


if __name__ == "__main__":
    main()
