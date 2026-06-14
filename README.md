# CVM Filing Intelligence System

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://cvm-intelligence.streamlit.app)
![Python](https://img.shields.io/badge/Python-3.10%2B-blue?logo=python)
![PyTorch](https://img.shields.io/badge/PyTorch-2.1-orange?logo=pytorch)
![HuggingFace](https://img.shields.io/badge/HuggingFace-BERTimbau-yellow?logo=huggingface)
![ChromaDB](https://img.shields.io/badge/Vector_Store-ChromaDB-purple)
![Streamlit](https://img.shields.io/badge/App-Streamlit-red?logo=streamlit)
![Gemini](https://img.shields.io/badge/LLM-Gemini_2.5_Flash-blue?logo=google)

An end-to-end pipeline that transforms raw Brazilian public company filings (CVM ITR and DFP documents) into a queryable intelligence system. The system covers 49 of the largest B3-listed companies over three years (~686 filings), combining a deterministic quantitative extraction engine validated against structured ground truth with a hybrid semantic RAG pipeline over 97,000 management commentary chunks.

---

## Live App

**[https://cvm-intelligence.streamlit.app](https://cvm-intelligence.streamlit.app)**

> The Financial Dashboard and Evaluation pages are fully functional on the live app.
> The RAG Query page requires the 1.8 GB ChromaDB index and is available in local setup only.

---

## Key Results

### Quantitative Extraction (Phase 3)

Rule-based extraction against CVM structured CSV ground truth across 7 financial metrics and 686 filings:

| Metric | Exact Match |
|---|---|
| Total Assets | 97% |
| Total Equity | 97% |
| Gross Profit | 96% |
| Revenue | 96% |
| Operating Cash Flow | 95% |
| COGS | 93% |
| Net Income | 92% |
| **Overall** | **95.2%** |

- **MAPE:** 0.06% on matched values
- **Coverage:** 632 / 686 filings fully extracted; 4,425 / 4,651 metric-period values exactly matched
- No LLM used for number extraction — fully deterministic, auditable rules

### Retrieval Evaluation (Phase 5)

Fine-tuned BERTimbau vs baselines on 94-query test set (5,000-chunk subsample):

| Model | Recall@5 | Recall@10 | MRR | NDCG@10 |
|---|---|---|---|---|
| **Fine-tuned BERTimbau** | **20.4%** | **26.9%** | **28.0%** | **21.9%** |
| Base BERTimbau | 9.6% | 12.1% | 18.8% | 11.4% |
| Multilingual MiniLM-L12 | 9.6% | 13.5% | 16.8% | 11.5% |

Fine-tuned model achieves **2.1× Recall@5** and **1.5× MRR** over the base BERTimbau.

### Sentiment Labeling (Phase 4)

500 management commentary chunks labeled via Gemini API bootstrap, used to train a 3-class classifier:
- Neutral: 76% | Optimistic: 22% | Pessimistic: 2%

---

## Architecture

Two independent pipelines share a SQLite backbone:

### Pipeline 1 — Quantitative Extraction

```
CVM CSVs (ground truth) ──┐
                           ├──► Validation ──► SQLite metrics table
CVM PDFs ──► PyMuPDF ──► Rule-based extractor ──┘
             pdfplumber
```

**Key decisions:**
- **Rule-based, not LLM** — financial values are deterministic; LLMs introduce hallucination risk and are expensive at 686× scale. Regex + table position heuristics achieve 95.2% exact match.
- **Dual extraction** — PyMuPDF for narrative text blocks (positional metadata), pdfplumber for financial statement tables (preserves row/column structure).
- **Validated against CSV ground truth** — every extracted value is checked against CVM's own structured data before being trusted.

### Pipeline 2 — Qualitative RAG

```
PDFs ──► Section detector ──► Paragraph chunker ──► BERTimbau embedder
                                                           │
                                              ChromaDB (dense) + BM25 (sparse)
                                                           │
                                              Reciprocal Rank Fusion (top-10)
                                                           │
                                          Cross-encoder reranker (top-5)
                                                           │
                                              Gemini 2.5 Flash (generation)
```

**Key decisions:**
- **Section-aware chunking, not fixed-size windows** — splits on paragraph boundaries within detected sections (Relatório da Administração, Notas Explicativas), preserving document structure.
- **Fine-tuned BERTimbau** — `neuralmind/bert-base-portuguese-cased` fine-tuned with MultipleNegativesRankingLoss on 124k contrastive pairs generated from the filing corpus. 2.1× retrieval improvement over base model.
- **Hybrid retrieval + RRF** — dense (ChromaDB) + sparse (BM25) combined with Reciprocal Rank Fusion. Dense captures semantic similarity; sparse captures exact financial terminology (EBITDA, LAJIDA).
- **Cross-encoder reranking** — reranks top-10 RRF results with `cross-encoder/ms-marco-MiniLM-L-6-v2` before generation.
- **No RAG frameworks** — every component is built from primitives. No LangChain, no LlamaIndex.

---

## Tech Stack

| Component | Tool |
|---|---|
| PDF text extraction | PyMuPDF (fitz) |
| PDF table extraction | pdfplumber |
| Sentence embeddings | sentence-transformers + BERTimbau |
| Vector store | ChromaDB (persisted to disk) |
| Sparse retrieval | rank-bm25 |
| Cross-encoder reranker | cross-encoder/ms-marco-MiniLM-L-6-v2 |
| Structured storage | SQLite (stdlib) |
| LLM generation + labeling | Gemini 2.5 Flash |
| App framework | Streamlit |
| Charts | Plotly + Altair |
| ML framework | PyTorch + Transformers |
| Training hardware | NVIDIA RTX A1000 (6 GB VRAM) |

---

## Fine-tuned Model

**[condeg/cvm-bertimbau-sentence-transformer](https://huggingface.co/condeg/cvm-bertimbau-sentence-transformer)**

Base: `neuralmind/bert-base-portuguese-cased`
Training: MultipleNegativesRankingLoss, 124k contrastive pairs from CVM filings, 5 epochs, fp16, effective batch 64.

---

## Repository Structure

```
cvm-intelligence/
├── data/
│   ├── cvm_metrics_dashboard.db    # Slim SQLite (432 KB) — committed to git
│   └── evaluation/                 # Pre-computed evaluation JSON artefacts
├── models/
│   └── sentiment_classifier/       # .joblib classifier weights (committed)
├── src/
│   ├── acquisition/                # CVM CSV + PDF downloaders
│   ├── parsing/                    # PyMuPDF extraction, section detector, chunker
│   ├── extraction/                 # Rule-based metric extractor + validator
│   ├── nlp/                        # BERTimbau fine-tuning + sentiment classifier
│   ├── rag/                        # Indexer, retriever, reranker, generator
│   └── db/                         # SQLite schema + CRUD
├── app/
│   ├── app.py                      # Streamlit entry point
│   ├── load_vectorstore.py         # ChromaDB availability helper
│   └── pages/                      # 4 app pages
├── notebooks/                      # 6 development + evaluation notebooks
├── scripts/                        # Batch pipeline scripts
├── requirements.txt                # Streamlit Cloud runtime deps
└── requirements-dev.txt            # Full local dev deps (GPU pipeline)
```

---

## Local Setup

### Prerequisites

- Python 3.10+
- NVIDIA GPU with ≥ 6 GB VRAM (for training and indexing; app inference is CPU-only)

### Install

```bash
git clone https://github.com/conderafael/cvm-intelligence.git
cd cvm-intelligence
python -m venv .venv
source .venv/bin/activate

# Dashboard + evaluation only (Streamlit Cloud parity):
pip install -r requirements.txt

# Full local pipeline (RAG, training, indexing):
pip install -r requirements.txt -r requirements-dev.txt
```

### Configure

```bash
# Create a .env file with your Gemini API key:
echo "GEMINI_API_KEY=your_key_here" > .env
```

### Run the App

```bash
streamlit run app/app.py
```

The Financial Dashboard and Evaluation pages work immediately from the committed slim SQLite database. The RAG Query page additionally requires the ChromaDB index — see full pipeline below.

### Full Pipeline (RAG included, ~2–4 hours on GPU)

```bash
python scripts/run_extraction.py   # download PDFs + extract + validate metrics
python scripts/run_indexing.py     # embed 97k chunks into ChromaDB + BM25 index
streamlit run app/app.py           # all four pages now fully functional
```

---

## Notebooks

| Notebook | Description |
|---|---|
| `01_data_exploration.ipynb` | CVM data structure, company selection, manifest |
| `02_parsing_development.ipynb` | PyMuPDF + pdfplumber development, section detection |
| `03_extraction_validation.ipynb` | Metric extraction rules, failure taxonomy |
| `04_embedding_training.ipynb` | BERTimbau fine-tuning, contrastive pair generation |
| `05_retrieval_evaluation.ipynb` | Ablation study: dense vs sparse vs hybrid vs reranker |
| `06_sentiment_evaluation.ipynb` | Sentiment classifier training and evaluation |

---

## Author

**Rafael Condé Gomes** — [condeg.rafael@gmail.com](mailto:condeg.rafael@gmail.com)
