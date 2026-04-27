# CVM Filing Intelligence System

An end-to-end pipeline that transforms raw Brazilian public company filings (CVM ITR and DFP documents) into a queryable intelligence system. Built as a portfolio project demonstrating ML engineering across the full stack: data acquisition, PDF parsing, rule-based extraction, NLP model fine-tuning, vector search, and a Streamlit application.

**Author:** Rafael Condé Gomes

---

## What It Does

Two parallel pipelines over the same corpus of 686 CVM filings from 49 B3 large-caps (2022–2025):

1. **Quantitative pipeline** — Extracts financial metrics (revenue, net income, EBITDA, etc.) from PDF financial statements using rule-based parsing validated against CVM's structured CSV ground truth. Achieves **95.2% exact match** on 4,651 metric extractions.

2. **Qualitative pipeline** — Fine-tuned BERTimbau sentence transformer + BM25 hybrid retrieval over 97,138 management commentary chunks. Answers natural-language questions via Gemini 2.5 Flash with cited sources. Binary sentiment classifier tracks management tone over time.

---

## Architecture

```
┌─────────────────────────────────────────────────────────────────────┐
│                         Data Sources                                │
│  CVM Dados Abertos CSVs (ground truth)   CVM PDF Filing Portal      │
└────────────────┬───────────────────────────────┬────────────────────┘
                 │                               │
        ┌────────▼────────┐             ┌────────▼────────┐
        │  CSV Acquisition │             │  PDF Acquisition │
        │  download_csvs   │             │  download_pdfs   │
        └────────┬─────────┘            └────────┬─────────┘
                 │                               │
                 │                      ┌────────▼─────────────────┐
                 │                      │     PDF Parsing           │
                 │                      │  PyMuPDF + pdfplumber     │
                 │                      │  section_detector.py      │
                 │                      │  chunker.py               │
                 │                      └──────┬──────────┬─────────┘
                 │                             │          │
        ┌────────▼──────────────┐   ┌──────────▼──┐  ┌───▼──────────────┐
        │  Quantitative         │   │  NLP / RAG  │  │  SQLite DB       │
        │  Extraction           │   │  Pipeline   │  │  cvm_metrics.db  │
        │  metric_extractor.py  │   │             │  │  97k chunks      │
        │  value_parser.py      │   │ BERTimbau   │  │  4,651 metrics   │
        │  validator.py         │   │ fine-tuned  │  │  686 filings     │
        └──────────┬────────────┘   │ BM25 index  │  └──────────────────┘
                   │                │ ChromaDB    │
                   │                └──────┬──────┘
                   │                       │
                   └───────────┬───────────┘
                               │
                    ┌──────────▼────────────┐
                    │    Streamlit App       │
                    │  4-page portfolio UI  │
                    │  ┌─────────────────┐  │
                    │  │ 1. RAG Query    │  │
                    │  │ 2. Dashboard    │  │
                    │  │ 3. Sentiment    │  │
                    │  │ 4. Evaluation   │  │
                    │  └─────────────────┘  │
                    └───────────────────────┘
```

### Tech Stack

| Component | Tool |
|---|---|
| PDF text extraction | PyMuPDF (fitz) |
| PDF table extraction | pdfplumber |
| Sentence embeddings | sentence-transformers + BERTimbau |
| Vector store | ChromaDB (HNSW cosine, ~1.8 GB) |
| Sparse retrieval | rank_bm25 (BM25Okapi, pickled) |
| Fusion | Reciprocal Rank Fusion (k=60) |
| Cross-encoder reranker | cross-encoder/ms-marco-MiniLM-L-6-v2 |
| Structured DB | SQLite (~270 MB) |
| Generation | Gemini 2.5 Flash |
| Sentiment | LogisticRegression on frozen BERTimbau embeddings |
| App framework | Streamlit |
| Charts | Plotly + Altair |
| Training hardware | NVIDIA RTX A1000 (6GB VRAM), fp16 |

---

## Results

### Quantitative Extraction

Evaluated on 4,651 metric extractions across 686 filings (49 companies):

| Status | Count | Share |
|---|---|---|
| **Exact match** | 4,425 | **95.2%** |
| Mismatch | 141 | 3.0% |
| Missing | 70 | 1.5% |
| Close (≤1% error) | 15 | 0.3% |

**MAPE:** < 1% across exact + close + mismatch extractions.

**Key fix:** ITR income-statement pages have 4 columns `[quarterly, YTD, prior-quarterly, prior-YTD]`. Using index 1 (YTD) raised exact-match accuracy from 72% → 95%.

Metrics extracted: `revenue`, `cogs`, `gross_profit`, `net_income`, `total_assets`, `total_equity`, `operating_cash_flow` (plus `ebitda` and `net_debt` where reported).

### Retrieval Ablation

Evaluated on 94 synthetic queries (Gemini-generated) over 97,138 chunks from 623 filings:

| Configuration | Chunk Hit@5 | Filing Hit@5 | MRR |
|---|---|---|---|
| Dense only (fine-tuned BERTimbau) | 0.160 | 0.298 | 0.100 |
| **Sparse only (BM25)** | **0.287** | **0.394** | **0.191** |
| Hybrid (Dense + BM25 + RRF) | 0.287 | 0.362 | 0.173 |
| Hybrid + Cross-encoder Reranker | 0.245 | 0.362 | 0.148 |

**Best configuration:** BM25. *Filing Hit@5* (correct filing appears in top-5, not just exact chunk) reaches **39.4%** — a more meaningful metric given 97k chunks and narrow relevance labels.

**Why dense underperforms BM25:** The sentence transformer was fine-tuned with doc–doc contrastive pairs (adjacent chunks from same section). At query time the input is a natural-language question — a distribution never seen during training. The fix is GPL (Generative Pseudo Labeling): generate synthetic query–chunk pairs with Gemini, then fine-tune on those.

### Sentence Transformer Training

- **Base model:** `neuralmind/bert-base-portuguese-cased` (BERTimbau)
- **Loss:** MultipleNegativesRankingLoss
- **Training pairs:** 14,500 (adjacent same-section chunk pairs)
- **Epochs:** 10 | **Batch size:** 16 (effective 64 with gradient accumulation) | **fp16**
- **Training loss:** 2.201 (step 50) → **0.115** (step 2,270)
- **Hardware:** RTX A1000, ~2 hours

### Sentiment Classifier

Binary classifier (positive / negative) on frozen BERTimbau embeddings:

| Metric | Value |
|---|---|
| Macro F1 | **0.601** |
| Positive F1 | 0.462 |
| Negative F1 | 0.740 |
| Accuracy | 0.650 |

**Labels:** 500 bootstrapped via Gemini 2.5 Flash, 100 manually reviewed.
**Class collapse to binary:** Original 3-class labels (optimistic/neutral/pessimistic) had only 11 pessimistic examples — too few for a meaningful test split. Binary split gives 22%/78% positive/negative.

---

## Corpus

| Stat | Value |
|---|---|
| Companies | 49 (B3 large-caps; CCR absent from CVM open data) |
| Filing types | ITR (quarterly) + DFP (annual) |
| Date range | 2022-12-31 → 2025-09-30 |
| Total filings | 686 (504 ITR + 182 DFP) |
| Total metrics extracted | 4,651 |
| Total chunks indexed | 97,138 |
| ChromaDB size | ~1.8 GB |
| SQLite DB size | ~270 MB |

---

## Project Structure

```
cvm-intelligence/
├── src/
│   ├── acquisition/       # CVM CSV + PDF downloaders
│   ├── parsing/           # PyMuPDF extraction, section detection, chunking
│   ├── extraction/        # Rule-based metric extraction + validation
│   ├── nlp/               # BERTimbau fine-tuning, sentiment classifier
│   ├── rag/               # Indexer, retriever (hybrid+RRF), reranker, generator
│   └── db/                # SQLite schema + CRUD
├── app/
│   ├── app.py             # Streamlit entry point
│   └── pages/
│       ├── 1_query.py     # RAG query interface
│       ├── 2_dashboard.py # Financial metrics dashboard
│       ├── 3_sentiment.py # Sentiment timeline
│       └── 4_evaluation.py# System evaluation metrics
├── scripts/
│   ├── run_extraction.py  # Batch PDF extraction pipeline
│   ├── run_indexing.py    # Embed + ChromaDB + BM25 indexing
│   ├── run_validation.py  # Validation report
│   └── run_sentiment_scoring.py  # Classify 97k chunks
├── notebooks/             # 6 development notebooks (01–06)
├── models/
│   └── sentence_transformer/  # Fine-tuned BERTimbau weights
├── data/
│   ├── raw/               # CVM CSVs + PDFs (not committed)
│   ├── processed/         # Parsed chunks JSON, extracted metrics CSV
│   ├── evaluation/        # Retrieval test set, sentiment labels
│   └── cvm_metrics.db     # SQLite database
└── vectorstore/
    └── chromadb/          # ChromaDB persistence (~1.8 GB)
```

---

## Running Locally

**Prerequisites:** Python 3.10+, NVIDIA GPU for training (CPU inference works).

```bash
# 1. Clone and set up environment
git clone <repo-url>
cd cvm-intelligence
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

# 2. Set API key
echo "GEMINI_API_KEY=your_key_here" > .env

# 3. Run the app (requires pre-built DB and ChromaDB)
streamlit run app/app.py
```

**To reproduce the full pipeline from scratch:**

```bash
# Data acquisition
python src/acquisition/download_csvs.py
python src/acquisition/download_pdfs.py

# Extraction + validation
python scripts/run_extraction.py
python scripts/run_validation.py

# NLP training (GPU required, ~2h for sentence transformer)
python src/nlp/train_sentence_transformer.py
python src/nlp/train_sentiment.py

# Indexing (GPU helps, CPU viable)
python scripts/run_indexing.py

# Sentiment scoring (GPU recommended, ~15 min)
python scripts/run_sentiment_scoring.py

# Launch app
streamlit run app/app.py
```

---

## Design Decisions

See [`DECISIONS.md`](DECISIONS.md) for the full decision log: every problem encountered, the investigation, the fix, and its impact on metrics.

Key choices:
- **Rule-based extraction, not LLM** — deterministic, auditable, validated against ground truth
- **No LangChain/LlamaIndex** — every component (chunking, retrieval, reranking, generation) built explicitly for full understanding and control
- **ChromaDB + BM25 + RRF** — hybrid retrieval because Portuguese financial text has specific terminology that benefits from exact keyword matching
- **SQLite, not PostgreSQL** — single-file, zero-config, sufficient for 97k chunks and 4,651 metrics
- **Binary sentiment** — class imbalance (11 pessimistic out of 500 labels) made 3-class infeasible; binary gives a workable 22/78 split

---

## Notebooks

| Notebook | Contents |
|---|---|
| `01_data_exploration.ipynb` | CVM CSV structure, PDF inventory, filing manifest |
| `02_parsing_development.ipynb` | Section detection regex calibration, chunking validation |
| `03_extraction_validation.ipynb` | Extraction accuracy analysis, failure taxonomy |
| `04_embedding_training.ipynb` | BERTimbau fine-tuning, training curve |
| `05_retrieval_evaluation.ipynb` | Retrieval ablation study (4 configurations) |
| `06_sentiment_evaluation.ipynb` | Sentiment classifier training, confusion matrix, coverage |

---

## Tests

```bash
pytest tests/ -v
```

Tests cover: Brazilian number format parsing, metric extraction rules, RRF fusion correctness, section detection regex, retrieval interface contracts.
