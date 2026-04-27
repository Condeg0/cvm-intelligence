# CVM Filing Intelligence System

## Project Identity

An end-to-end pipeline that transforms raw Brazilian public company filings (CVM's ITR and DFP documents) into a queryable intelligence system. Two pipelines: (1) quantitative extraction of financial metrics from PDFs, validated against structured CSV ground truth, and (2) qualitative RAG over management commentary using a fine-tuned sentence transformer.

**Author:** Rafael Condé Gomes
**Full architecture:** See `DOCS/CVM_Filing_Intelligence_System_Architecture.md` (read-only reference — do NOT modify)

## Tech Stack

| Component | Tool | Notes |
|---|---|---|
| Language | Python 3.10+ | venv, NOT conda |
| PDF text extraction | PyMuPDF (fitz) | For text blocks + positional metadata |
| PDF table extraction | pdfplumber | For financial statement tables |
| Sentence embeddings | sentence-transformers + BERTimbau | `neuralmind/bert-base-portuguese-cased` |
| Vector store | ChromaDB | With metadata filtering, persisted to disk |
| Sparse retrieval | rank_bm25 | BM25 scoring for keyword matching |
| Cross-encoder reranker | cross-encoder/ms-marco-MiniLM-L-6-v2 | Reranks top-10 from RRF |
| Structured DB | SQLite | stdlib sqlite3, single-file database |
| API (generation + labeling) | Gemini 2.5 Flash | `google-generativeai` SDK, model: `models/gemini-2.5-flash` |
| App framework | Streamlit | Multi-page app, deployed to Streamlit Community Cloud |
| Charts | Plotly + Altair | Plotly for dashboard, Altair where simpler |
| ML framework | PyTorch | With CUDA on RTX A1000 (6GB VRAM) |
| Testing | pytest | Lightweight — test critical paths only |

## Hardware Constraints

- **GPU:** NVIDIA RTX A1000 — 6GB VRAM. This limits:
  - Fine-tuning batch size to 16 with gradient accumulation (effective 64)
  - Must use fp16 mixed precision
  - Max sequence length 256 tokens for BERTimbau
  - Cannot run local LLM generation — use Gemini API instead
- **RAM:** 15.3GB. ChromaDB + models must fit comfortably.
- **Deployment target:** Streamlit Community Cloud — 1GB RAM, CPU only. Pre-compute all embeddings. No GPU at inference.

## Directory Structure

```
cvm-intelligence/
├── CLAUDE.md                    # THIS FILE — project instructions
├── DOCS/                        # Learning materials (IGNORED by git and Claude Code)
├── requirements.txt
├── .env                         # GEMINI_API_KEY (not committed)
├── .streamlit/
│   └── secrets.toml             # Streamlit secrets (not committed)
├── data/
│   ├── raw/
│   │   ├── csvs/                # CVM structured data (ground truth)
│   │   └── pdfs/                # CVM filing PDFs
│   ├── processed/
│   │   ├── chunks/              # Parsed and chunked text (JSON per filing)
│   │   └── metrics/             # Extracted financial metrics (CSV)
│   ├── evaluation/
│   │   ├── retrieval_test_set.json
│   │   └── sentiment_labels.json
│   └── cvm_metrics.db           # SQLite database
├── models/
│   └── sentiment_classifier/    # Saved classifier weights
├── vectorstore/
│   └── chromadb/                # ChromaDB persistence directory
├── src/
│   ├── __init__.py
│   ├── config.py                # Centralized config (paths, constants, API setup)
│   ├── acquisition/
│   │   ├── __init__.py
│   │   ├── download_csvs.py
│   │   └── download_pdfs.py
│   ├── parsing/
│   │   ├── __init__.py
│   │   ├── pdf_parser.py        # PyMuPDF text extraction
│   │   ├── section_detector.py  # Regex-based section identification
│   │   └── chunker.py           # Section-aware paragraph chunking
│   ├── extraction/
│   │   ├── __init__.py
│   │   ├── metric_extractor.py  # Rule-based metric extraction
│   │   ├── value_parser.py      # Brazilian number format parsing
│   │   └── validator.py         # PDF vs CSV ground truth validation
│   ├── nlp/
│   │   ├── __init__.py
│   │   ├── train_sentence_transformer.py
│   │   ├── train_sentiment.py
│   │   └── embedder.py          # Embedding generation for indexing/querying
│   ├── rag/
│   │   ├── __init__.py
│   │   ├── indexer.py           # ChromaDB + BM25 indexing
│   │   ├── retriever.py         # Hybrid retrieval + RRF
│   │   ├── reranker.py          # Cross-encoder reranking
│   │   └── generator.py         # Gemini API context assembly + generation
│   └── db/
│       ├── __init__.py
│       └── schema.py            # SQLite schema + CRUD operations
├── notebooks/
│   ├── 01_data_exploration.ipynb
│   ├── 02_parsing_development.ipynb
│   ├── 03_extraction_validation.ipynb
│   ├── 04_embedding_training.ipynb
│   ├── 05_retrieval_evaluation.ipynb
│   └── 06_sentiment_evaluation.ipynb
├── app/
│   ├── app.py                   # Streamlit entry point
│   ├── pages/
│   │   ├── 1_query.py           # RAG query interface
│   │   ├── 2_dashboard.py       # Financial metrics dashboard
│   │   ├── 3_sentiment.py       # Sentiment timeline
│   │   └── 4_evaluation.py      # System evaluation metrics
│   └── components/
│       └── charts.py            # Reusable chart components
├── scripts/
│   ├── run_extraction.py        # Batch extraction pipeline
│   ├── run_indexing.py          # Batch embedding + ChromaDB indexing
│   └── run_validation.py        # Validation report generation
└── tests/
    ├── __init__.py
    ├── test_parser.py
    ├── test_extractor.py
    ├── test_value_parser.py
    └── test_retriever.py
```

## Project Phases

The project is built in 6 sequential phases. Each phase is independent and produces demonstrable artifacts. **Always check which phase is currently active before starting work.**

### Phase 1: Data Acquisition & Exploration
- Download CVM DADOS_ABERTOS CSVs for 50 largest B3 companies (3 years)
- Download corresponding PDF filings from CVM document portal
- Build manifest mapping company/period → CSV rows + PDF path
- Exploration notebook documenting data structure and quirks
- **Deliverable:** Populated `data/raw/`, manifest file, notebook `01_data_exploration.ipynb`

### Phase 2: PDF Parsing & Section Detection
- Text extraction via PyMuPDF with positional metadata
- Table extraction via pdfplumber
- Regex-based section detector (Balanço Patrimonial, DRE, Relatório da Administração, Notas Explicativas)
- Section-aware paragraph chunking (split on paragraphs, then sentences if >384 tokens)
- Manual verification on 20 sample documents to calibrate regex
- **Deliverable:** Working `src/parsing/` module, notebook `02_parsing_development.ipynb`

### Phase 3: Quantitative Extraction & Validation
- Rule-based metric extraction using Portuguese label patterns + table position heuristics
- Brazilian number format parsing (`1.234.567,89` → `1234567.89`)
- Validation against CSV ground truth: exact match rate, MAPE, coverage
- Failure taxonomy: table not found, wrong row, value parsing error, wrong column, section misidentified
- Populate SQLite database
- **Target:** >90% exact match rate on top 50 companies
- **Deliverable:** Populated SQLite DB, notebook `03_extraction_validation.ipynb`, working `src/extraction/`

### Phase 4: NLP — Embeddings & Sentiment
- Generate contrastive training pairs from parsed chunks
- Fine-tune BERTimbau as sentence transformer with `MultipleNegativesRankingLoss`
- Bootstrap 500 sentiment labels via Gemini API, manually review 100
- Train 3-class sentiment classifier (pessimistic/neutral/optimistic) on frozen embeddings
- **Targets:** Retrieval improvement over base model (any positive delta), sentiment F1 > 0.65
- **Deliverable:** Fine-tuned model on HuggingFace Hub, sentiment classifier, notebooks `04` and `06`

### Phase 5: Vector Store & RAG Pipeline
- Embed all chunks → ChromaDB with metadata (ticker, date, section, sentiment)
- Build BM25 index over same chunks
- Implement hybrid retrieval: dense + sparse + Reciprocal Rank Fusion
- Cross-encoder reranking (top 10 → top 5)
- Gemini API generation with structured prompt template
- Ablation study: dense-only vs sparse-only vs hybrid vs hybrid+reranker
- **Targets:** Recall@5 > 0.70, MRR > 0.65
- **Deliverable:** Working `src/rag/`, ChromaDB populated, notebook `05_retrieval_evaluation.ipynb`

### Phase 6: Application & Deployment
- 4-page Streamlit app: RAG query, financial dashboard, sentiment timeline, evaluation metrics
- Deploy to Streamlit Community Cloud
- Public URL accessible
- **Deliverable:** Live Streamlit app, README with architecture and results

## Coding Conventions

### Style
- **Formatting:** Use `black` defaults (line length 88). No need to run black — just follow the style.
- **Imports:** stdlib → third-party → local, separated by blank lines. Absolute imports only.
- **Type hints:** Use on all function signatures. Use `from __future__ import annotations` for modern syntax.
- **Docstrings:** Google style. Required on all public functions and classes. One-liner is fine for simple functions.
- **Naming:** snake_case for functions/variables, PascalCase for classes, UPPER_SNAKE for constants.

### Code Patterns
- **Config:** All paths, constants, and API setup live in `src/config.py`. Never hardcode paths in module files.
- **Logging:** Use `logging` stdlib, not `print()`. Set up per-module loggers: `logger = logging.getLogger(__name__)`.
- **Error handling:** Catch specific exceptions. Never bare `except:`. PDF parsing should catch and log failures, not crash the batch.
- **Data flow:** Functions take explicit inputs and return explicit outputs. No global state mutation. Dataclasses or TypedDicts for structured data.
- **File I/O:** Use `pathlib.Path` everywhere, never string concatenation for paths.

### Testing
- Tests go in `tests/`. Run with `pytest tests/`.
- Test critical logic: value parsing, metric extraction rules, RRF fusion, section detection regex.
- Do NOT test trivial wrappers or API calls.
- Use fixtures for sample PDF pages and CSV rows.

### Domain-Specific Rules
- **Never use LangChain or LlamaIndex.** Every component (chunking, retrieval, reranking, generation) is built explicitly.
- **Never use naive chunking** (RecursiveCharacterTextSplitter or fixed-size windows). Always section-aware, paragraph-boundary chunking.
- **Never use an LLM for metric extraction.** Quantitative extraction is rule-based and deterministic.
- **All financial values must be validated** against CSV ground truth before being trusted.
- **Brazilian number format:** Thousands separator is `.`, decimal separator is `,`. The value parser must handle this correctly. Example: `1.234.567,89` → `1234567.89`.

## Gemini API Usage

```python
import google.generativeai as genai
import os

genai.configure(api_key=os.getenv("GEMINI_API_KEY"))
model = genai.GenerativeModel("models/gemini-2.5-flash")

response = model.generate_content("Your prompt here")
print(response.text)
```

- API key stored in `.env` as `GEMINI_API_KEY` (loaded with `python-dotenv`)
- For Streamlit deployment: stored in `.streamlit/secrets.toml`
- Use Gemini for: RAG answer generation, sentiment bootstrap labeling
- Do NOT use Gemini for: metric extraction (rule-based), embedding generation (sentence-transformers)

## Key Data Sources

- **Structured CSVs (ground truth):** `https://dados.cvm.gov.br/dados/CIA_ABERTA/DOC/ITR/` and `.../DFP/`
- **PDF filings:** CVM document portal (linked from structured data)
- **Target companies:** 50 largest by market cap on B3 — Petrobras, Vale, Itaú, Bradesco, Ambev, B3, WEG, Suzano, etc.
- **Time range:** Last 3 years (~12 ITR + 3 DFP per company = ~750 documents)

## SQLite Schema

```sql
CREATE TABLE companies (
    ticker TEXT PRIMARY KEY,
    name TEXT,
    sector TEXT
);

CREATE TABLE filings (
    filing_id TEXT PRIMARY KEY,
    ticker TEXT REFERENCES companies(ticker),
    filing_type TEXT CHECK(filing_type IN ('ITR', 'DFP')),
    reference_date DATE,
    pdf_path TEXT,
    extraction_status TEXT CHECK(extraction_status IN ('success', 'partial', 'failed'))
);

CREATE TABLE metrics (
    metric_id INTEGER PRIMARY KEY AUTOINCREMENT,
    filing_id TEXT REFERENCES filings(filing_id),
    metric_name TEXT,
    extracted_value REAL,
    validated_value REAL,
    match_status TEXT CHECK(match_status IN ('exact', 'close', 'mismatch', 'missing')),
    percentage_error REAL
);

CREATE TABLE chunks (
    chunk_id TEXT PRIMARY KEY,
    filing_id TEXT REFERENCES filings(filing_id),
    section_name TEXT,
    chunk_text TEXT,
    sentiment_label TEXT,
    sentiment_score REAL,
    chromadb_id TEXT
);
```

## Target Metrics for Extraction

| Metric | Portuguese Label Patterns |
|---|---|
| Revenue | `Receita de Venda`, `Receita Líquida` |
| COGS | `Custo dos Bens`, `Custo dos Produtos` |
| Gross Profit | `Resultado Bruto`, `Lucro Bruto` |
| EBITDA | `EBITDA`, `LAJIDA` |
| Net Income | `Lucro (Prejuízo) Líquido` |
| Total Assets | `Ativo Total` |
| Total Equity | `Patrimônio Líquido` |
| Net Debt | `Dívida Líquida` |
| Operating Cash Flow | `Caixa das Atividades Operacionais` |

## Common Commands

```bash
# Activate environment
source .venv/bin/activate

# Run tests
pytest tests/ -v

# Run Streamlit app locally
streamlit run app/app.py

# Run batch extraction
python scripts/run_extraction.py

# Run validation report
python scripts/run_validation.py

# Run indexing (embed + store in ChromaDB)
python scripts/run_indexing.py
```

## What NOT to Do

- Do NOT modify anything in `DOCS/` — that directory is for learning materials only.
- Do NOT install packages globally — always use the venv.
- Do NOT use `print()` for logging — use the `logging` module.
- Do NOT commit API keys, `.env`, or `.streamlit/secrets.toml`.
- Do NOT add `data/raw/pdfs/` to git — PDFs are too large. Add to `.gitignore`.
- Do NOT use LangChain, LlamaIndex, or any RAG framework. Build every component explicitly.
- Do NOT use an LLM for number extraction. Rule-based only.
- Do NOT use fixed-size chunking. Section-aware, paragraph-boundary only.
- Do NOT over-engineer early phases. Each phase should be completed and tested before moving to the next.
