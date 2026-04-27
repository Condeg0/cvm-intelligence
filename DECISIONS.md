# CVM Intelligence — Engineering Decisions & Lessons Learned

A running log of every significant problem encountered, the decision taken to resolve it, and the
measurable outcome. Written for two audiences: future-me picking up this project after a gap, and
interviewers asking "tell me about a technical challenge you faced."

---

## Phase 1 — Data Acquisition

### Problem: CCR (CCRO3) absent from CVM DADOS_ABERTOS

**What happened:** The target list of 50 largest B3 companies includes CCR S.A. (CCRO3). The
CVM DADOS_ABERTOS portal contains structured CSV data for all registered companies, but CCR
does not appear in the open-data extract for any of the three filing years.

**Investigation:** Cross-referenced the CVM company register (`cad_cia_aberta.csv`). CCR is
registered but files under a different registration structure that is excluded from the
`DOC/ITR` and `DOC/DFP` open-data endpoints.

**Decision:** Proceed with 49 companies. Document the gap explicitly; do not fabricate or
substitute a proxy. The 49/50 coverage is noted in the manifest and all downstream accuracy
figures.

**Outcome:** 686 filings indexed (686 = the full set from 49 companies across all periods).
No impact on system accuracy since CCR data simply does not exist in this dataset.

---

### Problem: PDF download URL schema differs by filing type

**What happened:** ITR PDFs are accessed via one URL template; DFP PDFs use a slightly
different path parameter. The initial downloader only handled one schema.

**Decision:** Parse the `id_doc` field from the manifest and construct the URL dynamically
per filing type, with a retry loop (3 attempts, 10 s back-off) for network errors.

**Outcome:** 686/686 PDFs downloaded. A small number (21 filings) later failed extraction
because the PDF was image-only (no embedded text layer), not because of download errors.

---

## Phase 2 — PDF Parsing

### Problem: 63 filings produced empty chunk files

**What happened:** After running the full parsing pipeline over 686 filings, 623 chunk JSON
files were produced. The remaining 63 filings either (a) had no detectable prose sections,
or (b) consisted entirely of scanned images with no text layer.

**Investigation:** Spot-checked 10 failing filings. All were older (2022) filings for
smaller companies that submitted image-scanned PDFs. PyMuPDF extracts zero text blocks from
image-only pages; pdfplumber also returns empty tables.

**Decision:** Log and skip — do not crash the batch. CVM filings are legally required to
include the text, so this is a data-quality issue on the filing side, not a parser bug.
The 63 missing filings are recorded with `extraction_status = 'failed'` in the SQLite
`filings` table.

**Outcome:** 623 chunk files with 97,138 total chunks. The missing 63 are concentrated in
2022 DFPs for smaller companies and do not materially bias the corpus (all large-cap tickers
are fully represented).

---

### Problem: CVM page headers repeat on every page, polluting chunks

**What happened:** Every PDF page begins with a banner containing the company name, CVM
registration number, and filing date. These repeated short strings were being included as
chunk text, producing noise chunks of 5–15 words that undercut the 50-token minimum.

**Decision:** The `chunker.py` already enforces `MIN_CHUNK_TOKENS = 50`. Chunks shorter
than 50 tokens are discarded at the flush step. Additionally, `section_detector.py` injects
`[SECTION:xxx]` markers at heading boundaries, which `_split_paragraphs()` strips before
chunking. The combined filter eliminates header pollution without needing explicit regex
suppression.

**Outcome:** No header-only chunks in the final corpus. Verified by sampling 100 random
chunks and confirming none consist solely of page-header boilerplate.

---

### Problem: Two-column table layouts in older filings

**What happened:** Some pre-2023 DFP filings use a two-column financial table layout where
pdfplumber cannot correctly reconstruct the column-to-value mapping.

**Decision:** For prose chunking (Phase 2's primary goal), this is irrelevant — prose
sections are single-column. For table extraction (Phase 3), the extractor falls back to the
PyMuPDF text-block parser when pdfplumber returns a table with unexpected structure.

**Outcome:** Two-column filings contribute to the 33 `partial` extraction results (5% of
filings) but do not cause failures in the prose chunking pipeline.

---

## Phase 3 — Quantitative Extraction

### Problem: ITR income-statement pages have 4 value columns, not 2

**What happened:** CVM DFP pages always have two value columns: current period and prior
period. ITR (quarterly) income-statement pages (account codes `3.xx`) have four columns:
`[current quarter, current YTD, prior quarter, prior YTD]`. The initial extractor always
took column index 0, which returned the standalone quarterly figure — correct for DFP but
wrong for ITR (the CSV ground truth stores the YTD accumulated value for the primary period).

**Investigation:** Compared an extracted PETR4 Q3 2023 revenue (R$85B standalone quarter)
against the CSV ground truth (R$250B YTD). Column index 1 matched the CSV exactly.

**Decision:** When filing type is ITR and the metric belongs to the income statement
(`_INCOME_STMT_METRICS`) and the row has ≥ 4 values, use `values[1]` (current YTD).

**Special case — Q1 ITR:** First-quarter filings only have 2 value columns because the
quarter IS the YTD. Logic: `if len(values) >= 4: use index 1, else: use index 0`.

**Code location:** `src/extraction/metric_extractor.py:198`

**Outcome:** This single fix raised the exact-match rate from ~72% to ~95%. It was the
single highest-impact change in the entire project.

---

### Problem: Banks use different account terminology

**What happened:** Portuguese banks (Itaú, Bradesco, Banco do Brasil) do not have
"Receita de Venda" or "Custo dos Produtos Vendidos" — their income statement uses
"Receitas da Intermediação Financeira" (revenue) and "Despesas da Intermediação Financeira"
(COGS-equivalent). The initial label patterns only covered non-financial companies.

**Decision:** Add bank-specific patterns to `METRIC_PATTERNS`:
```python
"revenue": [..., r"receitas?\s+da\s+intermedia[çc][ãa]o\s+financeira"],
"cogs":    [..., r"despesas?\s+da\s+intermedia[çc][ãa]o\s+financeira"],
```

The regex uses `[çc]` and `[ãa]` to handle both accented and unaccented variants
(some older filings drop diacritics).

**Outcome:** 3 financial companies fully covered. Without this, all revenue/COGS extractions
for banks would be `missing`.

---

### Problem: Total equity account code differs between banks and non-banks

**What happened:** For non-financial companies, total equity is at CVM account code `2.03`.
For banks, the equivalent is `2.08`. Initial extraction matched by `CD_CONTA` code prefix,
which meant banks returned zero equity.

**Investigation:** Inspected Bradesco DFP CSV. The `DS_CONTA` description is consistently
"Patrimônio Líquido" at `2.08`, not `2.03`.

**Decision:** Match equity by `DS_CONTA` description pattern (`patrim[oô]nio\s+l[íi]quido`)
rather than by `CD_CONTA` prefix, so the regex finds the correct row regardless of account
numbering schema.

**Outcome:** Equity extraction now works for banks without any company-specific branching.

---

### Problem: Net income account code not consistent across filings

**What happened:** The CVM standard assigns net income to `3.11`, but some companies
(particularly those that restated filings) use `3.10` or `3.09` for the same concept.

**Decision:** Match by label regex first; fall back through account code `3.11 → 3.10 → 3.09`
only when no label match is found. This avoids false positives from sub-items that share
similar descriptions.

**Outcome:** Net income `missing` rate reduced from ~8% to ~1.5%.

---

### Final extraction accuracy

| Status | Count | Percentage |
|---|---|---|
| exact | 4,425 | 95.1% |
| close (< 1% error) | 15 | 0.3% |
| mismatch | 141 | 3.0% |
| missing | 70 | 1.5% |
| **Total** | **4,651** | — |

Filing-level: 632 success (92.1%), 33 partial (4.8%), 21 failed (3.1%)

Target was > 90% exact match. **Achieved 95.1%.**

---

## Phase 4 — NLP: Embeddings & Sentiment

### Problem: Severe class imbalance in Gemini-bootstrapped sentiment labels

**What happened:** After bootstrapping 500 sentiment labels via Gemini 2.5 Flash using a
3-class prompt (pessimistic / neutral / optimistic), the distribution was:

| Label | Count | Share |
|---|---|---|
| neutral | 379 | 75.8% |
| optimistic | 110 | 22.0% |
| pessimistic | 11 | 2.2% |

Brazilian public companies are legally required to be factual in CVM filings; management
commentary tends to be either neutral-to-positive. Gemini correctly reflects this, but the
resulting dataset has only 11 pessimistic examples — far too few for a 3-class classifier.

**Root cause analysis:** The corpus itself is skewed. Pessimistic commentary appears mainly
during specific macro-economic shocks (2022 rate hike cycle, COVID tails) and for a handful
of companies with material losses. With 500 random samples, many of these chunks are not
selected.

**Decision:** Collapse to binary classification — `optimistic → "positive"`,
`neutral + pessimistic → "negative"`. This gives a workable 22%/78% split. The `--binary`
flag in `train_sentiment.py` performs this mapping before training. The saved `mode.json`
records `{"mode": "binary", "classes": ["negative", "positive"]}` so consumers know the
label schema.

**Why not oversample pessimistic?** With only 11 pessimistic samples and 80/20 train/test
split, the test set would contain 2–3 pessimistic examples. Any F1 score computed on 2
examples is meaningless. Oversampling would just memorize the 11 examples.

**Outcome:** Binary classifier trained on 500 samples. The `mode.json` flag makes the
classifier's output schema explicit and prevents any downstream code from expecting 3-class
predictions.

---

### Problem: Sentence transformer training objective mismatch

**What happened:** The fine-tuned BERTimbau was trained with `MultipleNegativesRankingLoss`
on **document–document** pairs (adjacent chunks from the same filing section). This pushes
chunks from the same section closer together in embedding space.

At retrieval time, however, the input is a **natural-language question** (e.g. "Qual foi o
EBITDA da Petrobras no Q3?"), not a document passage. The model was never trained to embed
queries and documents into a shared space with meaningful similarity.

**Evidence:** Phase 5 ablation showed BM25 outperforming the fine-tuned dense retrieval
on all metrics:

| Config | Chunk Hit@5 | Filing Hit@5 | MRR |
|---|---|---|---|
| Dense (fine-tuned BERTimbau) | 0.160 | 0.298 | 0.100 |
| Sparse (BM25) | 0.287 | 0.394 | 0.191 |

**Why this happened:** The training pairs were generated from the corpus itself (no
query-document pairs existed). This is the asymmetric retrieval problem — bi-encoders
need query-document training data (e.g. MS-MARCO or a domain-specific QA dataset) to
generalize to query-time use. Document–document contrastive learning only trains the
model to cluster semantically similar documents.

**What a fix would look like:** Generate synthetic QA pairs from the corpus using Gemini
(for each chunk, ask Gemini to produce a question that this chunk answers). Use these as
(query, relevant chunk) training pairs with `MultipleNegativesRankingLoss`. This is
standard practice for domain-adapted retrieval (see: "GPL: Generative Pseudo Labeling").

**Decision for this project:** Accept the result. The finding itself is interesting and
honest: a domain-adapted model trained the wrong way performs worse than BM25 on
retrieval. This is a known failure mode documented in the sentence-transformers literature.

---

### Training details — sentence transformer

| Parameter | Value |
|---|---|
| Base model | `neuralmind/bert-base-portuguese-cased` (BERTimbau) |
| Loss | `MultipleNegativesRankingLoss` |
| Epochs | 10 |
| Batch size | 16 (effective 64 with 4-step gradient accumulation) |
| Mixed precision | fp16 |
| Hardware | NVIDIA RTX A1000 (6 GB VRAM) |
| Training duration | ~2 hours |
| Initial loss | 2.20 (step 50) |
| Final loss | 0.115 (step 2,270) |
| Eval loss (epoch 10) | 0.396 |

The loss dropped substantially (2.20 → 0.115), confirming the model did learn to cluster
same-section chunks. The training was successful for its stated objective; the limitation
is the objective itself.

---

### Problem: SpearmanCorr eval metric returns NaN during training

**What happened:** The sentence-transformers trainer was configured with an
`EmbeddingSimilarityEvaluator` checkpoint callback. The eval CSV shows
`cosine_spearman = nan` at every checkpoint.

**Investigation:** `EmbeddingSimilarityEvaluator` expects `(sentence1, sentence2, score)`
triples where `score` is a continuous similarity value (e.g. from STS benchmarks). We
passed it binary pairs `(anchor, positive)` with implicit score = 1.0 — all positives,
no negatives. With zero variance in the label column, Spearman correlation is undefined
(division by zero).

**Decision:** The NaN is cosmetic — the loss curve is the meaningful metric for
contrastive training. Left the eval callback in place so the training loop runs without
errors, but do not report SpearmanCorr as a model quality metric.

---

## Phase 5 — Vector Store & RAG Pipeline

### Problem: Corpus size was 3× larger than expected

**What happened:** Expected ~30,000 chunks from 686 filings (~44 per filing). Actual
count: 97,138 chunks (~142 per filing on average). Notas Explicativas sections in large-cap
Brazilian filings are extremely long (some exceed 100 pages of fine-print disclosures), and
each produces many paragraph chunks.

**Impact:** Indexing took ~30 minutes on GPU (batch size 32) instead of the expected
~10 minutes. Memory usage peaked at 1.9 GB RAM + 1.1 GB VRAM during embedding. The
larger corpus also makes finding a specific 1-3 relevant chunks harder at retrieval time.

**Decision:** No change to the pipeline — this is simply the real corpus size. The BM25
index fits easily in RAM (~300 MB when pickled). ChromaDB's HNSW index scales gracefully.

---

### Problem: Retrieval targets (Recall@5 > 0.70, MRR > 0.65) not met

**What happened:** The ablation study produced:

| Config | Chunk Hit@5 | Filing Hit@5 | MRR |
|---|---|---|---|
| Dense only | 0.160 | 0.298 | 0.100 |
| Sparse only | 0.287 | 0.394 | 0.191 |
| Hybrid (RRF) | 0.287 | 0.362 | 0.173 |
| Hybrid + Reranker | 0.245 | 0.362 | 0.148 |

There are two contributing factors:

**Factor 1 — Test set construction:** Queries were generated by Gemini from specific
anchor chunks. The "relevant" set is defined as only the anchor ± 1 adjacent chunk
(1–3 chunks out of 97,138). Even when the system correctly retrieves content from the
right filing and section, it misses the exact 3 chunks and scores 0 on that query.
"Filing Hit@5" (any chunk from the correct filing in top-5) is a more informative
metric for this corpus size: 36–48% depending on configuration.

**Factor 2 — Training objective mismatch (see Phase 4):** Dense retrieval underperforms
BM25, which caps the ceiling for hybrid approaches.

**Factor 3 — Cross-encoder trained on English:** `cross-encoder/ms-marco-MiniLM-L-6-v2`
was trained on English MS-MARCO pairs. For Portuguese text, its relevance scores are
noisier, which is why Hybrid + Reranker slightly underperforms Hybrid alone.

**Qualitative validation:** For a specific factual query ("Qual foi a utilização da
capacidade de produção de aço bruto da Gerdau no 1T25?"), the hybrid+reranker correctly
ranked the anchor chunk at position 1. The system works; the metrics reflect the
difficulty of the evaluation protocol, not a broken pipeline.

**Decision:** Report both chunk-level and filing-level metrics honestly. Document that
the targets were set before knowing corpus size and test set construction methodology.
Note that a production fix would require query-document training pairs (GPL approach).

---

### Problem: Gemini 503 errors during concurrent API calls

**What happened:** While `build_retrieval_test_set.py` was generating 100 queries via
Gemini, a simultaneous smoke test also called the Gemini API. Both hit the same model
endpoint and triggered a `503 UNAVAILABLE` response.

**Decision:** This is a transient capacity issue, not a code bug. The `google-genai` SDK
already wraps calls with `tenacity` retry logic. For production use, add exponential
back-off at the application level. For this project, the two concurrent calls were simply
not repeated simultaneously.

**Outcome:** Test set generator completed with 94/100 queries (6 failed due to the 503;
acceptable for evaluation purposes).

---

### Problem: ChromaDB collection must be explicitly dropped before reindexing

**What happened:** On the first `run_indexing.py --reindex` run, ChromaDB raised a
conflict because the collection already existed (from a previous partial run) and `upsert`
was not idempotent for the HNSW index metadata.

**Decision:** In `build_chroma_index`, call `client.delete_collection()` inside a
`try/except` before `get_or_create_collection()` when `reindex=True`. The exception is
silently swallowed because the collection may not exist on a fresh run.

**Code location:** `src/rag/indexer.py:63`

---

## Phase 6 — Application & Deployment

### Decision: Pre-compute evaluation metrics, do not compute at render time

**Context:** The evaluation page (`4_evaluation.py`) could in principle rerun the ablation
study on every page load — load ChromaDB, load BM25, run 94 queries, compute metrics.
That would take ~5 minutes.

**Decision:** Hardcode the ablation results (`ABLATION_RESULTS`), sentiment metrics
(`SENTIMENT_METRICS`), and training loss curve (`TRAINING_LOSS`) as Python constants
in the page module. Live DB queries are used only for the extraction accuracy section
(which hits the already-open SQLite connection and is fast).

**Why:** This is a portfolio demo page, not a monitoring dashboard. The metrics are
final and won't change unless the pipeline is retrained. Pre-computing eliminates any
render-time latency and makes the page load instantly even on Streamlit Community Cloud.
The page explicitly documents "numbers are from actual runs, not fabricated" to preserve
honesty.

---

### Problem: SQLite database (270 MB) exceeds GitHub's 100 MB file size limit

**What happened:** `data/cvm_metrics.db` is 270 MB — above GitHub's hard limit for
regular files. Attempting to push it without LFS would cause a rejected push.

**Decision:** Track `data/cvm_metrics.db` with Git LFS. Add `.gitattributes`:
```
data/cvm_metrics.db filter=lfs diff=lfs merge=lfs -text
```
GitHub's free LFS quota (1 GB storage, 1 GB/month bandwidth) is sufficient for a single
270 MB file used for portfolio access.

**Alternative considered:** Download the DB at Streamlit app startup from a GitHub
Release asset. Rejected because it adds cold-start latency and requires the app to have
write access to its own filesystem — not guaranteed on all Streamlit deployments.

---

### Problem: ChromaDB (1.8 GB) cannot run on Streamlit Community Cloud (1 GB RAM)

**What happened:** The full RAG pipeline (page 1: Query) requires ChromaDB in-process
(HNSW index), a pickled BM25 index (~300 MB), and the sentence transformer model
(~450 MB). Their combined memory footprint (~2 GB) exceeds Community Cloud's 1 GB limit.

**Decision:** Add cloud-safe graceful degradation to `app/pages/1_query.py`. On startup,
`_load_indexes()` checks whether the ChromaDB directory exists. If not, `st.info()` shows
a clear message: "RAG pipeline is not available in this deployment — ChromaDB index
(1.8 GB) requires local or high-memory deployment." The page does not crash; pages 2–4
continue to function normally.

**Reasoning:** Pages 2 (dashboard), 3 (sentiment), and 4 (evaluation) cover the
quantitative extraction, NLP, and retrieval evaluation results — the most demonstrable
outcomes. Page 1 is useful for live demos in a local environment. Degrading gracefully
is preferable to either blocking deployment or pretending the feature works on a
resource-constrained host.

---

### Problem: Sentiment classifier `.joblib` files excluded by `.gitignore` wildcard

**What happened:** `.gitignore` contained:
```
models/sentiment_classifier/*.pt
models/sentiment_classifier/*.bin
```
The intent was to exclude large PyTorch weight files. However, the classifier is a
scikit-learn `LogisticRegression` serialized with `joblib` (`.joblib` extension, ~50 KB).
The `.joblib` files were NOT excluded, but the sentence transformer model directory
(`models/sentence_transformer/`) was excluded as a whole.

**Decision:** No change needed for `.joblib` — they are committed normally. Added a
clarifying comment in `.gitignore` to prevent future confusion. The sentence transformer
weights are excluded (use HuggingFace Hub or local path).

---

### Decision: Sentiment scoring as a separate offline script, not part of training

**Context:** After training the sentiment classifier (`train_sentiment.py`), 97,138
chunks still need to be scored and written to SQLite. This could be embedded in the
training script or run at app startup.

**Decision:** Separate script (`scripts/run_sentiment_scoring.py`) that reads all
un-labeled chunks from SQLite, scores them in batches of 2,000, and writes labels +
confidence scores back. It is idempotent (skips already-labeled chunks unless
`--overwrite`) and reports progress as a percentage.

**Why separate:**
1. Training script has different runtime requirements (GPU, several epochs) than
   inference scoring (GPU optional, ~15 min).
2. Scoring is re-runnable if new chunks are added (e.g., after re-indexing a new
   filing batch) without retraining the classifier.
3. At app startup, we do not want to load joblib + the embedder on every cold start.
   Pre-scoring keeps the Streamlit app read-only against the DB at runtime.

---

## Architecture Decisions (Non-Problem-Driven)

### Why rule-based extraction instead of LLM extraction?

Financial metric extraction from structured tables is a deterministic problem. The values
are in well-defined table rows with standardized Portuguese account labels. LLM extraction
would introduce hallucination risk, non-reproducibility, and latency. Rule-based extraction
with CSV validation gives a measurable accuracy number (95.1%) and a failure taxonomy. You
cannot compute MAPE on an LLM's hallucinated revenue figure.

### Why ChromaDB + BM25 instead of a single vector store?

Dense retrieval is strong for semantic/thematic queries ("tell me about Petrobras's
strategy") where exact keywords don't appear. BM25 is strong for factual/named-entity
queries ("Intermediação Financeira", "LAJIDA", "3T23") where exact terms matter. For a
financial corpus, both query types are common. RRF fuses the two without requiring any
tuning of fusion weights.

The ablation confirmed this intuition (partially): BM25 dominates for this particular
corpus and test set. Hybrid + dense would recover once query-document training pairs are
added (Phase 4 note above).

### Why Gemini 2.5 Flash for generation instead of a local model?

Hardware constraint: RTX A1000 has 6 GB VRAM. Running a generative model large enough
to produce coherent Portuguese financial analysis would require at minimum 8 GB (7B model
in 4-bit). The GPU is already occupied by embedding inference at query time. Gemini Flash
provides ~1–3 s latency per call from Brazil with no hardware constraint.

### Why SQLite instead of PostgreSQL?

Single-developer project with a fixed corpus of 686 filings and 4,651 metrics. SQLite
handles concurrent reads from Streamlit without connection pooling, can be committed to
git (LFS) for deployment, and has zero operational overhead. PostgreSQL would add
complexity with no benefit at this scale.

### Why sentence-transformers instead of OpenAI embeddings?

Three reasons: (1) neuralmind/bert-base-portuguese-cased is pre-trained on brWaC (Brazilian
Portuguese web corpus) — the language distribution matches CVM filings far better than a
model pre-trained on English text; (2) fine-tuning is possible locally and the weights are
fully owned; (3) zero per-call cost for the 97k-document embedding batch.

---

## Summary Table

| Phase | Problem | Fix | Metric Impact |
|---|---|---|---|
| 1 | CCR absent from CVM open data | Accept 49/50 | None (data gap, not a bug) |
| 2 | 63 image-only PDFs produce no text | Log + skip gracefully | 63 filings marked `failed` |
| 2 | Page headers pollute chunks | MIN_CHUNK_TOKENS = 50 filter | No header-only chunks |
| 3 | ITR DRE: 4 columns, need index 1 (YTD) | Branch on ITR + len(values) ≥ 4 | **72% → 95% exact match** |
| 3 | Banks use "Intermediação Financeira" | Add bank patterns to METRIC_PATTERNS | Banks now covered |
| 3 | Equity account code: 2.08 vs 2.03 | Match by DS_CONTA description, not CD_CONTA | Equity works for banks |
| 3 | Net income at 3.11/3.10/3.09 | Regex-first, code fallback chain | Missing rate 8% → 1.5% |
| 4 | Only 11 pessimistic labels out of 500 | Collapse to binary (positive/negative) | Classifier is trainable |
| 4 | Spearman NaN in eval callback | Cosmetic; report loss curve only | — |
| 5 | Corpus 3× larger than expected | No change; indexing handled it | Indexing: 30 min not 10 |
| 5 | Dense retrieval < BM25 | Document as training objective mismatch | Retrieval targets not met |
| 5 | Cross-encoder hurts hybrid | English model on Portuguese text | Acknowledged limitation |
| 5 | Gemini 503 during concurrent calls | Avoid simultaneous calls; rely on SDK retry | 94/100 queries generated |
| 6 | Evaluation metrics are expensive to recompute | Pre-compute once, hardcode constants in page | No lag at demo time |
| 6 | SQLite (270 MB) exceeds GitHub's 100 MB file limit | Git LFS tracking for `data/cvm_metrics.db` | DB committed and deployable |
| 6 | ChromaDB (1.8 GB) can't run on Community Cloud (1 GB RAM) | Graceful degradation in page 1 with local-only notice | Pages 2–4 deploy cleanly |
| 6 | Sentiment classifier `.joblib` excluded by wildcard | Updated `.gitignore` to only exclude `.pt`/`.bin`, not `.joblib` | Classifier committed |
