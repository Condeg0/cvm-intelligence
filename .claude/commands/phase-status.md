Analyze the current state of the project and report which phase we're in based on what actually exists on disk.

Check each phase's deliverables:

**Phase 1 — Data Acquisition:**
- [ ] `data/raw/csvs/` contains CVM CSV files
- [ ] `data/raw/pdfs/` contains PDF filings
- [ ] Manifest file exists mapping companies to files
- [ ] `notebooks/01_data_exploration.ipynb` has content beyond placeholder

**Phase 2 — PDF Parsing:**
- [ ] `src/parsing/pdf_parser.py` has working implementation (not just stubs)
- [ ] `src/parsing/section_detector.py` has regex patterns defined
- [ ] `src/parsing/chunker.py` has section-aware chunking logic
- [ ] `data/processed/chunks/` contains parsed JSON files
- [ ] `notebooks/02_parsing_development.ipynb` has content

**Phase 3 — Quantitative Extraction:**
- [ ] `src/extraction/metric_extractor.py` has working implementation
- [ ] `src/extraction/value_parser.py` handles Brazilian number formatting
- [ ] `src/extraction/validator.py` compares against CSV ground truth
- [ ] `data/cvm_metrics.db` exists and has rows in the metrics table
- [ ] `notebooks/03_extraction_validation.ipynb` has accuracy results

**Phase 4 — NLP:**
- [ ] `src/nlp/train_sentence_transformer.py` has training pipeline
- [ ] Training pairs generated in `data/processed/`
- [ ] Fine-tuned model exists (locally or on HuggingFace)
- [ ] `src/nlp/train_sentiment.py` has classifier training
- [ ] Sentiment labels exist in `data/evaluation/sentiment_labels.json`

**Phase 5 — RAG Pipeline:**
- [ ] `vectorstore/chromadb/` contains indexed data
- [ ] `src/rag/retriever.py` implements hybrid retrieval + RRF
- [ ] `src/rag/reranker.py` implements cross-encoder reranking
- [ ] `src/rag/generator.py` implements Gemini API generation
- [ ] `data/evaluation/retrieval_test_set.json` exists with labeled queries
- [ ] `notebooks/05_retrieval_evaluation.ipynb` has ablation results

**Phase 6 — Application:**
- [ ] `app/app.py` runs without errors
- [ ] All 4 pages functional (query, dashboard, sentiment, evaluation)
- [ ] README.md has architecture diagram and results

Print a clear summary: current phase, what's done, what's next, any blockers you can detect (missing files, empty modules, broken imports).
