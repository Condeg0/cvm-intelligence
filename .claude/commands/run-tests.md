Run the test suite and report results.

1. Activate the virtual environment: `source .venv/bin/activate`
2. Run: `pytest tests/ -v --tb=short`
3. If any tests fail, analyze the failure and suggest a fix.
4. Report: total tests, passed, failed, errors.

If no tests exist yet (empty test files), say which phase needs to create them:
- `test_value_parser.py` → Phase 3 (Extraction)
- `test_parser.py` → Phase 2 (Parsing)
- `test_extractor.py` → Phase 3 (Extraction)
- `test_retriever.py` → Phase 5 (RAG)
