Scaffold the full project directory structure as defined in CLAUDE.md.

Create all directories and placeholder files with proper `__init__.py` files. For each Python module file, include:
- Module docstring explaining its purpose (from the architecture)
- Required imports (leave empty function stubs with docstrings and `raise NotImplementedError`)
- Type hints on all stubs

Also create:
- `src/config.py` with all path constants using `pathlib.Path`, project root detection, and Gemini API setup via `python-dotenv`
- `src/db/schema.py` with the full SQLite schema from CLAUDE.md (CREATE TABLE statements) and an `init_db()` function
- Empty notebooks (just title + description markdown cells) numbered 01–06
- `app/app.py` with Streamlit multi-page app boilerplate
- `scripts/` directory with placeholder batch scripts
- `tests/` directory with placeholder test files importing pytest

Do NOT create any files inside `DOCS/`.
Do NOT create `.env` or `.streamlit/secrets.toml` — just log that the user needs to create these manually.

After scaffolding, run `python -c "from src import config; print('Config OK')"` to verify the package structure works.
