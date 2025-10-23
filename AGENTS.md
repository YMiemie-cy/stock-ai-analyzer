# Repository Guidelines

## Project Structure & Module Organization
- `src/`: Core Python packages. `src/data.py` handles ingestion/caching, `src/model.py` wraps training artifacts, and `src/cli.py` orchestrates batch analysis.
- `app_server.py`: FastAPI entry point consumed by the `web/` frontend assets.
- `models/`: Serialized `.joblib` models per market profile; regenerate via the CLI when dependencies change.
- `data_cache/` & `reports/`: Cached OHLCV parquet files and exported summaries; safe to delete when troubleshooting.
- `tests/`: Pytest suites mirroring `src/` modules (e.g., `tests/test_data.py`).

## Build, Test, and Development Commands
- `source .venv/bin/activate`: Activate the project-managed Python 3.9 environment.
- `.venv/bin/pip install -r requirements.txt`: Install or refresh dependencies; rerun after dependency upgrades.
- `.venv/bin/python -m src.cli --tickers AAPL MSFT --train`: Train and emit signals from the CLI.
- `.venv/bin/uvicorn app_server:app --reload`: Launch the interactive web console at `http://localhost:8000`.
- `.venv/bin/pytest`: Execute the automated test suite.

## Coding Style & Naming Conventions
- Python files use 4-space indentation, type hints for public functions, and `snake_case` for variables/functions.
- Favor dataclasses, explicit enums, and small helpers in `src/core.py` and `src/utils/` (if adding).
- Keep comments actionable; document non-obvious financial logic, thresholds, or caching behaviors.
- When adding CLI flags, update `src/cli.py` help strings and README command samples.

## Testing Guidelines
- Use `pytest` with test modules named `test_*.py`; place fixtures alongside consuming tests.
- Add integration-style tests for new CLI workflows in `tests/test_cli.py` using temporary directories.
- Aim to cover edge cases around empty datasets, incompatible model artifacts, and API fallbacks.
- Run `pytest -k <target>` before pushing focused fixes; finish with a full run for release branches.

## Commit & Pull Request Guidelines
- Follow Conventional Commits (`feat`, `fix`, `refactor`, `docs`) as seen in `git log`.
- Scope commits around a single concern and include rationale in the body when behavior changes.
- Pull requests should detail motivation, summarize testing (`pytest`, CLI run, or manual web checks), and link related issues or tickets.
- Provide screenshots or terminal snippets when adjusting CLI output or web UI indicators.

## Environment & Configuration Tips
- Export secrets (e.g., `DEEPSEEK_API_KEY`) rather than hard-coding. Reference them with `os.getenv`.
- Parquet reads rely on `pyarrow`; reinstall with `pip install --force-reinstall pyarrow==14.0.2` if architecture mismatches appear.
- Clear `data_cache/` after modifying feature engineering logic to avoid stale datasets.
