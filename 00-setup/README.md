# 00-setup — Repo Standards

This folder documents the shared engineering standards for this repo:

- Python 3.11 pinned via `.python-version`
- Dependency management via `uv` and `pyproject.toml`
- Formatting & linting via `black` + `ruff`
- Git hooks via `pre-commit`
- Secrets handled via `.env` (tracked via `.env.example`)

## Quick start

```bash
brew install uv
uv venv
uv sync
uv run pre-commit install
make precommit
