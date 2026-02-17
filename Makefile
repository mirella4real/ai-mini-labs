.PHONY: setup fmt lint test precommit

setup:
	uv sync

fmt:
	uv run ruff format .
	uv run black .

lint:
	uv run ruff check .

test:
	uv run python -m pytest -q

precommit:
	uv run pre-commit run --all-files
