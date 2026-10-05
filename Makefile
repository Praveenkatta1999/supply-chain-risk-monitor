.PHONY: install test lint format api evals

install:
	uv sync

test:
	uv run pytest

lint:
	uv run ruff check .
	uv run ruff format --check .

format:
	uv run ruff check --fix .
	uv run ruff format .

api:
	uv run uvicorn scrm.api:app --reload --port 8080

evals:
	uv run python -m evals.run_evals
