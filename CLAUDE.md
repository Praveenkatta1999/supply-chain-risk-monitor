# CLAUDE.md

## Project
Multi-agent system that monitors GDELT news in BigQuery (dataset `scrm`, US multi-region)
for events that could disrupt a fictional laptop maker's 22 sites (12 suppliers, 6 ports,
4 chokepoints) and writes a daily Markdown risk brief with citations. Portfolio project:
readability and best practice beat feature count. Stack: Python 3.12, uv, Google ADK,
Gemini on Gemini Enterprise Agent Platform (formerly Vertex AI), BigQuery, Pydantic v2, FastAPI.

Pipeline (`src/scrm/agents/`), run in order by `orchestrator` (plain Python):
1. `query_agent`: fixed, parameterised SQL over `scrm.*` tables, ranked in Python (no LLM) -> `QueryResult`
2. `entity_resolver`: GDELT actor/org names -> companies in `scrm.sites` -> `EntityResolutionResult`
3. `verifier`: fetches article, yes/no/unverifiable with a verbatim quote -> `VerifiedEvent`
4. `risk_scorer`: severity and impact 1-5 with reason -> `RiskScore`
5. `report_writer`: Markdown brief with source links -> `RiskBrief`

## Commands
- Install: `uv sync`
- Test: `uv run pytest`
- Lint: `uv run ruff check . && uv run ruff format --check .`
- API locally: `uv run uvicorn scrm.api:app --reload --port 8080`
- Evals: `uv run python -m evals.run_evals`
- First slice (one site): `uv run --env-file .env python scripts/run_brief.py --site P05 --days 30`
- BigQuery smoke test: `uv run --env-file .env python scripts/check_bigquery.py`
(`make install|test|lint|format|api|evals` wraps these.)

## Hard constraints
- All BigQuery access goes through `scrm.tools.bigquery_tool.BigQueryTool`. It dry-runs
  every query and refuses it if estimated bytes exceed `SCRM_MAX_BYTES_BILLED`
  (default 10 GiB). Never call `bigquery.Client.query` anywhere else.
- Queries may read only from the `scrm` dataset, never from `gdelt-bq` directly, and
  must be `SELECT`. The guard enforces this through dry-run metadata; do not weaken it.
- Every agent returns a Pydantic model from `scrm/schemas.py`, never free text.
- Every claim in a brief carries the source URL it came from (`BriefItem.source_urls`).
- Logging is structured JSON (`scrm.telemetry`). Every agent takes a `RunContext`, and
  work runs inside `ctx.bind()` so `run_id` appears on every log line.

## Conventions
- Type hints everywhere; small, single-purpose functions; Pydantic models at boundaries.
- No new dependencies without asking first.
- Every new tool gets tests alongside it (`tests/` mirrors `src/scrm/`).
- Model IDs come from settings (`SCRM_GEMINI_MODEL`); check current ADK and Gemini docs
  before changing ADK code or versions. Do not rely on memory.

## Cost and safety
- Never run a BigQuery query without the dry-run guard.
- Never commit credentials. Auth is Application Default Credentials only; no key files.
- Ask before deploying, creating GCP resources, or running anything that touches GCP.

## Status
- [x] Project skeleton, schemas, config, JSON logging with run IDs
- [x] BigQuery dry-run guard (cost, dataset allowlist, SELECT only) with tests
- [x] FastAPI `/health`; `/brief` returns 501 until the orchestrator is built
- [ ] Copy the three SQL files into `sql/`
- [ ] Expose `BigQueryTool` to `query_agent` as an ADK function tool (deferred: query_agent
      uses fixed SQL for now; revisit only if free-form questions are needed)
- [x] Implement `article_fetcher` (httpx, timeout, size cap)
- [x] First slice end to end for one site: `query_agent` -> `verifier` -> `report_writer`
      via `scripts/run_brief.py`, with tests
- [ ] Implement remaining agents (`entity_resolver`, `risk_scorer`), each with tests
- [ ] Improve candidate ranking (first run: 13 of 20 P05 candidates were false positives)
- [ ] Implement orchestrator, so `/brief` returns a real brief
- [ ] Label 20+ eval cases in `evals/dataset.jsonl`; add metrics to `run_evals.py`
- [ ] Containerise and deploy (ask first)
