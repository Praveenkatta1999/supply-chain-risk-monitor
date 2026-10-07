# CLAUDE.md

## Project
Multi-agent system that monitors GDELT news in BigQuery (dataset `scrm`, US multi-region)
for events that could disrupt a fictional laptop maker's 22 sites (12 suppliers, 6 ports,
4 chokepoints) and writes a daily Markdown risk brief with citations. Portfolio project:
readability and best practice beat feature count. Stack: Python 3.12, uv, Google ADK,
Gemini on Gemini Enterprise Agent Platform (formerly Vertex AI), BigQuery, Pydantic v2, FastAPI.

Pipeline (`src/scrm/agents/`), run by `orchestrator` (plain Python, sites in parallel):
1. `query_agent`: fixed, parameterised SQL over `scrm.*` tables (no LLM); ranks candidates
   (`ranking.py`) and groups same-story articles (`clustering.py`) -> `QueryResult`
2. `entity_resolver`: GDELT actor/org names -> companies in `scrm.sites` (stub, not wired in)
3. `verifier`: one article per story cluster; yes/no/unverifiable with a verbatim quote
   -> `VerifiedEvent`
4. `risk_scorer`: severity and impact 1-5 with a one-sentence reason, "yes" only -> `RiskScore`
5. `report_writer`: one brief for all sites, ordered by risk (severity x impact) -> `RiskBrief`
`orchestrator.run_pipeline` returns `PipelineResult` (brief + per-site detail + model calls).

## Commands
- Install: `uv sync`
- Test: `uv run pytest`
- Lint: `uv run ruff check . && uv run ruff format --check .`
- API locally: `uv run uvicorn scrm.api:app --reload --port 8080`
- Evals (verifier vs human labels: accuracy, precision, recall, disagreements):
  `uv run python -m evals.run_evals`
- Label eval cases: `uv run python -m scripts.label_evals`
- Brief (calls Gemini and BigQuery):
  `uv run --env-file .env python scripts/run_brief.py --sites P05 S01 C01 P02 --days 30 [--end YYYY-MM-DD]`
  (saves the full result to `data/runs/<run_id>.json`, gitignored)
- BigQuery smoke test: `uv run --env-file .env python scripts/check_bigquery.py`
- Export scrm tables to Parquet (`data/*.parquet`, gitignored; checks row counts):
  `uv run --env-file .env python scripts/export_parquet.py`
(`make install|test|lint|format|api|evals` wraps these.)

## Hard constraints
- All BigQuery access goes through `scrm.tools.bigquery_tool.BigQueryTool`. It dry-runs
  every query and refuses it if estimated bytes exceed `SCRM_MAX_BYTES_BILLED`
  (default 10 GiB). Never call `bigquery.Client.query` anywhere else.
- Queries may read only from the `scrm` dataset, never from `gdelt-bq` directly, and
  must be `SELECT`. The guard enforces this through dry-run metadata; do not weaken it.
- Every agent returns a Pydantic model from `scrm/schemas.py`, never free text.
- Every claim in a brief carries the source URL it came from (`BriefItem.source_urls`).
  Models never write URLs or IDs: the report writer cites findings by ID and Python maps
  them back; confirmed findings a draft drops are still reported (verifier's reason).
- A "yes" verdict needs a quote found verbatim in the article; otherwise "unverifiable".
- Logging is structured JSON (`scrm.telemetry`). Every agent takes a `RunContext`, and
  work runs inside `ctx.bind()` so `run_id` appears on every log line.

## Conventions
- Type hints everywhere; small, single-purpose functions; Pydantic models at boundaries.
- No new dependencies without asking first. `pyarrow` is a dev-only dependency (Parquet
  export); runtime code must not import it.
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
- [x] FastAPI `/health`; `/brief` runs the orchestrator (400 on unknown site_ids)
- [ ] Copy the three SQL files into `sql/`
- [ ] Expose `BigQueryTool` to `query_agent` as an ADK function tool (deferred: query_agent
      uses fixed SQL for now; revisit only if free-form questions are needed)
- [x] Implement `article_fetcher` (httpx, timeout, size cap)
- [x] First slice end to end for one site: `query_agent` -> `verifier` -> `report_writer`
      via `scripts/run_brief.py`, with tests
- [x] Story clustering in the query stage; cluster size as a ranking signal
- [x] Site-match ranking signal (site, operator or company named, not just the city)
- [x] `risk_scorer` (LlmAgent, severity and impact 1-5, one-sentence reason), with tests
- [x] Orchestrator: sites in parallel, one brief ordered by risk; `/brief` returns it
- [x] Eval dataset: 20 P05 candidates from run b3467493805d with verifier verdicts;
      `scripts/label_evals.py` records human labels; `run_evals.py` reports agreement
- [x] Human-labelled the 20 P05 eval cases; `run_evals.py` reports metrics. Verifier vs
      human (2026-10-07): accuracy 95%, precision 100% (5/5), recall 83% (5/6), coverage
      90%; one miss (NRC, Chinese dumping and Dutch factory closures: human yes, verifier no)
- [ ] Add labelled cases for S01, C01 and P02 (20 cases is too few to tune prompts on)
- [x] Exported `scrm.sites`, `events_near_sites`, `gkg_near_sites` to `data/*.parquet`
      (2026-10-07, row counts match BigQuery) before the GCP credits expired
- [ ] Ranking is still weak: verifier-judged false positive rate on 2026-09-06..10-05 was
      P05 65% (baseline 72%), C01 78%, S01 100%, P02 100%. Next: theme lists per site
      type (factory themes for S01), negative themes (crime, culture), source language
- [ ] Verifier looks lenient on chokepoints (C01 accepted Iran-proxy and Saudi pipeline
      stories); P05 labels show no false "yes", so label C01 cases before tuning the prompt
- [ ] `entity_resolver`: implement and wire in (stub today)
- [ ] Containerise and deploy (ask first)
