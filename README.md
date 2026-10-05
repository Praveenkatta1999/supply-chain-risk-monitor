# Supply Chain Risk Monitor

A multi-agent system that watches global news (the [GDELT](https://www.gdeltproject.org/)
dataset in BigQuery) for events that could disrupt a company's suppliers, ports and
shipping routes, and writes a daily risk brief in which every claim links to its source.

The demo company is a fictional laptop maker with 22 real sites: 12 supplier factories,
6 ports and 4 shipping chokepoints.

> Status: first working slice. `query_agent`, `verifier` and `report_writer` run end to end
> for one site (`uv run --env-file .env python scripts/run_brief.py --site P05`); the entity
> resolver, risk scorer and orchestrator are still stubs.

## How it works

```
BriefRequest(sites, dates)
  -> query_agent       SQL over scrm.* tables (dry-run guarded)        -> Event[]
  -> entity_resolver   messy GDELT names -> known companies            -> EntityMatch[]
  -> verifier          reads the article, rejects false positives      -> VerifiedEvent[]
  -> risk_scorer       severity and impact, 1 to 5, with a reason      -> RiskScore[]
  -> report_writer     Markdown brief with a source link on every claim -> RiskBrief
```

The `orchestrator` runs these steps in order as plain Python and gives each run a run ID
that appears in every log line, so one brief can be traced end to end.

Built with [Google ADK](https://adk.dev/), Gemini on Gemini Enterprise Agent Platform
(formerly Vertex AI), BigQuery, Pydantic v2 and FastAPI.

## Design choices

- **Cost guard first.** Every query is dry-run before execution and refused above a
  configurable byte limit (default 10 GiB). It is also refused if it touches a table outside
  the `scrm` dataset or is anything other than a `SELECT`.
- **Structured output only.** Every agent returns a Pydantic model, never free text.
- **Citations are enforced by validation.** A `RiskBrief` whose Markdown omits a source URL
  fails validation.
- **No keys.** Authentication uses Application Default Credentials only.

## Data

Dataset `scrm` (US multi-region), built from public GDELT tables by the SQL in `sql/`:

| Table | Contents |
|---|---|
| `scrm.sites` | The 22 monitored sites with coordinates and radius |
| `scrm.events_near_sites` | GDELT coded events (protests, strikes, sanctions, violence) near sites |
| `scrm.gkg_near_sites` | News articles with disruption themes near sites, with URL, themes, orgs, tone |

## Getting started

Requires Python 3.12 and [uv](https://docs.astral.sh/uv/).

```bash
gcloud auth application-default login
cp .env.example .env          # set GOOGLE_CLOUD_PROJECT
uv sync
uv run pytest
uv run uvicorn scrm.api:app --reload --port 8080
```

Then open http://localhost:8080/docs. The Makefile wraps these commands (`make test`, `make lint`, `make api`, `make evals`).

## Layout

```
src/scrm/agents/     one module per agent
src/scrm/tools/      bigquery_tool.py (guarded queries), article_fetcher.py
src/scrm/schemas.py  all Pydantic models
src/scrm/config.py   settings from environment variables
src/scrm/telemetry.py  JSON logging with run IDs
src/scrm/api.py      FastAPI: POST /brief, GET /health
evals/               labelled cases and the eval runner
sql/                 SQL that builds the scrm tables
tests/               mirrors src/
```
