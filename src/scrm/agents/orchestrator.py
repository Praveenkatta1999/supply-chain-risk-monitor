"""Orchestrator: takes a site list and date range, routes work, assembles the brief.

Pipeline, per site and with sites running concurrently:
    query_agent -> verifier (one article per story cluster) -> risk_scorer ("yes" only)
then once for all sites:
    report_writer -> one brief, ordered by risk score

The orchestrator is deterministic Python rather than an LLM: the order of steps is fixed,
so letting a model choose it would add cost and non-determinism without benefit. Every
step runs inside the caller's RunContext, so one run ID flows through all logs and model
calls are counted per agent. entity_resolver is not wired in yet.

Input:  BriefRequest
Output: RiskBrief (``run_pipeline`` also returns the per-site detail behind it)
"""

import asyncio

from scrm.agents import query_agent, report_writer, risk_scorer, verifier
from scrm.config import get_settings
from scrm.schemas import (
    BriefRequest,
    DateRange,
    PipelineResult,
    QueryRequest,
    ReportRequest,
    RiskBrief,
    RiskScore,
    ScoringRequest,
    Site,
    SiteRun,
    Verdict,
    VerificationRequest,
    VerifiedEvent,
)
from scrm.telemetry import RunContext, get_logger
from scrm.tools.bigquery_tool import BigQueryTool

NAME = "orchestrator"
INPUT_SCHEMA = BriefRequest
OUTPUT_SCHEMA = RiskBrief

CANDIDATES_PER_SITE = 20
# Model calls in flight at once, across all sites, to stay well inside Gemini quotas.
MAX_CONCURRENT_MODEL_CALLS = 8

log = get_logger(__name__)


async def _score_confirmed(
    verified: list[VerifiedEvent], site: Site, ctx: RunContext, limit: asyncio.Semaphore
) -> list[RiskScore]:
    async def one(event: VerifiedEvent) -> RiskScore | None:
        async with limit:
            try:
                return await risk_scorer.run(ScoringRequest(verified_event=event, site=site), ctx)
            except Exception:  # an unscored event still appears in the brief, ranked last
                log.exception("orchestrator.score_failed", extra={"event_id": event.event.event_id})
                return None

    confirmed = [v for v in verified if v.verdict is Verdict.YES]
    scores = await asyncio.gather(*(one(v) for v in confirmed))
    return [s for s in scores if s is not None]


async def _run_site(
    site: Site,
    date_range: DateRange,
    ctx: RunContext,
    tool: BigQueryTool,
    limit: asyncio.Semaphore,
) -> SiteRun:
    query = await query_agent.run(
        QueryRequest(site_ids=[site.site_id], date_range=date_range, limit=CANDIDATES_PER_SITE),
        ctx,
        tool=tool,
        sites=[site],
    )
    verified = await verifier.run_many(
        [VerificationRequest(event=e, site=site) for e in query.events], ctx, limit=limit
    )
    scores = await _score_confirmed(verified, site, ctx, limit)
    log.info(
        "orchestrator.site_done",
        extra={
            "site_id": site.site_id,
            "candidates": len(query.events),
            "confirmed": len(scores),
        },
    )
    return SiteRun(
        site=site,
        candidates=query.events,
        verified_events=verified,
        scores=scores,
        bytes_processed=query.bytes_processed,
    )


async def run_pipeline(
    request: BriefRequest, ctx: RunContext, *, tool: BigQueryTool | None = None
) -> PipelineResult:
    """Run the full pipeline and return the brief with the per-site detail behind it."""
    tool = tool or BigQueryTool.from_settings(get_settings())
    limit = asyncio.Semaphore(MAX_CONCURRENT_MODEL_CALLS)
    with ctx.bind():
        sites = await asyncio.to_thread(query_agent.load_sites, tool, request.site_ids)
        log.info("orchestrator.start", extra={"site_ids": [s.site_id for s in sites]})
        site_runs = await asyncio.gather(
            *(_run_site(site, request.date_range, ctx, tool, limit) for site in sites)
        )
        brief = await report_writer.run(
            ReportRequest(
                run_id=ctx.run_id,
                date_range=request.date_range,
                sites=sites,
                verified_events=[v for r in site_runs for v in r.verified_events],
                scores=[s for r in site_runs for s in r.scores],
            ),
            ctx,
        )
        log.info(
            "orchestrator.done",
            extra={"items": len(brief.items), "model_calls": dict(ctx.model_calls)},
        )
        return PipelineResult(brief=brief, site_runs=site_runs, model_calls=dict(ctx.model_calls))


async def run(request: BriefRequest, ctx: RunContext) -> RiskBrief:
    """Run the full pipeline for one request and return the brief."""
    return (await run_pipeline(request, ctx)).brief
