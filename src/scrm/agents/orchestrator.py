"""Orchestrator: takes a site list and date range, routes work, assembles the brief.

Pipeline, per site and with sites running concurrently:
    query_agent    up to 50 ranked story clusters (location and entity retrieval)
    triage         one model call: which stories are plausibly a disruption here
    verifier       reads one article per triaged story: direct / indirect / not relevant
    risk_scorer    severity and impact for confirmed stories
    investigator   corroborates findings with risk >= 12, and unverifiable ones
then once for all sites:
    report_writer  one brief: direct findings first, then indirect, then investigated
                   unverifiable stories; each section ordered by risk

The orchestrator is deterministic Python rather than an LLM: the order of steps is fixed,
so letting a model choose it would add cost and non-determinism without benefit. Every
step runs inside the caller's RunContext, so one run ID flows through all logs and model
calls are counted per agent. Model work across all sites shares one concurrency limit.

Input:  BriefRequest
Output: RiskBrief (``run_pipeline`` also returns the per-site detail behind it)
"""

import asyncio
from datetime import date

from scrm.agents import investigator, query_agent, report_writer, risk_scorer, triage, verifier
from scrm.config import get_settings
from scrm.schemas import (
    BriefRequest,
    DateRange,
    Event,
    Investigation,
    PipelineResult,
    QueryRequest,
    ReportRequest,
    RiskBrief,
    RiskScore,
    ScoringRequest,
    Site,
    SiteRun,
    TriageResult,
    Verdict,
    VerificationRequest,
    VerifiedEvent,
)
from scrm.telemetry import RunContext, get_logger
from scrm.tools.bigquery_tool import BigQueryTool

NAME = "orchestrator"
INPUT_SCHEMA = BriefRequest
OUTPUT_SCHEMA = RiskBrief

CANDIDATES_PER_SITE = triage.MAX_CANDIDATES
# Model work in flight at once, across all sites, to stay well inside Gemini quotas.
# An investigation holds one slot for all of its model and tool calls.
MAX_CONCURRENT_MODEL_CALLS = 8

log = get_logger(__name__)


async def _triage(
    site: Site, events: list[Event], ctx: RunContext, limit: asyncio.Semaphore
) -> TriageResult:
    async with limit:
        return await triage.run(site, events, ctx)


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


async def _investigate(
    verified: list[VerifiedEvent],
    scores: list[RiskScore],
    site: Site,
    search_until: date,
    tool: BigQueryTool,
    limit: asyncio.Semaphore,
) -> list[Investigation]:
    by_url = {str(s.source_url): s for s in scores}

    async def one(event: VerifiedEvent) -> Investigation | None:
        async with limit:
            score = by_url.get(str(event.evidence_url))
            return await investigator.run(event, score, site, search_until, tool)

    todo = [
        v for v in verified if investigator.needs_investigation(v, by_url.get(str(v.evidence_url)))
    ]
    results = await asyncio.gather(*(one(v) for v in todo))
    return [r for r in results if r is not None]


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
    triaged = await _triage(site, query.events, ctx, limit)
    chosen = {s.event_id for s in triaged.selected}
    to_verify = [e for e in query.events if e.event_id in chosen]
    verified = await verifier.run_many(
        [VerificationRequest(event=e, site=site) for e in to_verify], ctx, limit=limit
    )
    scores = await _score_confirmed(verified, site, ctx, limit)
    investigations = await _investigate(verified, scores, site, date_range.end, tool, limit)
    log.info(
        "orchestrator.site_done",
        extra={
            "site_id": site.site_id,
            "candidates": len(query.events),
            "triaged_in": len(to_verify),
            "confirmed": len(scores),
            "investigated": len(investigations),
        },
    )
    return SiteRun(
        site=site,
        candidates=query.events,
        triage=triaged,
        verified_events=verified,
        scores=scores,
        investigations=investigations,
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
                investigations=[i for r in site_runs for i in r.investigations],
                considered={r.site.site_id: len(r.candidates) for r in site_runs},
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
