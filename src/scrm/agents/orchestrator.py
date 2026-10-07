"""Orchestrator: takes a site list and date range, routes work, assembles the brief.

Pipeline. Per site, with sites running concurrently:
    query_agent    up to 50 ranked story clusters (location and entity retrieval)
    triage         one model call: which stories are plausibly a disruption here
    verifier       reads one article per triaged story: direct / indirect / not relevant
    risk_scorer    severity and impact for confirmed stories
Then once, for all sites together:
    report_writer  drafts claims, merging findings about the same event (I1, I2, ...)
    investigator   once per story: each claim with risk >= 12, each unverifiable story
    reviewer       critic loop: checks claims against the evidence, sends corrections
                   back to the report writer (at most 2 rounds), sets status/confidence
    report_writer  renders the brief: direct, indirect, investigated unverifiable stories

Investigating after claims are drafted means one investigation per story, however many
articles the story was built from. The orchestrator is deterministic Python rather than
an LLM: the order of steps is fixed, so letting a model choose it would add cost and
non-determinism without benefit. Every step runs inside the caller's RunContext, so one
run ID flows through all logs and model calls are counted per agent. Model work across
all sites shares one concurrency limit.

Input:  BriefRequest
Output: RiskBrief (``run_pipeline`` also returns the detail behind it)
"""

import asyncio
from datetime import date

from scrm.agents import (
    investigator,
    query_agent,
    report_writer,
    reviewer,
    risk_scorer,
    triage,
    verifier,
)
from scrm.config import get_settings
from scrm.schemas import (
    BriefItem,
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
    log.info(
        "orchestrator.site_done",
        extra={
            "site_id": site.site_id,
            "candidates": len(query.events),
            "triaged_in": len(to_verify),
            "confirmed": len(scores),
        },
    )
    return SiteRun(
        site=site,
        candidates=query.events,
        triage=triaged,
        verified_events=verified,
        scores=scores,
        bytes_processed=query.bytes_processed,
    )


async def _investigate(
    items: list[BriefItem],
    unverifiable: list[VerifiedEvent],
    events: reviewer.EventIndex,
    sites: dict[str, Site],
    search_until: date,
    tool: BigQueryTool,
    limit: asyncio.Semaphore,
) -> list[Investigation]:
    """One investigation per high-risk claim and per unverifiable story, concurrently."""

    async def claim(item: BriefItem) -> Investigation | None:
        async with limit:
            return await investigator.investigate_claim(
                item, reviewer.sources_of(item, events), sites[item.site_id], search_until, tool
            )

    async def story(event: VerifiedEvent) -> Investigation | None:
        async with limit:
            return await investigator.investigate_story(
                event, sites[event.event.site_id], search_until, tool
            )

    jobs = [claim(i) for i in items if investigator.needs_investigation(i)]
    jobs += [story(v) for v in unverifiable]
    results = await asyncio.gather(*jobs)
    return [r for r in results if r is not None]


async def run_pipeline(
    request: BriefRequest, ctx: RunContext, *, tool: BigQueryTool | None = None
) -> PipelineResult:
    """Run the full pipeline and return the brief with the detail behind it."""
    tool = tool or BigQueryTool.from_settings(get_settings())
    limit = asyncio.Semaphore(MAX_CONCURRENT_MODEL_CALLS)
    with ctx.bind():
        sites = await asyncio.to_thread(query_agent.load_sites, tool, request.site_ids)
        log.info("orchestrator.start", extra={"site_ids": [s.site_id for s in sites]})
        site_runs = await asyncio.gather(
            *(_run_site(site, request.date_range, ctx, tool, limit) for site in sites)
        )
        verified = [v for r in site_runs for v in r.verified_events]
        report = ReportRequest(
            run_id=ctx.run_id,
            date_range=request.date_range,
            sites=sites,
            verified_events=verified,
            scores=[s for r in site_runs for s in r.scores],
            considered={r.site.site_id: len(r.candidates) for r in site_runs},
        )
        items = await report_writer.draft_items(report, ctx)
        events = reviewer.index_events(verified)
        investigations = await _investigate(
            items,
            [v for v in verified if v.verdict is Verdict.UNVERIFIABLE],
            events,
            {s.site_id: s for s in sites},
            request.date_range.end,
            tool,
            limit,
        )
        items, rounds = await reviewer.review_and_revise(items, events, investigations, ctx)
        report = report.model_copy(update={"investigations": investigations})
        brief = report_writer.render(report, items)
        log.info(
            "orchestrator.done",
            extra={
                "items": len(brief.items),
                "investigations": len(investigations),
                "review_rounds": len(rounds),
                "model_calls": dict(ctx.model_calls),
            },
        )
        return PipelineResult(
            brief=brief,
            site_runs=site_runs,
            investigations=investigations,
            review_rounds=rounds,
            model_calls=dict(ctx.model_calls),
        )


async def run(request: BriefRequest, ctx: RunContext) -> RiskBrief:
    """Run the full pipeline for one request and return the brief."""
    return (await run_pipeline(request, ctx)).brief
