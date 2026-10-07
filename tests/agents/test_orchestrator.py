"""The orchestrator is tested with every agent replaced by a fake: no BigQuery, no Gemini."""

import asyncio
import functools

from scrm.agents import (
    investigator,
    orchestrator,
    query_agent,
    report_writer,
    reviewer,
    risk_scorer,
    triage,
    verifier,
)
from scrm.schemas import (
    BriefRequest,
    ClaimReview,
    DraftItem,
    Investigation,
    QueryResult,
    ReportDraft,
    ReviewDraft,
    RiskBrief,
    RiskScore,
    TriageResult,
    TriageSelection,
    Verdict,
    VerifiedEvent,
)
from scrm.telemetry import RunContext, current_run_id


def test_orchestrator_contract():
    assert orchestrator.OUTPUT_SCHEMA is RiskBrief


def test_pipeline_routes_work_and_investigates_once_per_story(monkeypatch, site, event, date_range):
    sites = [site.model_copy(update={"site_id": sid}) for sid in ("A", "B")]
    impact_by_site = {"A": 2, "B": 5}
    seen_run_ids, verified_ids, investigated = set(), [], []

    def load_sites(tool, site_ids):
        assert site_ids == ["A", "B"]
        return sites

    async def query_run(request, ctx, *, tool, sites):
        (s,) = sites
        events = [
            event.model_copy(
                update={
                    "event_id": f"{s.site_id}-{n}",
                    "site_id": s.site_id,
                    "source_url": f"https://news.example/{s.site_id}/{n}",
                }
            )
            for n in (1, 2, 3)
        ]
        return QueryResult(events=events, sql=[], bytes_processed=10)

    async def triage_run(site, events, ctx):
        keep = events[:2]  # triage drops the third candidate
        return TriageResult(
            site_id=site.site_id,
            considered=len(events),
            selected=[TriageSelection(event_id=e.event_id, reason="Strike.") for e in keep],
        )

    async def run_many(requests, ctx, *, limit):
        seen_run_ids.add(current_run_id())
        verified_ids.extend(r.event.event_id for r in requests)
        return [
            VerifiedEvent(
                event=r.event,
                verdict=Verdict.YES,
                reason="Strike.",
                quote="halted",
                evidence_url=r.event.source_url,
            )
            for r in requests
        ]

    async def score_run(request, ctx):
        v = request.verified_event
        return RiskScore(
            event_id=v.event.event_id,
            site_id=request.site.site_id,
            severity=4,
            impact=impact_by_site[request.site.site_id],
            reason="r",
            source_url=v.evidence_url,
        )

    async def draft(payload):
        # Both findings per site describe the same strike: one claim per site.
        by_site: dict[str, list[str]] = {}
        for f in payload.findings:
            by_site.setdefault(f.site_id, []).append(f.finding_id)
        return ReportDraft(
            items=[
                DraftItem(site_id=s, claim=f"Strike at {s}.", finding_ids=ids)
                for s, ids in by_site.items()
            ]
        )

    async def investigate_claim(item, events, site, search_until, tool):
        investigated.append((item.item_id, item.site_id, len(events)))
        return Investigation(
            item_id=item.item_id,
            site_id=site.site_id,
            source_url=item.source_urls[0],
            source_urls=item.source_urls,
            trigger="high_risk",
            summary="Still closed.",
            status="ongoing",
            confidence="high",
            corroborating_urls=["https://other.example/report"],
            tool_calls=3,
        )

    async def review(payload):
        return ReviewDraft(
            reviews=[
                ClaimReview(item_id=i.item_id, status="ongoing", confidence="high")
                for i in payload.items
            ]
        )

    monkeypatch.setattr(query_agent, "load_sites", load_sites)
    monkeypatch.setattr(query_agent, "run", query_run)
    monkeypatch.setattr(triage, "run", triage_run)
    monkeypatch.setattr(verifier, "run_many", run_many)
    monkeypatch.setattr(risk_scorer, "run", score_run)
    monkeypatch.setattr(investigator, "investigate_claim", investigate_claim)
    monkeypatch.setattr(
        report_writer, "draft_items", functools.partial(report_writer.draft_items, draft=draft)
    )
    monkeypatch.setattr(
        reviewer, "review_and_revise", functools.partial(reviewer.review_and_revise, review=review)
    )

    ctx = RunContext.new()
    request = BriefRequest(site_ids=["A", "B"], date_range=date_range)
    result = asyncio.run(orchestrator.run_pipeline(request, ctx, tool=object()))

    assert verified_ids == ["A-1", "A-2", "B-1", "B-2"]  # only triaged candidates
    # One investigation for B's two-source claim (risk 20); A's claim is risk 8.
    assert investigated == [("I1", "B", 2)]
    assert [i.site_id for i in result.brief.items] == ["B", "A"]
    assert {i.status for i in result.brief.items} == {"ongoing"}
    assert len(result.review_rounds) == 1
    assert "Investigator (ongoing, high confidence, 3 tool calls)" in result.brief.markdown
    assert seen_run_ids == {ctx.run_id}


def test_a_failed_score_leaves_the_event_unscored(monkeypatch, site, event):
    async def broken(request, ctx):
        raise RuntimeError("quota")

    monkeypatch.setattr(risk_scorer, "run", broken)
    confirmed = VerifiedEvent(
        event=event, verdict=Verdict.YES, reason="r", quote="q", evidence_url=event.source_url
    )
    scores = asyncio.run(
        orchestrator._score_confirmed([confirmed], site, RunContext.new(), asyncio.Semaphore(1))
    )
    assert scores == []
