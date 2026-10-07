"""The orchestrator is tested with every agent replaced by a fake: no BigQuery, no Gemini."""

import asyncio
import functools

from scrm.agents import orchestrator, query_agent, report_writer, risk_scorer, verifier
from scrm.schemas import (
    BriefRequest,
    DraftItem,
    QueryResult,
    ReportDraft,
    RiskBrief,
    RiskScore,
    Verdict,
    VerifiedEvent,
)
from scrm.telemetry import RunContext, current_run_id


def test_orchestrator_contract():
    assert orchestrator.OUTPUT_SCHEMA is RiskBrief


def test_runs_sites_and_orders_the_brief_by_risk(monkeypatch, site, event, date_range):
    sites = [site.model_copy(update={"site_id": sid}) for sid in ("A", "B")]
    impact_by_site = {"A": 2, "B": 5}
    seen_run_ids = set()

    def load_sites(tool, site_ids):
        assert site_ids == ["A", "B"]
        return sites

    async def query_run(request, ctx, *, tool, sites):
        (s,) = sites
        e = event.model_copy(
            update={"site_id": s.site_id, "source_url": f"https://news.example/{s.site_id}"}
        )
        return QueryResult(events=[e], sql=[], bytes_processed=10)

    async def run_many(requests, ctx, *, limit):
        seen_run_ids.add(current_run_id())
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
        ctx.record_model_call("risk_scorer")
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
        return ReportDraft(
            items=[
                DraftItem(
                    site_id=f.site_id, claim=f"Strike at {f.site_id}.", finding_ids=[f.finding_id]
                )
                for f in payload.findings
            ]
        )

    monkeypatch.setattr(query_agent, "load_sites", load_sites)
    monkeypatch.setattr(query_agent, "run", query_run)
    monkeypatch.setattr(verifier, "run_many", run_many)
    monkeypatch.setattr(risk_scorer, "run", score_run)
    monkeypatch.setattr(report_writer, "run", functools.partial(report_writer.run, draft=draft))

    ctx = RunContext.new()
    request = BriefRequest(site_ids=["A", "B"], date_range=date_range)
    result = asyncio.run(orchestrator.run_pipeline(request, ctx, tool=object()))

    assert [i.site_id for i in result.brief.items] == ["B", "A"]  # risk 20 before risk 8
    assert [r.site.site_id for r in result.site_runs] == ["A", "B"]
    assert result.model_calls == {"risk_scorer": 2}
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
