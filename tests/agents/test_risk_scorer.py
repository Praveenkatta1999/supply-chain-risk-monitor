import asyncio

import pytest

from scrm.agents import risk_scorer
from scrm.config import Settings
from scrm.schemas import RiskJudgement, ScoringRequest, Verdict, VerifiedEvent
from scrm.telemetry import RunContext


def verified(event, verdict=Verdict.YES) -> VerifiedEvent:
    return VerifiedEvent(
        event=event,
        verdict=verdict,
        reason="Workers are on strike at the plant.",
        quote="Workers walked out" if verdict is Verdict.YES else None,
        evidence_url=event.source_url,
    )


def test_risk_scorer_builds_with_structured_output():
    agent = risk_scorer.build_agent(Settings())
    assert agent.name == "risk_scorer"
    assert agent.output_schema is RiskJudgement


def test_scores_come_from_the_model_and_ids_from_the_event(event, site):
    seen = []

    async def score(payload):
        seen.append(payload)
        return RiskJudgement(severity=3, impact=4, reason="A multi-day strike halts output.")

    request = ScoringRequest(verified_event=verified(event), site=site)
    result = asyncio.run(risk_scorer.run(request, RunContext.new(), score=score))
    assert (result.severity, result.impact) == (3, 4)
    assert result.event_id == event.event_id
    assert result.site_id == site.site_id
    assert result.source_url == event.source_url
    assert seen[0].quote == "Workers walked out"


def test_only_confirmed_events_are_scored(event, site):
    async def score(payload):
        raise AssertionError("model must not be called")

    request = ScoringRequest(verified_event=verified(event, Verdict.NO), site=site)
    with pytest.raises(ValueError, match="yes"):
        asyncio.run(risk_scorer.run(request, RunContext.new(), score=score))
