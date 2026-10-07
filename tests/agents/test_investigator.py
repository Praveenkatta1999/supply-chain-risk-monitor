import asyncio
from datetime import UTC, date, datetime
from types import SimpleNamespace

from scrm.agents import investigator
from scrm.schemas import (
    Article,
    BriefItem,
    FetchFailure,
    FetchFailureReason,
    InvestigationJudgement,
    RiskScore,
    Scope,
    Verdict,
    VerifiedEvent,
)
from scrm.telemetry import RunContext


class FakeTool:
    allowed_project = "proj"
    allowed_dataset = "scrm"

    def __init__(self, rows):
        self.rows = rows
        self.calls = []

    def run_query(self, sql, params=None):
        self.calls.append((sql, params))
        return self.rows


ROWS = [
    {"url": "https://a.example/rhine-low-water-barges", "source": "a.example",
     "day": date(2026, 10, 1), "place": "Rhine River"},
    {"url": "https://b.example/rhine-traffic-restricted", "source": "b.example",
     "day": date(2026, 9, 28), "place": "Rhine River"},
]  # fmt: skip


def confirmed(event) -> VerifiedEvent:
    return VerifiedEvent(
        event=event, verdict=Verdict.YES, scope=Scope.INDIRECT, reason="Low water.",
        quote="halted", evidence_url=event.source_url,
    )  # fmt: skip


def score(event, severity, impact) -> RiskScore:
    return RiskScore(
        event_id=event.event_id, site_id=event.site_id, severity=severity, impact=impact,
        reason="r", source_url=event.source_url,
    )  # fmt: skip


def toolkit(site, rows=ROWS, fetch=None):
    async def fetch_ok(url):
        return Article(url=url, title="T", text="Barges stuck.", fetched_at=datetime.now(UTC))

    return investigator.Toolkit(
        tool=FakeTool(rows), site=site, subject_id="I1", start=date(2026, 9, 1),
        end=date(2026, 10, 5), fetch=fetch or fetch_ok,
        results={"F1": "https://own.example/story"},
    )  # fmt: skip


def claim_item(event, severity, impact, item_id="I1") -> BriefItem:
    return BriefItem(
        item_id=item_id, site_id=event.site_id, claim="Barges are stuck.", scope=Scope.INDIRECT,
        severity=severity, impact=impact, source_urls=[event.source_url],
    )  # fmt: skip


def test_only_high_risk_claims_are_investigated(event):
    assert investigator.needs_investigation(claim_item(event, 4, 3))  # risk 12
    assert not investigator.needs_investigation(claim_item(event, 3, 3))  # risk 9
    assert not investigator.needs_investigation(claim_item(event, None, None))


def test_keywords_are_cleaned_and_passed_as_parameters(site):
    assert investigator.clean_keywords("Rhine, low-water; Rhine; to x'; DROP") == [
        "rhine", "low", "water", "drop",
    ]  # fmt: skip
    tool = FakeTool([])
    sql = investigator.search_sql(tool, 2)
    assert "@k0" in sql and "@k1" in sql and "rhine" not in sql
    params = investigator.search_params(["rhine", "low"], "P05", date(2026, 9, 1),
                                        date(2026, 10, 5), True)  # fmt: skip
    assert [p.value for p in params if p.name.startswith("k")] == [r"\brhine", r"\blow"]


def test_search_returns_result_ids_not_urls(site):
    kit = toolkit(site)
    out = asyncio.run(kit.search_news("rhine low water"))
    assert [r["id"] for r in out["results"]] == ["R1", "R2"]
    assert "http" not in str(out)
    assert kit.results["R1"] == ROWS[0]["url"]


def test_read_article_only_fetches_known_ids(site):
    fetched = []

    async def fetch(url):
        fetched.append(url)
        return FetchFailure(url=url, reason=FetchFailureReason.PAYWALL, detail="403")

    kit = toolkit(site, fetch=fetch)
    assert asyncio.run(kit.read_article("R7"))["status"] == "error"
    assert fetched == []
    assert asyncio.run(kit.read_article("f1"))["status"] == "error"  # paywalled
    assert fetched == ["https://own.example/story"]


def test_tool_budget_is_enforced_and_every_call_logged(site, caplog):
    kit = toolkit(site)
    tool = SimpleNamespace(name="search_news")
    with RunContext.new().bind():
        results = [kit.before_tool(tool, {"keywords": "x"}, None) for _ in range(10)]
    assert results[:8] == [None] * 8  # allowed
    assert all(r and r["status"] == "error" for r in results[8:])
    logged = [r for r in caplog.records if r.getMessage() == "investigator.tool_call"]
    assert len(logged) == 10 and all(hasattr(r, "tool_args") for r in logged)


def second_source(event) -> VerifiedEvent:
    other = event.model_copy(
        update={"event_id": "e2", "source_url": "https://two.example/same-story"}
    )
    return confirmed(other)


def test_claim_is_investigated_once_with_all_its_sources(event, site):
    seen = []

    async def investigate(payload, kit):
        seen.append(payload)
        await kit.search_news("rhine low water")
        return InvestigationJudgement(
            summary="Still low.", status="ongoing", confidence="medium",
            corroborating_ids=["R2", "F1", "R9"],
        )  # fmt: skip

    sources = [confirmed(event), second_source(event)]
    result = asyncio.run(
        investigator.investigate_claim(
            claim_item(event, 4, 4), sources, site, date(2026, 10, 5), FakeTool(ROWS),
            investigate=investigate,
        )
    )  # fmt: skip
    (payload,) = seen  # one investigation for the whole claim
    assert payload.subject == "claim"
    assert [s.source_id for s in payload.sources] == ["F1", "F2"]
    assert "http" not in payload.model_dump_json()
    assert result.item_id == "I1" and result.trigger == "high_risk"
    assert [str(u) for u in result.source_urls] == [
        str(event.source_url),
        "https://two.example/same-story",
    ]
    # Own sources (F1) and unknown IDs (R9) are not corroboration.
    assert [str(u) for u in result.corroborating_urls] == [ROWS[1]["url"]]


def test_unverifiable_story_is_investigated_alone(event, site):
    unverifiable = VerifiedEvent(
        event=event, verdict=Verdict.UNVERIFIABLE, reason="403", evidence_url=event.source_url
    )

    async def investigate(payload, kit):
        assert payload.subject == "unverifiable_story" and len(payload.sources) == 1
        return InvestigationJudgement(summary="A drill.", status="resolved", confidence="high")

    result = asyncio.run(
        investigator.investigate_story(
            unverifiable, site, date(2026, 10, 5), FakeTool([]), investigate=investigate
        )
    )
    assert result.event_id == event.event_id and result.trigger == "unverifiable"


def test_call_limit_gives_a_low_confidence_result(event, site):
    from google.adk.agents.invocation_context import LlmCallsLimitExceededError

    async def runaway(payload, kit):
        raise LlmCallsLimitExceededError("limit")

    result = asyncio.run(
        investigator.investigate_claim(
            claim_item(event, 5, 5), [confirmed(event)], site, date(2026, 10, 5), FakeTool([]),
            investigate=runaway,
        )
    )  # fmt: skip
    assert result.hit_limit and result.confidence == "low" and result.status == "unclear"


def test_other_failures_return_none(event, site):
    async def broken(payload, kit):
        raise RuntimeError("boom")

    assert asyncio.run(
        investigator.investigate_claim(
            claim_item(event, 5, 5), [confirmed(event)], site, date(2026, 10, 5), FakeTool([]),
            investigate=broken,
        )
    ) is None  # fmt: skip


def test_only_verbatim_evidence_from_what_was_read_is_kept(event, site):
    from scrm.schemas import EvidenceSnippet

    async def investigate(payload, kit):
        await kit.search_news("rhine low water")  # R1, R2 titles registered
        await kit.read_article("F1")  # fetch_ok returns "Barges stuck."
        return InvestigationJudgement(
            summary="s", status="ongoing", confidence="medium",
            evidence=[
                EvidenceSnippet(result_id="F1", quote="barges  STUCK", stance="supports"),
                EvidenceSnippet(result_id="R2", quote="rhine traffic restricted", stance="context"),
                EvidenceSnippet(
                    result_id="R1", quote="Traffic returned to normal.", stance="contradicts"
                ),
                EvidenceSnippet(result_id="R9", quote="anything", stance="supports"),
            ],
        )  # fmt: skip

    async def fetch_ok(url):
        return Article(url=url, title="T", text="Barges stuck.", fetched_at=datetime.now(UTC))

    result = asyncio.run(
        investigator.investigate_claim(
            claim_item(event, 4, 4), [confirmed(event)], site, date(2026, 10, 5),
            FakeTool(ROWS), investigate=investigate, fetch=fetch_ok,
        )
    )  # fmt: skip
    assert [(e.stance, e.quote) for e in result.evidence] == [
        ("supports", "barges  STUCK"),
        ("context", "rhine traffic restricted"),
    ]  # the invented R1 quote and the unknown R9 are dropped
    assert str(result.evidence[1].url) == ROWS[1]["url"]


def test_investigator_temperature_comes_from_settings(site):
    from scrm.config import Settings

    kit = toolkit(site)
    unset = investigator.build_agent(Settings(investigator_temperature=None), kit)
    assert (
        unset.generate_content_config is None or unset.generate_content_config.temperature is None
    )
    assert investigator.build_agent(Settings(), kit).generate_content_config.temperature == 0.2
