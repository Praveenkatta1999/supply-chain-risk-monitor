import asyncio
from datetime import UTC, date, datetime
from types import SimpleNamespace

from scrm.agents import investigator
from scrm.schemas import (
    Article,
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
        tool=FakeTool(rows), site=site, event_id="e1", start=date(2026, 9, 1),
        end=date(2026, 10, 5), fetch=fetch or fetch_ok,
        results={"F0": "https://own.example/story"},
    )  # fmt: skip


def test_only_high_risk_or_unverifiable_findings_are_investigated(event):
    v = confirmed(event)
    assert investigator.needs_investigation(v, score(event, 4, 3))  # risk 12
    assert not investigator.needs_investigation(v, score(event, 3, 3))  # risk 9
    assert not investigator.needs_investigation(v, None)
    unverifiable = VerifiedEvent(
        event=event, verdict=Verdict.UNVERIFIABLE, reason="403", evidence_url=event.source_url
    )
    assert investigator.needs_investigation(unverifiable, None)


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
    assert asyncio.run(kit.read_article("f0"))["status"] == "error"  # paywalled
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


def test_run_maps_cited_ids_to_urls_and_drops_unknown_ones(event, site):
    async def investigate(payload, kit):
        assert payload.finding_id == "F0" and "http" not in payload.model_dump_json()
        asyncio.get_running_loop()  # tools would be called here by the real agent
        await kit.search_news("rhine low water")
        return InvestigationJudgement(
            summary="Still low.", status="ongoing", confidence="medium",
            corroborating_ids=["R2", "F0", "R9"],
        )  # fmt: skip

    result = asyncio.run(
        investigator.run(
            confirmed(event), score(event, 4, 4), site, date(2026, 10, 5), FakeTool(ROWS),
            investigate=investigate,
        )
    )  # fmt: skip
    assert result.trigger == "high_risk" and result.status == "ongoing"
    assert [str(u) for u in result.corroborating_urls] == [ROWS[1]["url"]]
    assert result.tool_calls == 0  # the fake called the tool directly, not via the callback


def test_call_limit_gives_a_low_confidence_result(event, site):
    from google.adk.agents.invocation_context import LlmCallsLimitExceededError

    async def runaway(payload, kit):
        raise LlmCallsLimitExceededError("limit")

    result = asyncio.run(
        investigator.run(
            confirmed(event), score(event, 5, 5), site, date(2026, 10, 5), FakeTool([]),
            investigate=runaway,
        )
    )  # fmt: skip
    assert result.hit_limit and result.confidence == "low" and result.status == "unclear"


def test_other_failures_return_none(event, site):
    async def broken(payload, kit):
        raise RuntimeError("boom")

    assert asyncio.run(
        investigator.run(
            confirmed(event), score(event, 5, 5), site, date(2026, 10, 5), FakeTool([]),
            investigate=broken,
        )
    ) is None  # fmt: skip
