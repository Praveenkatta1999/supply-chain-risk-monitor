import asyncio
from datetime import UTC, date, datetime
from types import SimpleNamespace

from scrm.agents import root_agent
from scrm.schemas import BriefItem, PipelineResult, RiskBrief, RootAnswer, Site, SiteType
from scrm.telemetry import RunContext

TODAY = date(2026, 10, 7)


def make_site(site_id: str, component: str | None) -> Site:
    return Site(
        site_id=site_id, site_name=f"Site {site_id}", company="Co", site_type=SiteType.FACTORY,
        component=component, country="X", lat=0, lon=0, radius_km=50,
    )  # fmt: skip


SITES = {s.site_id: s for s in [make_site("S01", "processors"), make_site("S07", "memory")]}


def fake_result(request) -> PipelineResult:
    item = BriefItem(
        item_id="I1", site_id=request.site_ids[0], claim="A fab outage.", severity=4,
        impact=4, status="ongoing", confidence="high", source_urls=["https://news.example/a"],
    )  # fmt: skip
    brief = RiskBrief(
        run_id="r1", generated_at=datetime.now(UTC), date_range=request.date_range,
        site_ids=request.site_ids, items=[item], markdown="[s](https://news.example/a)",
    )  # fmt: skip
    return PipelineResult(brief=brief, site_runs=[], model_calls={})


def toolkit(runs: list):
    async def run_pipeline(request, ctx):
        runs.append(request)
        return fake_result(request)

    kit = root_agent.RootToolkit(
        tool=None, ctx=RunContext.new(), today=TODAY, run_pipeline=run_pipeline
    )
    kit.sites = SITES  # skip BigQuery
    return kit


def run(kit, *args):
    return asyncio.run(kit.run_risk_brief(*args))


def test_list_sites_shows_components():
    sites = asyncio.run(toolkit([]).list_sites())["sites"]
    assert {(s["site_id"], s["component"]) for s in sites} == {
        ("S01", "processors"),
        ("S07", "memory"),
    }


def test_run_risk_brief_validates_arguments_before_running():
    runs: list = []
    kit = toolkit(runs)
    assert "Unknown site_ids" in run(kit, ["S99"], "2026-10-01", "2026-10-07")["message"]
    assert "YYYY-MM-DD" in run(kit, ["S01"], "last week", "2026-10-07")["message"]
    assert "after today" in run(kit, ["S01"], "2026-10-01", "2026-10-09")["message"]
    assert "at most 31 days" in run(kit, ["S01"], "2026-08-01", "2026-10-07")["message"]
    assert "on or before" in run(kit, ["S01"], "2026-10-07", "2026-10-01")["message"]
    assert runs == []


def test_run_risk_brief_runs_once_and_returns_claims_without_urls():
    runs: list = []
    kit = toolkit(runs)
    out = run(kit, ["S01", "S07", "S01"], "2026-10-01", "2026-10-07")
    assert out["status"] == "ok"
    assert runs[0].site_ids == ["S01", "S07"]  # duplicates dropped
    assert out["claims"][0]["status"] == "ongoing"
    assert out["sites_with_nothing_confirmed"] == ["S07"]
    assert "http" not in str(out)
    again = run(kit, ["S01"], "2026-10-01", "2026-10-07")
    assert again["status"] == "error" and len(runs) == 1


def test_tool_budget_is_enforced():
    kit = toolkit([])
    tool = SimpleNamespace(name="list_sites")
    results = [kit.before_tool(tool, {}, None) for _ in range(root_agent.MAX_TOOL_CALLS + 1)]
    assert results[:-1] == [None] * root_agent.MAX_TOOL_CALLS
    assert results[-1]["status"] == "error"


def test_answer_reports_what_actually_ran():
    async def ask(payload, kit):
        assert payload.today == TODAY
        kit.sites = SITES  # skip BigQuery
        await kit.run_risk_brief(["S01"], "2026-10-01", "2026-10-07")
        # The model misreports what it ran; Python corrects it.
        return RootAnswer(
            summary="One ongoing fab outage.", rationale="Chip fabs, last 7 days.",
            site_ids=["S01", "S07"], start_date=date(2026, 9, 1), end_date=TODAY,
        )  # fmt: skip

    async def run_pipeline(request, ctx):
        return fake_result(request)

    result = asyncio.run(
        root_agent.run(
            "Chip risks this week?", RunContext.new(), today=TODAY, tool=object(), ask=ask,
            run_pipeline=run_pipeline,
        )
    )  # fmt: skip
    assert result.answer.site_ids == ["S01"]
    assert result.answer.start_date == date(2026, 10, 1)
    assert result.pipeline.brief.items[0].claim == "A fab outage."
