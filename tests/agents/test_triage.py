import asyncio

from scrm.agents import triage
from scrm.config import Settings
from scrm.schemas import TriageDraft, TriagePick
from scrm.telemetry import RunContext


def candidates(event, n):
    return [
        event.model_copy(
            update={
                "event_id": f"e{i}",
                "source_url": f"https://news.example/{i}",
                "themes": ["TAX_FNCACT_POLICE", "STRIKE", "MARITIME"],
            }
        )
        for i in range(1, n + 1)
    ]


def drafter(picks):
    seen = []

    async def draft(payload):
        seen.append(payload)
        return TriageDraft(picks=picks)

    draft.seen = seen
    return draft


def test_triage_builds_with_structured_output():
    agent = triage.build_agent(Settings())
    assert agent.name == "triage"
    assert agent.output_schema is TriageDraft


def test_model_sees_metadata_by_id_never_urls(event, site):
    draft = drafter([])
    asyncio.run(triage.run(site, candidates(event, 3), RunContext.new(), draft=draft))
    payload = draft.seen[0]
    assert [c.candidate_id for c in payload.candidates] == ["C1", "C2", "C3"]
    assert payload.candidates[0].themes == ["STRIKE", "MARITIME"]  # TAX_ noise dropped
    assert "http" not in payload.model_dump_json()


def test_picks_map_back_in_rank_order_and_unknown_ids_are_ignored(event, site):
    picks = [
        TriagePick(candidate_id="C3", reason="Port strike."),
        TriagePick(candidate_id="C1", reason="Terminal fire."),
        TriagePick(candidate_id="C9", reason="Made up."),
    ]
    result = asyncio.run(
        triage.run(site, candidates(event, 3), RunContext.new(), draft=drafter(picks))
    )
    assert [(s.event_id, s.reason) for s in result.selected] == [
        ("e1", "Terminal fire."),
        ("e3", "Port strike."),
    ]
    assert result.considered == 3 and not result.fallback


def test_at_most_fifty_candidates_are_considered(event, site):
    draft = drafter([])
    result = asyncio.run(triage.run(site, candidates(event, 60), RunContext.new(), draft=draft))
    assert result.considered == 50 and len(draft.seen[0].candidates) == 50


def test_failed_call_falls_back_to_top_ranked(event, site):
    async def broken(payload):
        raise RuntimeError("quota")

    result = asyncio.run(triage.run(site, candidates(event, 30), RunContext.new(), draft=broken))
    assert result.fallback
    assert [s.event_id for s in result.selected] == [f"e{i}" for i in range(1, 21)]
