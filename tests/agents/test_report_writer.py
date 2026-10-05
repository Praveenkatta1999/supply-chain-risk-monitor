import asyncio

from scrm.agents import report_writer
from scrm.config import Settings
from scrm.schemas import DraftItem, ReportDraft, ReportRequest, Verdict, VerifiedEvent
from scrm.telemetry import RunContext


def verified(event, verdict: Verdict, url: str) -> VerifiedEvent:
    return VerifiedEvent(
        event=event.model_copy(update={"source_url": url}),
        verdict=verdict,
        reason="Because.",
        quote="halted" if verdict is Verdict.YES else None,
        evidence_url=url,
    )


def make_request(event, site, date_range) -> ReportRequest:
    return ReportRequest(
        run_id="r1",
        date_range=date_range,
        sites=[site],
        verified_events=[
            verified(event, Verdict.YES, "https://news.example/a"),
            verified(event, Verdict.YES, "https://news.example/b"),
            verified(event, Verdict.NO, "https://news.example/c"),
            verified(event, Verdict.UNVERIFIABLE, "https://news.example/d"),
        ],
    )


def drafter(items: list[DraftItem]):
    calls = []

    async def draft(payload):
        calls.append(payload)
        return ReportDraft(items=items)

    draft.calls = calls
    return draft


def write(request, draft):
    return asyncio.run(report_writer.run(request, RunContext.new(), draft=draft))


def test_report_writer_builds_with_structured_output():
    agent = report_writer.build_agent(Settings())
    assert agent.name == "report_writer"
    assert agent.output_schema is ReportDraft


def test_only_confirmed_events_reach_the_model(event, site, date_range):
    draft = drafter([])
    write(make_request(event, site, date_range), draft)
    urls = [str(f.source_url) for f in draft.calls[0].findings]
    assert urls == ["https://news.example/a", "https://news.example/b"]


def test_every_claim_is_rendered_with_its_links(event, site, date_range):
    item = DraftItem(
        site_id=site.site_id,
        claim="A strike has halted the port.",
        source_urls=["https://news.example/a", "https://news.example/b"],
    )
    brief = write(make_request(event, site, date_range), drafter([item]))
    assert len(brief.items) == 1
    assert (
        "- A strike has halted the port. ([source 1](https://news.example/a), "
        "[source 2](https://news.example/b))" in brief.markdown
    )
    assert "4 candidate articles reviewed: 2 confirmed, 1 rejected" in brief.markdown


def test_claims_citing_unconfirmed_urls_are_dropped(event, site, date_range):
    bad = DraftItem(site_id=site.site_id, claim="Made up.", source_urls=["https://news.example/c"])
    brief = write(make_request(event, site, date_range), drafter([bad]))
    assert brief.items == []
    assert "Made up." not in brief.markdown


def test_no_confirmed_events_skips_the_model(event, site, date_range):
    request = make_request(event, site, date_range)
    request.verified_events = [v for v in request.verified_events if v.verdict is not Verdict.YES]
    draft = drafter([])
    brief = write(request, draft)
    assert draft.calls == []
    assert "No confirmed disruptions" in brief.markdown
