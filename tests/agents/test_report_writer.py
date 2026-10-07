import asyncio

from scrm.agents import report_writer
from scrm.config import Settings
from scrm.schemas import (
    BriefItem,
    DraftItem,
    ReportDraft,
    ReportRequest,
    RevisedClaim,
    RevisionDraft,
    RiskScore,
    Verdict,
    VerifiedEvent,
)
from scrm.telemetry import RunContext

A, B, C, D = (f"https://news.example/{x}" for x in "abcd")
SUPPORT = "https://other.example/a-copy"


def verified(event, verdict: Verdict, url: str, supporting=()) -> VerifiedEvent:
    return VerifiedEvent(
        event=event.model_copy(
            update={
                "source_url": url,
                "event_id": url,
                "cluster_size": 1 + len(supporting),
                "supporting_urls": list(supporting),
            }
        ),
        verdict=verdict,
        reason="Because.",
        quote="halted" if verdict is Verdict.YES else None,
        evidence_url=url,
    )


def score(site, url, severity, impact) -> RiskScore:
    return RiskScore(
        event_id=url,
        site_id=site.site_id,
        severity=severity,
        impact=impact,
        reason="r",
        source_url=url,
    )


def make_request(event, site, date_range) -> ReportRequest:
    return ReportRequest(
        run_id="r1",
        date_range=date_range,
        sites=[site],
        verified_events=[
            verified(event, Verdict.YES, A, supporting=[SUPPORT]),
            verified(event, Verdict.YES, B),
            verified(event, Verdict.NO, C),
            verified(event, Verdict.UNVERIFIABLE, D),
        ],
        scores=[score(site, A, 2, 2), score(site, B, 4, 5)],
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


def test_only_confirmed_events_reach_the_model_by_id_without_urls(event, site, date_range):
    draft = drafter([])
    write(make_request(event, site, date_range), draft)
    findings = draft.calls[0].findings
    assert [f.finding_id for f in findings] == ["F1", "F2"]  # A and B; C, D not confirmed
    assert [(f.severity, f.impact) for f in findings] == [(2, 2), (4, 5)]
    assert "http" not in draft.calls[0].model_dump_json()


def test_items_are_ordered_by_risk_and_carry_links(event, site, date_range):
    items = [
        DraftItem(site_id=site.site_id, claim="Minor delay.", finding_ids=["F1"]),
        DraftItem(site_id=site.site_id, claim="Port closed.", finding_ids=["F2"]),
    ]
    brief = write(make_request(event, site, date_range), drafter(items))
    assert [i.claim for i in brief.items] == ["Port closed.", "Minor delay."]
    assert [i.risk for i in brief.items] == [20, 4]
    md = brief.markdown
    assert md.index("Port closed.") < md.index("Minor delay.")
    assert f"Port closed. ([source 1]({B}))" in md
    assert "risk 20/25: severity 4, impact 5" in md
    assert [i.item_id for i in brief.items] == ["I1", "I2"]
    assert f"Also reported, not verified: [link 1]({SUPPORT})" in md


def test_merged_item_takes_its_highest_risk_finding(event, site, date_range):
    merged = DraftItem(site_id=site.site_id, claim="Strike.", finding_ids=["F1", "F2"])
    brief = write(make_request(event, site, date_range), drafter([merged]))
    assert len(brief.items) == 1
    assert (brief.items[0].severity, brief.items[0].impact) == (4, 5)
    assert [str(u) for u in brief.items[0].source_urls] == [A, B]


def test_invalid_item_is_dropped_and_its_findings_fall_back_to_verifier_reasons(
    event, site, date_range
):
    bad = DraftItem(site_id=site.site_id, claim="Made up.", finding_ids=["F2", "F9"])
    brief = write(make_request(event, site, date_range), drafter([bad]))
    assert "Made up." not in brief.markdown
    assert len(brief.items) == 2  # both confirmed findings still reported
    assert {i.claim for i in brief.items} == {"Because."}
    assert {str(u) for i in brief.items for u in i.source_urls} == {A, B}


def test_findings_the_model_skipped_are_still_reported(event, site, date_range):
    only_b = DraftItem(site_id=site.site_id, claim="Port closed.", finding_ids=["F2"])
    brief = write(make_request(event, site, date_range), drafter([only_b]))
    assert [i.claim for i in brief.items] == ["Port closed.", "Because."]
    assert A in brief.markdown


def test_item_citing_another_sites_finding_is_dropped(event, site, date_range):
    wrong_site = DraftItem(site_id="OTHER", claim="Elsewhere.", finding_ids=["F1"])
    brief = write(make_request(event, site, date_range), drafter([wrong_site]))
    assert "Elsewhere." not in brief.markdown
    assert all(i.site_id == site.site_id for i in brief.items)


def test_coverage_table_counts_verdicts_and_articles(event, site, date_range):
    brief = write(make_request(event, site, date_range), drafter([]))
    # Candidates defaults to stories checked when triage counts are not given.
    assert f"| {site.site_id} {site.site_name} | 4 | 4 | 0 | 0 | 1 | 1 |" in brief.markdown


def test_no_confirmed_events_skips_the_model(event, site, date_range):
    request = make_request(event, site, date_range)
    request.verified_events = [v for v in request.verified_events if v.verdict is not Verdict.YES]
    draft = drafter([])
    brief = write(request, draft)
    assert draft.calls == []
    assert "No confirmed disruptions" in brief.markdown


def reviewed_item(item_id, claim, risk_sev, status, **extra):
    return BriefItem(
        item_id=item_id, site_id="SUP-01", claim=claim, severity=risk_sev, impact=4,
        status=status, confidence="high", source_urls=[f"https://news.example/{item_id}"],
        **extra,
    )  # fmt: skip


def test_items_show_status_confidence_and_contradictions(site, date_range):
    request = ReportRequest(run_id="r1", date_range=date_range, sites=[site], verified_events=[])
    items = [
        reviewed_item("I1", "Pipeline shut.", 5, "resolved"),
        reviewed_item("I2", "Port closed.", 3, "ongoing", contradictions=["'closed for a month'"]),
    ]
    md = report_writer.render(request, items).markdown
    # Resolved disruptions come after live ones, even at higher risk.
    assert md.index("Port closed.") < md.index("Pipeline shut.")
    assert "(risk 12/25: severity 3, impact 4) · ongoing · high confidence" in md
    assert "Contradicted by the evidence: 'closed for a month'" in md


def test_revisions_change_only_the_claims_that_were_asked_about():
    items = [reviewed_item("I1", "Old one.", 3, None), reviewed_item("I2", "Old two.", 3, None)]
    revision = RevisionDraft(
        claims=[
            RevisedClaim(item_id="I1", claim="New one."),
            RevisedClaim(item_id="I2", claim="Not asked."),
            RevisedClaim(item_id="I9", claim="Made up."),
        ]
    )
    revised = report_writer.apply_revisions(items, revision, asked={"I1"})
    assert [i.claim for i in revised] == ["New one.", "Old two."]
    assert [i.revisions for i in revised] == [1, 0]
    assert revised[0].source_urls == items[0].source_urls  # sources never change


def test_contradicting_evidence_is_shown_with_its_source(site, date_range):
    from scrm.schemas import Investigation, InvestigationEvidence

    item = reviewed_item("I1", "Suez traffic is falling.", 5, "ongoing")
    inv = Investigation(
        item_id="I1", site_id="SUP-01", source_url=item.source_urls[0], trigger="high_risk",
        summary="Mixed.", status="ongoing", confidence="medium", tool_calls=5,
        evidence=[
            InvestigationEvidence(
                url="https://x.example/normal", quote="Traffic is back to normal.",
                stance="contradicts",
            ),
            InvestigationEvidence(
                url="https://x.example/s", quote="Ships divert.", stance="supports"
            ),
        ],
    )  # fmt: skip
    request = ReportRequest(
        run_id="r1", date_range=date_range, sites=[site], verified_events=[], investigations=[inv]
    )
    md = report_writer.render(request, [item]).markdown
    assert (
        'Contradicting evidence: "Traffic is back to normal." ([source](https://x.example/normal))'
        in md
    )
    assert "Ships divert." not in md
