import asyncio

from scrm.agents import reviewer
from scrm.config import Settings
from scrm.schemas import (
    BriefItem,
    ClaimReview,
    Investigation,
    ReviewDraft,
    RevisedClaim,
    RevisionDraft,
    Verdict,
    VerifiedEvent,
)
from scrm.telemetry import RunContext


def item(item_id: str, claim: str, url: str) -> BriefItem:
    return BriefItem(
        item_id=item_id, site_id="SUP-01", claim=claim, severity=4, impact=4, source_urls=[url]
    )


def setup(event):
    url = str(event.source_url)
    verified = VerifiedEvent(
        event=event, verdict=Verdict.YES, reason="Pipeline shut.", quote="shut", evidence_url=url
    )
    investigation = Investigation(
        item_id="I1", site_id="SUP-01", source_url=url, trigger="high_risk",
        summary="It restarted on 25 September.", status="resolved", confidence="high",
        tool_calls=4,
    )  # fmt: skip
    items = [item("I1", "The pipeline is shut and the port is closed.", url)]
    return items, reviewer.index_events([verified]), [investigation]


def scripted_reviews(*rounds: list[ClaimReview]):
    """A fake reviewer that returns one scripted round per call and records its input."""
    seen = []

    async def review(payload):
        seen.append(payload)
        return ReviewDraft(reviews=rounds[len(seen) - 1])

    review.seen = seen
    return review


def reviser(text: str):
    calls = []

    async def revise(payload):
        calls.append(payload)
        return RevisionDraft(
            claims=[RevisedClaim(item_id=i.item_id, claim=text) for i in payload.items]
        )

    revise.calls = calls
    return revise


def loop(items, events, investigations, review, revise):
    return asyncio.run(
        reviewer.review_and_revise(
            items, events, investigations, RunContext.new(), review=review, revise=revise
        )
    )


def test_reviewer_builds_with_structured_output():
    agent = reviewer.build_agent(Settings())
    assert agent.name == "reviewer" and agent.output_schema is ReviewDraft


def test_reviewer_sees_claims_evidence_and_investigation_without_urls(event):
    items, events, investigations = setup(event)
    review = scripted_reviews([ClaimReview(item_id="I1", status="resolved", confidence="high")])
    loop(items, events, investigations, review, reviser("unused"))
    (row,) = review.seen[0].items
    assert row.verifier_evidence[0].quote == "shut"
    assert row.investigation_status == "resolved"
    assert "http" not in review.seen[0].model_dump_json()


def test_accurate_claims_need_one_round(event):
    items, events, investigations = setup(event)
    revise = reviser("unused")
    review = scripted_reviews([ClaimReview(item_id="I1", status="ongoing", confidence="medium")])
    final, rounds = loop(items, events, investigations, review, revise)
    assert len(rounds) == 1 and revise.calls == []
    assert (final[0].status, final[0].confidence) == ("ongoing", "medium")


def test_corrections_go_back_to_the_writer_then_are_reviewed_again(event):
    items, events, investigations = setup(event)
    fix = "Remove 'the pipeline is shut'; it restarted on 25 September."
    review = scripted_reviews(
        [ClaimReview(item_id="I1", status="resolved", confidence="high",
                     contradicted=["'the pipeline is shut'"], corrections=[fix])],
        [ClaimReview(item_id="I1", status="ongoing", confidence="high")],
    )  # fmt: skip
    revise = reviser("The port is closed.")
    final, rounds = loop(items, events, investigations, review, revise)
    assert revise.calls[0].items[0].corrections == [fix]
    assert [r.revised_item_ids for r in rounds] == [["I1"], []]
    assert final[0].claim == "The port is closed." and final[0].revisions == 1
    assert final[0].contradictions == []  # cleared by the final review
    assert review.seen[1].items[0].claim == "The port is closed."


def test_at_most_two_revisions_then_flags_remain(event):
    items, events, investigations = setup(event)
    stubborn = ClaimReview(
        item_id="I1", status="unclear", confidence="low",
        contradicted=["'closed'"], corrections=["Remove 'closed'."],
    )  # fmt: skip
    review = scripted_reviews([stubborn], [stubborn], [stubborn])
    revise = reviser("Still says closed.")
    final, rounds = loop(items, events, investigations, review, revise)
    assert len(revise.calls) == reviewer.MAX_REVISIONS == 2
    assert len(rounds) == 3 and rounds[-1].revised_item_ids == []
    assert final[0].contradictions == ["'closed'"] and final[0].revisions == 2


def test_unknown_or_duplicate_reviews_are_ignored(event):
    items, events, investigations = setup(event)
    review = scripted_reviews(
        [
            ClaimReview(item_id="I1", status="ongoing", confidence="high"),
            ClaimReview(item_id="I1", status="resolved", confidence="low"),
            ClaimReview(item_id="I9", status="resolved", confidence="low", corrections=["x"]),
        ]
    )
    final, _ = loop(items, events, investigations, review, reviser("unused"))
    assert final[0].status == "ongoing"


def test_failed_review_publishes_claims_unreviewed(event):
    items, events, investigations = setup(event)

    async def broken(payload):
        raise RuntimeError("quota")

    final, rounds = loop(items, events, investigations, broken, reviser("unused"))
    assert final == items and rounds == []


def test_failed_revision_keeps_the_reviewed_claim_with_its_flags(event):
    items, events, investigations = setup(event)
    review = scripted_reviews(
        [ClaimReview(item_id="I1", status="resolved", confidence="high",
                     contradicted=["'shut'"], corrections=["Remove 'shut'."])]
    )  # fmt: skip

    async def broken(payload):
        raise RuntimeError("quota")

    final, _ = loop(items, events, investigations, review, broken)
    assert final[0].claim == items[0].claim
    assert final[0].status == "resolved" and final[0].contradictions == ["'shut'"]
