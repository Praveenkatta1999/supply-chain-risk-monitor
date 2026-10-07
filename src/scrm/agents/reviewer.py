"""Reviewer: a critic loop that checks every claim against its evidence before publication.

For each claim the reviewer sees the verifier's evidence (reasons and verbatim quotes)
and, for investigated claims, the investigator's summary, status and confidence. It
returns, per claim: status (ongoing / resolved / unclear), confidence, statements the
evidence contradicts, and specific corrections.

Loop (``review_and_revise``):
    review all claims (one model call)
    -> if any claim has corrections, the report writer rewrites just those claims
    -> review again; at most ``MAX_REVISIONS`` rewrite rounds, then a final review
The final review's status, confidence and contradictions are attached to each claim.
The reviewer cites claims by ID (I1, I2, ...); it cannot add claims, change sources or
change scores. If a review call fails, the claims are published unreviewed.

Input:  ReviewInput (built from brief items, verified events and investigations)
Output: list of ClaimReview, applied to the brief items
"""

from collections.abc import Awaitable, Callable

from google.adk.agents import LlmAgent

from scrm.agents import report_writer
from scrm.agents._adk import run_structured
from scrm.config import Settings, get_settings
from scrm.schemas import (
    BriefItem,
    ClaimReview,
    Investigation,
    ReviewDraft,
    ReviewEvidence,
    ReviewInput,
    ReviewItem,
    ReviewRound,
    RevisionItem,
    VerifiedEvent,
)
from scrm.telemetry import RunContext, get_logger

NAME = "reviewer"
INPUT_SCHEMA = ReviewInput
OUTPUT_SCHEMA = ReviewDraft
DESCRIPTION = (__doc__ or "").splitlines()[0]

MAX_REVISIONS = 2

log = get_logger(__name__)

INSTRUCTION = """\
You review the claims of a supply chain risk brief against their evidence before it is
published. The input is JSON with the claims. Each has an item_id, the claim text, its
scope ("direct": the site itself; "indirect": a connected route or region), risk scores,
the verifier's evidence (one-sentence reasons and verbatim quotes from the source
articles), and, for high-risk claims, the investigator's summary, status and confidence
from searching for corroborating and newer reports.

Return one review per claim:
- status: "ongoing", "resolved" or "unclear". Use the investigator's finding when there
  is one; otherwise judge from the evidence; "unclear" if it does not say.
- confidence: "high" (independent sources agree, or the investigator corroborated it
  with high confidence), "medium" (one solid source or partial corroboration), "low"
  (weak, single unclear source, or conflicting evidence).
- contradicted: statements in the claim that the evidence contradicts, each quoted
  briefly from the claim with what the evidence says instead. Empty if none.
- corrections: specific edits for the writer, e.g. "Remove the statement that the
  pipeline is shut; the investigator found it restarted on 25 September." Ask to remove
  parts that are resolved, contradicted or unsupported (including separate events merged
  into one claim). Empty if the claim is accurate. No style or wording preferences.
Never invent facts; rely only on the evidence given.
"""


def build_agent(settings: Settings) -> LlmAgent:
    """Return the ADK agent definition. Constructing it makes no network calls."""
    return LlmAgent(
        name=NAME,
        model=settings.gemini_model,
        description=DESCRIPTION,
        instruction=INSTRUCTION,
        input_schema=ReviewInput,
        output_schema=ReviewDraft,
        output_key=NAME,
    )


Review = Callable[[ReviewInput], Awaitable[ReviewDraft]]


async def _gemini_review(payload: ReviewInput) -> ReviewDraft:
    return await run_structured(build_agent(get_settings()), payload, ReviewDraft)


EventIndex = dict[tuple[str, str], VerifiedEvent]  # (site_id, evidence URL) -> event


def index_events(events: list[VerifiedEvent]) -> EventIndex:
    """Key verified events by site and URL: one article can be a story for two sites."""
    return {(v.event.site_id, str(v.evidence_url)): v for v in events}


def sources_of(item: BriefItem, events: EventIndex) -> list[VerifiedEvent]:
    keys = [(item.site_id, str(u)) for u in item.source_urls]
    return [events[k] for k in keys if k in events]


def evidence_for(item: BriefItem, events: EventIndex) -> list[ReviewEvidence]:
    """The verifier's reasons and quotes for the item's sources."""
    cited = sources_of(item, events)
    return [ReviewEvidence(reason=v.reason, quote=v.quote) for v in cited]


def investigation_for(item: BriefItem, investigations: list[Investigation]) -> Investigation | None:
    return next((i for i in investigations if i.item_id and i.item_id == item.item_id), None)


def review_input(
    items: list[BriefItem],
    events: EventIndex,
    investigations: list[Investigation],
) -> ReviewInput:
    rows = []
    for item in items:
        inv = investigation_for(item, investigations)
        rows.append(
            ReviewItem(
                item_id=item.item_id or "",
                site_id=item.site_id,
                scope=item.scope,
                claim=item.claim,
                severity=item.severity,
                impact=item.impact,
                verifier_evidence=evidence_for(item, events),
                investigation_summary=inv.summary if inv else None,
                investigation_status=inv.status if inv else None,
                investigation_confidence=inv.confidence if inv else None,
            )
        )
    return ReviewInput(items=rows)


def known_reviews(draft: ReviewDraft, items: list[BriefItem]) -> list[ClaimReview]:
    """Reviews for claims that exist, one per claim (the first if the model repeats one)."""
    ids = {i.item_id for i in items}
    reviews: dict[str, ClaimReview] = {}
    for review in draft.reviews:
        if review.item_id in ids:
            reviews.setdefault(review.item_id, review)
    if unknown := sorted({r.item_id for r in draft.reviews} - ids):
        log.warning("reviewer.unknown_ids", extra={"item_ids": unknown})
    if missing := sorted(ids - set(reviews)):
        log.warning("reviewer.missing_reviews", extra={"item_ids": missing})
    return list(reviews.values())


def apply_reviews(items: list[BriefItem], reviews: list[ClaimReview]) -> list[BriefItem]:
    """Attach the final status, confidence and contradictions to each claim."""
    by_id = {r.item_id: r for r in reviews}
    return [
        item.model_copy(
            update={
                "status": by_id[item.item_id].status,
                "confidence": by_id[item.item_id].confidence,
                "contradictions": by_id[item.item_id].contradicted,
            }
        )
        if item.item_id in by_id
        else item
        for item in items
    ]


def revision_requests(
    items: list[BriefItem],
    reviews: list[ClaimReview],
    events: EventIndex,
    investigations: list[Investigation],
) -> list[RevisionItem]:
    """What to send back to the report writer: only claims with corrections."""
    corrections = {r.item_id: r.corrections for r in reviews if r.corrections}
    requests = []
    for item in items:
        if item.item_id not in corrections:
            continue
        inv = investigation_for(item, investigations)
        requests.append(
            RevisionItem(
                item_id=item.item_id,
                claim=item.claim,
                corrections=corrections[item.item_id],
                verifier_evidence=evidence_for(item, events),
                investigation_summary=inv.summary if inv else None,
            )
        )
    return requests


async def review_and_revise(
    items: list[BriefItem],
    events: EventIndex,
    investigations: list[Investigation],
    ctx: RunContext,
    *,
    review: Review = _gemini_review,
    revise: report_writer.Reviser = report_writer._gemini_revise,
) -> tuple[list[BriefItem], list[ReviewRound]]:
    """Run the critic loop; return the final claims and a record of every round."""
    rounds: list[ReviewRound] = []
    if not items:
        return items, rounds
    with ctx.bind():
        for number in range(1, MAX_REVISIONS + 2):
            try:
                reviews = known_reviews(
                    await review(review_input(items, events, investigations)), items
                )
            except Exception:  # publish unreviewed rather than lose the brief
                log.exception("reviewer.failed", extra={"round": number})
                return items, rounds
            requests = revision_requests(items, reviews, events, investigations)
            last = not requests or number > MAX_REVISIONS
            rounds.append(
                ReviewRound(
                    round=number,
                    reviews=reviews,
                    revised_item_ids=[] if last else [r.item_id for r in requests],
                )
            )
            log.info(
                "reviewer.round",
                extra={"round": number, "reviews": len(reviews), "corrections": len(requests)},
            )
            if last:
                return apply_reviews(items, reviews), rounds
            try:
                items = await report_writer.revise_items(items, requests, ctx, revise=revise)
            except Exception:  # keep the reviewed claims, with their flags, unrevised
                log.exception("reviewer.revision_failed", extra={"round": number})
                return apply_reviews(items, reviews), rounds
    return items, rounds  # unreachable: the last iteration always returns
