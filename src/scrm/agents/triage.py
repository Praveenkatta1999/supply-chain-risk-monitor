"""Triage: one model call per site that picks which candidate stories are worth verifying.

The verifier fetches and reads every article it is given, which is the expensive step.
Triage reads only metadata (headline words from the URL, source, date, place, themes,
organisations, entity matches) for up to ``MAX_CANDIDATES`` stories and returns the ones
plausibly about a disruption to this site, each with a one-line reason.

Candidates are shown to the model as C1, C2, ... and never with URLs; Python maps the
picks back to events. If the call fails, the top-ranked ``FALLBACK_COUNT`` stories go to
the verifier instead, so a triage outage degrades cost, not coverage.

Input:  TriageInput (built from the site and its ranked candidates)
Output: TriageResult
"""

from collections.abc import Awaitable, Callable

from google.adk.agents import LlmAgent

from scrm.agents._adk import run_structured
from scrm.config import Settings, get_settings
from scrm.schemas import (
    Event,
    Site,
    TriageCandidate,
    TriageDraft,
    TriageInput,
    TriageResult,
    TriageSelection,
)
from scrm.telemetry import RunContext, get_logger

NAME = "triage"
INPUT_SCHEMA = TriageInput
OUTPUT_SCHEMA = TriageResult
DESCRIPTION = (__doc__ or "").splitlines()[0]

MAX_CANDIDATES = 50
FALLBACK_COUNT = 20
MAX_THEMES = 15
MAX_ORGANIZATIONS = 8

log = get_logger(__name__)

INSTRUCTION = """\
You triage news for a supply chain risk team. The input is JSON with one monitored site
(a port, factory or shipping chokepoint) and up to 50 candidate stories found near it.
For each story you only see metadata: headline words taken from the URL (title, may be
missing for non-Latin URLs), source domain, date, geocoded place, GDELT themes,
organisations named, any organisations that matched the site's company
(matched_entities), and how many articles covered it (cluster_size).

Pick the stories that could plausibly describe a real, current disruption to this site's
operations or cargo flows, directly or through a connected route or region: strikes,
closures, accidents, fires, severe weather, low water, cyberattacks, congestion,
sanctions, conflict or security incidents, outages or production problems at the
company. Leave out stories that are clearly something else: markets and stock prices,
politics without an operational effect, culture, sport, travel, crime unrelated to the
site, history, opinion.

Be inclusive when unsure, because a later step reads the full article; but do not pick a
story only because it mentions the site or its company.

Return picks: for each chosen story, its candidate_id and a one-line reason in English.
Return an empty list if none qualify.
"""


def build_agent(settings: Settings) -> LlmAgent:
    """Return the ADK agent definition. Constructing it makes no network calls."""
    return LlmAgent(
        name=NAME,
        model=settings.gemini_model,
        description=DESCRIPTION,
        instruction=INSTRUCTION,
        input_schema=TriageInput,
        output_schema=TriageDraft,
        output_key=NAME,
    )


Drafter = Callable[[TriageInput], Awaitable[TriageDraft]]


async def _gemini_triage(payload: TriageInput) -> TriageDraft:
    return await run_structured(build_agent(get_settings()), payload, TriageDraft)


def _readable_themes(themes: list[str]) -> list[str]:
    """Drop GDELT's taxonomy noise (TAX_ actor and language tags) and cap the list."""
    return [t for t in themes if not t.startswith("TAX_")][:MAX_THEMES]


def to_candidate(event: Event, candidate_id: str) -> TriageCandidate:
    return TriageCandidate(
        candidate_id=candidate_id,
        title=event.title,
        source=event.source,
        event_date=event.event_date,
        place=event.place,
        themes=_readable_themes(event.themes),
        organizations=event.organizations[:MAX_ORGANIZATIONS],
        matched_entities=event.matched_entities,
        cluster_size=event.cluster_size,
    )


def _fallback(site: Site, events: list[Event], error: Exception) -> TriageResult:
    log.exception("triage.failed", extra={"site_id": site.site_id, "error": str(error)})
    return TriageResult(
        site_id=site.site_id,
        considered=len(events),
        selected=[
            TriageSelection(event_id=e.event_id, reason="Triage unavailable; top-ranked.")
            for e in events[:FALLBACK_COUNT]
        ],
        fallback=True,
    )


async def run(
    site: Site, events: list[Event], ctx: RunContext, *, draft: Drafter = _gemini_triage
) -> TriageResult:
    """Choose which of ``events`` (in rank order) go to the verifier."""
    events = events[:MAX_CANDIDATES]
    with ctx.bind():
        by_id = {f"C{n}": e for n, e in enumerate(events, 1)}
        if not events:
            return TriageResult(site_id=site.site_id, considered=0, selected=[])
        payload = TriageInput(
            site=site, candidates=[to_candidate(e, cid) for cid, e in by_id.items()]
        )
        try:
            picks = (await draft(payload)).picks
        except Exception as exc:  # a triage outage must not drop the site from the brief
            return _fallback(site, events, exc)
        unknown = [p.candidate_id for p in picks if p.candidate_id not in by_id]
        if unknown:
            log.warning("triage.unknown_ids", extra={"site_id": site.site_id, "ids": unknown})
        chosen = {p.candidate_id: p.reason for p in picks if p.candidate_id in by_id}
        # Keep the ranking order, not the order the model listed them in.
        selected = [
            TriageSelection(event_id=e.event_id, reason=chosen[cid])
            for cid, e in by_id.items()
            if cid in chosen
        ]
        log.info(
            "triage.done",
            extra={"site_id": site.site_id, "considered": len(events), "selected": len(selected)},
        )
        return TriageResult(site_id=site.site_id, considered=len(events), selected=selected)
