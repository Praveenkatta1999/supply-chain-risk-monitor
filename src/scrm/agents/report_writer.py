"""Report writer: produces the daily risk brief in Markdown with source links.

Gemini drafts the claims (``ReportDraft``): it merges findings about the same event and
writes each as a short English sentence citing the source URLs it used. Python then
checks every cited URL is one of the confirmed findings and renders the Markdown itself,
so each claim carries its links by construction. RiskBrief validation re-checks this.

Only "yes" verdicts become claims; rejected and unverifiable candidates are counted in
the brief's coverage line but never stated as facts.

Input:  ReportRequest
Output: RiskBrief
"""

from collections import Counter
from collections.abc import Awaitable, Callable
from datetime import UTC, datetime

from google.adk.agents import LlmAgent

from scrm.agents._adk import run_structured
from scrm.config import Settings, get_settings
from scrm.schemas import (
    BriefItem,
    Finding,
    ReportDraft,
    ReportDraftInput,
    ReportRequest,
    RiskBrief,
    Site,
    Verdict,
    VerifiedEvent,
)
from scrm.telemetry import RunContext, get_logger

NAME = "report_writer"
INPUT_SCHEMA = ReportRequest
OUTPUT_SCHEMA = RiskBrief
DESCRIPTION = (__doc__ or "").splitlines()[0]

log = get_logger(__name__)

INSTRUCTION = """\
You write claims for a daily supply chain risk brief read by procurement and logistics
managers. The input is JSON with the monitored sites and a list of findings: each is a
confirmed disruption with a site_id, date, source_url, a one-sentence reason and a
verbatim quote (possibly not in English) from the source article.

Write one item per distinct disruption:
- If several findings describe the same underlying event, merge them into one item and
  list all of their source_urls.
- claim: one or two plain English sentences saying what is happening and why it matters
  for the site. Use only facts present in the findings; do not add numbers, dates or
  causes that are not there.
- source_urls: copy the source_url values of the findings the claim is based on, exactly.
- Order items from most to least disruptive.
"""


def build_agent(settings: Settings) -> LlmAgent:
    """Return the ADK agent definition. Constructing it makes no network calls."""
    return LlmAgent(
        name=NAME,
        model=settings.gemini_model,
        description=DESCRIPTION,
        instruction=INSTRUCTION,
        input_schema=ReportDraftInput,
        output_schema=ReportDraft,
        output_key=NAME,
    )


Drafter = Callable[[ReportDraftInput], Awaitable[ReportDraft]]


async def _gemini_draft(payload: ReportDraftInput) -> ReportDraft:
    return await run_structured(build_agent(get_settings()), payload, ReportDraft)


def findings_from(events: list[VerifiedEvent]) -> list[Finding]:
    return [
        Finding(
            site_id=v.event.site_id,
            event_date=v.event.event_date,
            source_url=v.evidence_url,
            reason=v.reason,
            quote=v.quote or "",
        )
        for v in events
        if v.verdict is Verdict.YES
    ]


def items_from_draft(draft: ReportDraft, findings: list[Finding]) -> list[BriefItem]:
    """Keep only draft items whose every URL is a confirmed finding for that site."""
    allowed = {(f.site_id, str(f.source_url)) for f in findings}
    items = []
    for item in draft.items:
        unknown = [str(u) for u in item.source_urls if (item.site_id, str(u)) not in allowed]
        if unknown:
            log.warning("report.item_dropped", extra={"claim": item.claim, "unknown": unknown})
            continue
        items.append(
            BriefItem(site_id=item.site_id, claim=item.claim, source_urls=item.source_urls)
        )
    return items


def render_markdown(request: ReportRequest, items: list[BriefItem], generated_at: datetime) -> str:
    dr = request.date_range
    lines = [
        f"# Supply chain risk brief: {dr.start} to {dr.end}",
        "",
        f"Run `{request.run_id}`, generated {generated_at:%Y-%m-%d %H:%M} UTC.",
    ]
    for site in request.sites:
        lines += ["", f"## {site.site_id}: {site.site_name}", ""]
        site_items = [i for i in items if i.site_id == site.site_id]
        if not site_items:
            lines.append("No confirmed disruptions in this period.")
        for item in site_items:
            links = ", ".join(f"[source {n}]({url})" for n, url in enumerate(item.source_urls, 1))
            lines.append(f"- {item.claim} ({links})")
        lines += ["", _coverage_line(site, request.verified_events)]
    return "\n".join(lines) + "\n"


def _coverage_line(site: Site, events: list[VerifiedEvent]) -> str:
    counts = Counter(v.verdict for v in events if v.event.site_id == site.site_id)
    total = sum(counts.values())
    return (
        f"_Coverage: {total} candidate articles reviewed: {counts[Verdict.YES]} confirmed, "
        f"{counts[Verdict.NO]} rejected as false positives, "
        f"{counts[Verdict.UNVERIFIABLE]} unverifiable (dead link, paywall or unreadable)._"
    )


async def run(
    request: ReportRequest, ctx: RunContext, *, draft: Drafter = _gemini_draft
) -> RiskBrief:
    """Write the brief for one run within ``ctx.run_id``."""
    with ctx.bind():
        findings = findings_from(request.verified_events)
        log.info("report.start", extra={"findings": len(findings)})
        items: list[BriefItem] = []
        if findings:
            payload = ReportDraftInput(
                date_range=request.date_range, sites=request.sites, findings=findings
            )
            items = items_from_draft(await draft(payload), findings)
        generated_at = datetime.now(UTC)
        brief = RiskBrief(
            run_id=request.run_id,
            generated_at=generated_at,
            date_range=request.date_range,
            site_ids=[s.site_id for s in request.sites],
            items=items,
            markdown=render_markdown(request, items, generated_at),
        )
        log.info("report.done", extra={"items": len(items)})
        return brief
