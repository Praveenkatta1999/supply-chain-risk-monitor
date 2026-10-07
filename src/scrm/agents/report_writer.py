"""Report writer: produces the daily risk brief in Markdown with source links.

Gemini drafts the claims (``ReportDraft``): it merges findings about the same event and
writes each as a short English sentence citing the findings it used. Python then
maps the cited finding IDs back to their verified URLs (the model never handles URLs),
attaches risk scores and supporting URLs, orders items by risk (severity x impact) and
renders the Markdown itself, so each claim carries its links by construction. If a draft
item is invalid or a finding goes uncited, the verifier's reason is used as the claim.
RiskBrief validation re-checks the links.

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
    Investigation,
    ReportDraft,
    ReportDraftInput,
    ReportRequest,
    RiskBrief,
    Scope,
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
confirmed disruption with a finding_id, site_id, date, a one-sentence reason and a
verbatim quote (possibly not in English) from the source article.

Write one item per distinct disruption, covering every finding:
- If several findings for the SAME site describe the same underlying event, merge them
  into one item. Never merge findings across sites, findings about different events, or a
  "direct" finding with an "indirect" one (scope says which each finding is).
- claim: one or two plain English sentences saying what is happening and why it matters
  for the site. Use only facts present in the findings; do not add numbers, dates or
  causes that are not there.
- finding_ids: the finding_id of every finding the claim is based on.
- Each finding may carry severity and impact scores; do not restate them in the claim.
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


Evidence = dict[str, tuple[Finding, VerifiedEvent]]  # finding_id -> (finding, event)


def gather_evidence(request: ReportRequest) -> Evidence:
    """Number the confirmed events F1, F2, ... and attach their risk scores."""
    scores = {(s.site_id, str(s.source_url)): s for s in request.scores}
    confirmed = [v for v in request.verified_events if v.verdict is Verdict.YES]
    evidence: Evidence = {}
    for n, v in enumerate(confirmed, 1):
        score = scores.get((v.event.site_id, str(v.evidence_url)))
        finding = Finding(
            finding_id=f"F{n}",
            site_id=v.event.site_id,
            scope=v.scope,
            event_date=v.event.event_date,
            reason=v.reason,
            quote=v.quote or "",
            severity=score.severity if score else None,
            impact=score.impact if score else None,
        )
        evidence[finding.finding_id] = (finding, v)
    return evidence


def _brief_item(site_id: str, claim: str, cited: list[tuple[Finding, VerifiedEvent]]) -> BriefItem:
    """Build an item whose sources, supporting links and scores all come from ``cited``."""
    worst, _ = max(cited, key=lambda fv: (fv[0].severity or 0) * (fv[0].impact or 0))
    supporting = dict.fromkeys(u for _, v in cited for u in v.event.supporting_urls)
    scopes = {f.scope for f, _ in cited}
    # A claim built on any direct finding is direct; scope is None only for legacy runs.
    scope = Scope.DIRECT if Scope.DIRECT in scopes else next(iter(scopes - {None}), None)
    return BriefItem(
        site_id=site_id,
        claim=claim,
        scope=scope,
        severity=worst.severity,
        impact=worst.impact,
        source_urls=[v.evidence_url for _, v in cited],
        supporting_urls=list(supporting),
    )


def items_from_draft(draft: ReportDraft, evidence: Evidence) -> list[BriefItem]:
    """Turn draft items into brief items, ordered by risk (highest first).

    A draft item citing an unknown finding ID, or a finding from another site, is dropped.
    Any confirmed finding left uncited then gets its own item, worded by the verifier's
    quote-checked reason, so a confirmed disruption never disappears from the brief.
    """
    items: list[BriefItem] = []
    cited_ids: set[str] = set()
    for item in draft.items:
        ids = item.finding_ids
        if not all(f in evidence and evidence[f][0].site_id == item.site_id for f in ids):
            log.warning("report.item_dropped", extra={"claim": item.claim, "finding_ids": ids})
            continue
        items.append(_brief_item(item.site_id, item.claim, [evidence[f] for f in ids]))
        cited_ids.update(ids)
    for fid, (finding, event) in evidence.items():
        if fid not in cited_ids:
            log.warning(
                "report.finding_uncited", extra={"finding_id": fid, "url": str(event.evidence_url)}
            )
            items.append(_brief_item(finding.site_id, finding.reason, [(finding, event)]))
    return sorted(items, key=lambda i: i.risk, reverse=True)


def _links(urls: list, label: str) -> str:
    return ", ".join(f"[{label} {n}]({url})" for n, url in enumerate(urls, 1))


def _investigation_line(inv: Investigation) -> str:
    line = (
        f"   Investigator ({inv.status}, {inv.confidence} confidence, "
        f"{inv.tool_calls} tool calls): {inv.summary}"
    )
    if inv.corroborating_urls:
        line += f" Corroborating: {_links(inv.corroborating_urls, 'c')}."
    return line


def _render_item(
    rank: int, item: BriefItem, site: Site, investigations: dict[str, Investigation]
) -> list[str]:
    score = (
        f"severity {item.severity}, impact {item.impact}, risk {item.risk}/25"
        if item.severity
        else "unscored"
    )
    lines = [
        f"{rank}. **{site.site_id} {site.site_name}** ({score})",
        f"   {item.claim} ({_links(item.source_urls, 'source')})",
    ]
    if item.supporting_urls:
        lines.append(f"   Also reported, not verified: {_links(item.supporting_urls, 'link')}")
    lines += [
        _investigation_line(investigations[str(url)])
        for url in item.source_urls
        if str(url) in investigations
    ]
    return lines


def _render_unverifiable(request: ReportRequest, sites: dict[str, Site]) -> list[str]:
    """Stories whose article could not be read, with what the investigator found."""
    events = {str(v.evidence_url): v for v in request.verified_events}
    rows = [i for i in request.investigations if i.trigger == "unverifiable"]
    if not rows:
        return []
    lines = [
        "",
        "## Unverifiable stories, investigated",
        "",
        "Not confirmed: the source article could not be read. Shown with the investigator's",
        "findings so they can be followed up.",
        "",
    ]
    for inv in rows:
        site = sites[inv.site_id]
        title = events[str(inv.source_url)].event.title or "untitled story"
        lines += [
            f"- **{site.site_id} {site.site_name}**: {title} ([story]({inv.source_url}))",
            _investigation_line(inv),
        ]
    return lines


def _coverage_table(request: ReportRequest) -> list[str]:
    lines = [
        "| Site | Candidates | Sent to verifier | Direct | Indirect | Rejected | Unverifiable |",
        "|---|---|---|---|---|---|---|",
    ]
    for site in request.sites:
        events = [v for v in request.verified_events if v.event.site_id == site.site_id]
        verdicts = Counter(v.verdict for v in events)
        scopes = Counter(v.scope for v in events)
        considered = request.considered.get(site.site_id, len(events))
        lines.append(
            f"| {site.site_id} {site.site_name} | {considered} | {len(events)} "
            f"| {scopes[Scope.DIRECT]} | {scopes[Scope.INDIRECT]} | {verdicts[Verdict.NO]} "
            f"| {verdicts[Verdict.UNVERIFIABLE]} |"
        )
    return lines


def _section(
    title: str,
    items: list[BriefItem],
    sites: dict[str, Site],
    investigations: dict[str, Investigation],
) -> list[str]:
    lines = ["", f"## {title}", ""]
    if not items:
        return [*lines, "None in this period."]
    for rank, item in enumerate(items, 1):
        lines += _render_item(rank, item, sites[item.site_id], investigations)
    return lines


def render_markdown(request: ReportRequest, items: list[BriefItem], generated_at: datetime) -> str:
    dr = request.date_range
    sites = {s.site_id: s for s in request.sites}
    investigations = {str(i.source_url): i for i in request.investigations}
    direct = [i for i in items if i.scope is not Scope.INDIRECT]  # includes legacy None
    indirect = [i for i in items if i.scope is Scope.INDIRECT]
    lines = [
        f"# Supply chain risk brief: {dr.start} to {dr.end}",
        "",
        f"Run `{request.run_id}`, generated {generated_at:%Y-%m-%d %H:%M} UTC. "
        f"Sites: {', '.join(sites)}. Risk = severity x impact, each 1-5, highest first.",
        *_section("Direct disruptions: the site itself", direct, sites, investigations),
        *_section(
            "Indirect disruptions: connected routes and regions", indirect, sites, investigations
        ),
    ]
    quiet = [s for s in request.sites if not any(i.site_id == s.site_id for i in items)]
    if quiet:
        names = ", ".join(f"{s.site_id} {s.site_name}" for s in quiet)
        lines += ["", f"No confirmed disruptions: {names}."]
    lines += _render_unverifiable(request, sites)
    lines += ["", "## Coverage", "", *_coverage_table(request)]
    return "\n".join(lines) + "\n"


async def run(
    request: ReportRequest, ctx: RunContext, *, draft: Drafter = _gemini_draft
) -> RiskBrief:
    """Write the brief for one run within ``ctx.run_id``."""
    with ctx.bind():
        evidence = gather_evidence(request)
        log.info("report.start", extra={"findings": len(evidence)})
        items: list[BriefItem] = []
        if evidence:
            payload = ReportDraftInput(
                date_range=request.date_range,
                sites=request.sites,
                findings=[finding for finding, _ in evidence.values()],
            )
            items = items_from_draft(await draft(payload), evidence)
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
