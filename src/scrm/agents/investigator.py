"""Investigator: an ADK agent with tools that checks high-risk claims and unverifiable stories.

Runs once per story, after the report writer has merged findings into claims:
    investigate_claim   a brief claim with risk >= ``RISK_THRESHOLD`` (severity x impact),
                        however many findings it was built from
    investigate_story   a story the verifier marked "unverifiable"
It looks for corroborating sources, checks whether the disruption is ongoing or resolved,
and returns a short evidence summary with a confidence level.

Tools (closures over one story, so state never leaks between stories):
    search_news(keywords, this_site_only)  guarded keyword search of scrm.gkg_near_sites
                                           and scrm.events_near_sites via BigQueryTool:
                                           fixed SQL, keywords passed as parameters
    read_article(result_id)                fetch an article by result ID

The model never handles URLs: the story's own sources are F1, F2, ... and search results
come back as R1, R2, .... Python maps cited IDs back to URLs, and read_article can only
fetch those pages. A before_tool_callback caps the agent at ``MAX_TOOL_CALLS`` tool calls
per story and logs every call (run ID included via the bound RunContext);
``max_llm_calls`` is a hard backstop.

Input:  InvestigatorInput (a claim and its sources, or one unverifiable story)
Output: Investigation
"""

import asyncio
import re
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from datetime import date, timedelta
from typing import Any, Literal

from google.adk.agents import LlmAgent, RunConfig
from google.adk.agents.invocation_context import LlmCallsLimitExceededError
from google.cloud import bigquery
from google.genai import types
from pydantic import HttpUrl

from scrm.agents import clustering
from scrm.agents._adk import run_structured
from scrm.agents.verifier import quote_in_text
from scrm.config import Settings, get_settings
from scrm.schemas import (
    Article,
    BriefItem,
    FetchFailure,
    Investigation,
    InvestigationEvidence,
    InvestigationJudgement,
    InvestigatorInput,
    InvestigatorSource,
    Site,
    Verdict,
    VerifiedEvent,
)
from scrm.telemetry import get_logger
from scrm.tools.article_fetcher import fetch_article
from scrm.tools.bigquery_tool import BigQueryTool
from scrm.urls import article_key, unique_urls

NAME = "investigator"
INPUT_SCHEMA = InvestigatorInput
OUTPUT_SCHEMA = Investigation
DESCRIPTION = (__doc__ or "").splitlines()[0]

RISK_THRESHOLD = 12
MAX_TOOL_CALLS = 8
MAX_LLM_CALLS = MAX_TOOL_CALLS + 4  # tool round-trips plus the final answer, with slack
MAX_KEYWORDS = 5
MAX_RESULTS = 10
MAX_ARTICLE_CHARS = 5_000
LOOKBACK_DAYS = 7  # search from a week before the finding, up to the end of the run

log = get_logger(__name__)

INSTRUCTION = f"""\
You investigate one story for a supply chain risk team. The input is JSON with a
monitored site, the subject ("claim": a high-risk claim from the brief, or
"unverifiable_story": a story whose article could not be read, with its headline words
(story_title) and triage's reason for picking it (triage_note) when known), the claim text, the
story's own sources F1, F2, ... (with what the verifier concluded: verdict, scope, reason,
quote, date), risk scores if any, and the last date you may search (search_until).

Your job:
1. Corroborate: find other, independent reports of the same disruption.
2. Status: decide whether the disruption is still ongoing or has been resolved, using
   the most recent reports up to search_until.

Tools (at most {MAX_TOOL_CALLS} calls in total, so plan them):
- search_news(keywords, this_site_only): keyword search over news near monitored sites.
  Use 2-4 distinctive words likely to appear in headlines or organisation names, in the
  language of local coverage when useful (e.g. "haven staking" for Dutch). Results come
  back newest first as IDs R1, R2, ... with headline words, source, date and place.
- read_article(result_id): read an article by ID: the story's own sources (F1, F2, ...)
  or search results (R1, R2, ...).

How to work, so that the same story gets the same answer every time:
1. Read the story's own sources first (F1, ...).
2. Search with the most distinctive words from the claim or the source titles (a
   company, place or event name), then once more with words for an end or recovery
   (e.g. "reopened", "restarted", "resumed", "lifted") to check whether it is over.
3. Read the most recent relevant results.

Decide with these rules, not by impression:
- status "resolved": a report dated after the event explicitly says it ended, reopened,
  restarted or was lifted. "ongoing": a report within 7 days of search_until says it
  continues, or a source gives an end date after search_until. Otherwise "unclear".
- confidence "high": two or more independent sources (different publishers) agree with
  the claim and nothing you read contradicts it. "medium": exactly one independent
  source agrees, or sources partly conflict. "low": none found, or they mostly conflict.
- If the tools return nothing useful: status "unclear", confidence "low".

Return:
- summary: at most three sentences in English with what the evidence shows.
- status, confidence: by the rules above.
- corroborating_ids: search result IDs (R...) that independently support the story.
- evidence: up to five short excerpts copied VERBATIM from articles you read (or a
  search result's title), each with its result_id and stance: "supports" or
  "contradicts" the claim, or "context". Always include evidence that contradicts any
  part of the claim (for example, reports that traffic has returned to normal).
Base everything on what the tools returned; never invent sources or facts.
"""

# --------------------------------------------------------------------------------------
# Guarded search
# --------------------------------------------------------------------------------------

SEARCH_SQL = """
WITH hits AS (
  SELECT url, source, DATE(published_at) AS day, place
  FROM `{gkg}`
  WHERE published_at >= TIMESTAMP(@start_date)
    AND published_at < TIMESTAMP(DATE_ADD(@end_date, INTERVAL 1 DAY))
    AND (NOT @this_site_only OR site_id = @site_id)
    AND {gkg_conditions}
  UNION ALL
  SELECT source_url, NULL, event_date, place
  FROM `{events}`
  WHERE event_date BETWEEN @start_date AND @end_date
    AND (NOT @this_site_only OR site_id = @site_id)
    AND {event_conditions}
)
SELECT url, ANY_VALUE(source) AS source, MIN(day) AS day, ANY_VALUE(place) AS place
FROM hits
WHERE url IS NOT NULL
GROUP BY url
ORDER BY day DESC
LIMIT {limit}
"""


def clean_keywords(keywords: str) -> list[str]:
    """Lowercase alphanumeric words of 3+ characters, deduplicated, at most MAX_KEYWORDS."""
    words = re.findall(r"[^\W_]{3,}", keywords.lower())
    return list(dict.fromkeys(words))[:MAX_KEYWORDS]


def search_sql(tool: BigQueryTool, n_keywords: int) -> str:
    """The search statement for ``n_keywords`` keywords; every keyword must match."""

    def conditions(column: str) -> str:
        return " AND ".join(f"REGEXP_CONTAINS({column}, @k{i})" for i in range(n_keywords))

    prefix = f"{tool.allowed_project}.{tool.allowed_dataset}"
    return SEARCH_SQL.format(
        gkg=f"{prefix}.gkg_near_sites",
        events=f"{prefix}.events_near_sites",
        gkg_conditions=conditions("LOWER(CONCAT(url, ' ', IFNULL(organizations, '')))"),
        event_conditions=conditions("LOWER(source_url)"),
        limit=MAX_RESULTS,
    )


def search_params(
    keywords: list[str], site_id: str, start: date, end: date, this_site_only: bool
) -> list[bigquery.ScalarQueryParameter]:
    return [
        bigquery.ScalarQueryParameter("start_date", "DATE", start),
        bigquery.ScalarQueryParameter("end_date", "DATE", end),
        bigquery.ScalarQueryParameter("site_id", "STRING", site_id),
        bigquery.ScalarQueryParameter("this_site_only", "BOOL", this_site_only),
        *(
            bigquery.ScalarQueryParameter(f"k{i}", "STRING", rf"\b{re.escape(word)}")
            for i, word in enumerate(keywords)
        ),
    ]


# --------------------------------------------------------------------------------------
# Tools for one finding
# --------------------------------------------------------------------------------------

Fetcher = Callable[[HttpUrl | str], Awaitable[Article | FetchFailure]]


@dataclass
class Toolkit:
    """Per-finding tool state: the result-ID registry and the tool-call budget."""

    tool: BigQueryTool
    site: Site
    subject_id: str  # item ID of a claim, or event ID of an unverifiable story
    start: date
    end: date
    fetch: Fetcher = fetch_article
    results: dict[str, str] = field(default_factory=dict)  # result ID -> URL (F0 preset)
    searched: int = 0  # search results registered so far, numbered R1, R2, ...
    texts: dict[str, str] = field(default_factory=dict)  # result ID -> text/title seen
    calls: int = 0
    refused: int = 0

    def before_tool(self, tool: Any, args: dict[str, Any], tool_context: Any) -> dict | None:
        """ADK before_tool_callback: count and log every call, refuse past the budget."""
        self.calls += 1
        allowed = self.calls <= MAX_TOOL_CALLS
        log.info(
            "investigator.tool_call",
            extra={
                "subject_id": self.subject_id,
                "call": self.calls,
                "tool": tool.name,
                "tool_args": args,
                "allowed": allowed,
            },
        )
        if allowed:
            return None
        self.refused += 1
        return {
            "status": "error",
            "message": f"Tool call limit of {MAX_TOOL_CALLS} reached. Give your final answer.",
        }

    def remaining(self) -> int:
        return max(0, MAX_TOOL_CALLS - self.calls)

    async def search_news(self, keywords: str, this_site_only: bool = False) -> dict:
        """Search news near monitored sites for articles matching all keywords.

        Args:
            keywords: 2-4 distinctive words, space separated (e.g. "rhine low water").
            this_site_only: True to search only articles geocoded near this site.

        Returns:
            Matching articles, newest first, each with a result ID for read_article.
        """
        words = clean_keywords(keywords)
        if not words:
            return {"status": "error", "message": "Give at least one word of 3+ letters."}
        sql = search_sql(self.tool, len(words))
        params = search_params(words, self.site.site_id, self.start, self.end, this_site_only)
        try:
            rows = await asyncio.to_thread(self.tool.run_query, sql, params)
        except Exception as exc:  # report to the model instead of failing the investigation
            log.warning("investigator.search_failed", extra={"error": str(exc)})
            return {"status": "error", "message": "Search failed; try different keywords."}
        results = []
        for row in rows:
            self.searched += 1
            result_id = f"R{self.searched}"
            self.results[result_id] = row["url"]
            title = clustering.slug_title(row["url"])
            self.texts[result_id] = title or ""
            results.append(
                {
                    "id": result_id,
                    "title": title,
                    "source": row["source"],
                    "date": str(row["day"]),
                    "place": row["place"],
                }
            )
        return {"status": "ok", "results": results, "tool_calls_left": self.remaining()}

    async def read_article(self, result_id: str) -> dict:
        """Read an article found by search_news, or the finding's own article.

        Args:
            result_id: An ID from search_news results (R1, R2, ...) or F0.

        Returns:
            The article title and text (truncated), or why it could not be read.
        """
        url = self.results.get(result_id.strip().upper())
        if url is None:
            return {"status": "error", "message": f"Unknown result ID {result_id!r}."}
        article = await self.fetch(url)
        if isinstance(article, FetchFailure):
            return {"status": "error", "message": f"Could not read it ({article.reason})."}
        key = result_id.strip().upper()
        shown = article.text[:MAX_ARTICLE_CHARS]
        self.texts[key] = f"{article.title or ''}\n{self.texts.get(key, '')}\n{shown}"
        return {
            "status": "ok",
            "title": article.title,
            "text": shown,
            "tool_calls_left": self.remaining(),
        }


def build_agent(settings: Settings, toolkit: Toolkit) -> LlmAgent:
    """Return the agent for one story. Constructing it makes no network calls.

    Temperature comes from ``settings.investigator_temperature`` (0.2 by default, chosen to
    reduce run-to-run variance; None leaves the model default, which Google recommends for
    Gemini 3.x models).
    """
    config = (
        types.GenerateContentConfig(temperature=settings.investigator_temperature)
        if settings.investigator_temperature is not None
        else None
    )
    return LlmAgent(
        name=NAME,
        model=settings.gemini_model,
        description=DESCRIPTION,
        instruction=INSTRUCTION,
        input_schema=InvestigatorInput,
        output_schema=InvestigationJudgement,
        output_key=NAME,
        tools=[toolkit.search_news, toolkit.read_article],
        before_tool_callback=toolkit.before_tool,
        generate_content_config=config,
    )


# --------------------------------------------------------------------------------------
# Running an investigation
# --------------------------------------------------------------------------------------


def needs_investigation(item: BriefItem) -> bool:
    """A claim is investigated when its risk (severity x impact) reaches the threshold."""
    return item.risk >= RISK_THRESHOLD


Investigate = Callable[[InvestigatorInput, Toolkit], Awaitable[InvestigationJudgement]]


async def _gemini_investigate(
    payload: InvestigatorInput, toolkit: Toolkit
) -> InvestigationJudgement:
    agent = build_agent(get_settings(), toolkit)
    config = RunConfig(max_llm_calls=MAX_LLM_CALLS)
    return await run_structured(agent, payload, InvestigationJudgement, run_config=config)


def _source(source_id: str, verified: VerifiedEvent) -> InvestigatorSource:
    return InvestigatorSource(
        source_id=source_id,
        title=verified.event.title,
        event_date=verified.event.event_date,
        verdict=verified.verdict,
        scope=verified.scope,
        reason=verified.reason,
        quote=verified.quote,
    )


MAX_EVIDENCE = 5


def checked_evidence(
    judgement: InvestigationJudgement, toolkit: Toolkit, subject_id: str
) -> list[InvestigationEvidence]:
    """Keep only excerpts that appear verbatim in what the tools returned for that ID."""
    kept, dropped = [], 0
    for snippet in judgement.evidence[:MAX_EVIDENCE]:
        seen = toolkit.texts.get(snippet.result_id.strip().upper(), "")
        url = toolkit.results.get(snippet.result_id.strip().upper())
        if url and quote_in_text(snippet.quote, seen):
            kept.append(InvestigationEvidence(url=url, quote=snippet.quote, stance=snippet.stance))
        else:
            dropped += 1
    if dropped:
        log.warning("investigator.evidence_dropped", extra={"subject_id": subject_id, "n": dropped})
    return kept


async def investigate_claim(
    item: BriefItem,
    events: list[VerifiedEvent],
    site: Site,
    search_until: date,
    tool: BigQueryTool,
    *,
    investigate: Investigate = _gemini_investigate,
    fetch: Fetcher = fetch_article,
) -> Investigation | None:
    """Investigate one brief claim built from ``events`` (its verified sources)."""
    return await _run(
        subject="claim",
        subject_id=item.item_id or "claim",
        claim=item.claim,
        events=events,
        severity=item.severity,
        impact=item.impact,
        site=site,
        search_until=search_until,
        tool=tool,
        investigate=investigate,
        fetch=fetch,
        item_id=item.item_id,
    )


async def investigate_story(
    verified: VerifiedEvent,
    site: Site,
    search_until: date,
    tool: BigQueryTool,
    *,
    triage_note: str | None = None,
    investigate: Investigate = _gemini_investigate,
    fetch: Fetcher = fetch_article,
) -> Investigation | None:
    """Investigate one story the verifier could not judge ("unverifiable").

    The article itself could not be read, so the investigator gets what is known about the
    story: its headline words from the URL and triage's reason for picking it.
    """
    if verified.verdict is not Verdict.UNVERIFIABLE:
        raise ValueError("investigate_story is for unverifiable stories")
    title = verified.event.title
    return await _run(
        subject="unverifiable_story",
        subject_id=verified.event.event_id,
        claim=title or triage_note or "Untitled story; the article could not be read.",
        story_title=title,
        triage_note=triage_note,
        events=[verified],
        severity=None,
        impact=None,
        site=site,
        search_until=search_until,
        tool=tool,
        investigate=investigate,
        fetch=fetch,
        event_id=verified.event.event_id,
    )


async def _run(
    *,
    subject: Literal["claim", "unverifiable_story"],
    subject_id: str,
    claim: str,
    events: list[VerifiedEvent],
    severity: int | None,
    impact: int | None,
    site: Site,
    search_until: date,
    tool: BigQueryTool,
    investigate: Investigate,
    fetch: Fetcher,
    item_id: str | None = None,
    event_id: str | None = None,
    story_title: str | None = None,
    triage_note: str | None = None,
) -> Investigation | None:
    """Shared body: build the toolkit and input, run the agent, map IDs back to URLs.

    Call inside ``ctx.bind()`` so every tool-call log line carries the run ID.
    Returns None if the investigation itself failed.
    """
    sources = {f"F{n}": v for n, v in enumerate(events, 1)}
    earliest = min(v.event.event_date for v in events)
    toolkit = Toolkit(
        tool=tool,
        site=site,
        subject_id=subject_id,
        start=earliest - timedelta(days=LOOKBACK_DAYS),
        end=search_until,
        fetch=fetch,
        results={sid: str(v.evidence_url) for sid, v in sources.items()},
    )
    payload = InvestigatorInput(
        site=site,
        subject=subject,
        claim=claim,
        story_title=story_title,
        triage_note=triage_note,
        sources=[_source(sid, v) for sid, v in sources.items()],
        search_until=search_until,
        severity=severity,
        impact=impact,
    )
    hit_limit = False
    try:
        judgement = await investigate(payload, toolkit)
    except LlmCallsLimitExceededError:
        hit_limit = True
        judgement = InvestigationJudgement(
            summary="The investigation reached its model-call limit before concluding.",
            status="unclear",
            confidence="low",
        )
    except Exception:
        log.exception("investigator.failed", extra={"subject_id": subject_id})
        return None
    own = {article_key(v.evidence_url) for v in events}
    cited = [toolkit.results[i] for i in judgement.corroborating_ids if i in toolkit.results]
    unknown = [i for i in judgement.corroborating_ids if i not in toolkit.results]
    if unknown:
        log.warning("investigator.unknown_ids", extra={"subject_id": subject_id, "ids": unknown})
    source_urls = [v.evidence_url for v in events]
    evidence = checked_evidence(judgement, toolkit, subject_id)
    result = Investigation(
        item_id=item_id,
        event_id=event_id,
        title=story_title or triage_note or claim,
        site_id=site.site_id,
        source_url=source_urls[0],
        source_urls=source_urls,
        trigger="high_risk" if subject == "claim" else "unverifiable",
        summary=judgement.summary,
        status=judgement.status,
        confidence=judgement.confidence,
        corroborating_urls=[u for u in unique_urls(cited) if article_key(u) not in own],
        tool_calls=min(toolkit.calls, MAX_TOOL_CALLS),
        hit_limit=hit_limit or toolkit.refused > 0,
        evidence=evidence,
    )
    log.info(
        "investigator.done",
        extra={
            "subject_id": subject_id,
            "subject": subject,
            "status": result.status,
            "confidence": result.confidence,
            "corroborating": len(result.corroborating_urls),
            "tool_calls": result.tool_calls,
        },
    )
    return result
