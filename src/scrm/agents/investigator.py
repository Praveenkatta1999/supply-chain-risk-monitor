"""Investigator: an ADK agent with tools that checks high-risk and unverifiable findings.

Runs only for findings scored risk >= ``RISK_THRESHOLD`` (severity x impact) or marked
"unverifiable". It looks for corroborating sources, checks whether the disruption is
ongoing or resolved, and returns a short evidence summary with a confidence level.

Tools (closures over one finding, so state never leaks between findings):
    search_news(keywords, this_site_only)  guarded keyword search of scrm.gkg_near_sites
                                           and scrm.events_near_sites via BigQueryTool:
                                           fixed SQL, keywords passed as parameters
    read_article(result_id)                fetch an article by result ID

The model never handles URLs: search results come back as R1, R2, ... and the finding's
own article is F0. Python maps cited IDs back to URLs, and read_article can only fetch
pages the search returned. A before_tool_callback caps the agent at ``MAX_TOOL_CALLS``
tool calls per finding and logs every call (run ID included via the bound RunContext);
``max_llm_calls`` is a hard backstop.

Input:  InvestigatorInput (built from a verified event, its score and the run's dates)
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
from pydantic import HttpUrl

from scrm.agents import clustering
from scrm.agents._adk import run_structured
from scrm.config import Settings, get_settings
from scrm.schemas import (
    Article,
    FetchFailure,
    Investigation,
    InvestigationJudgement,
    InvestigatorInput,
    RiskScore,
    Site,
    VerifiedEvent,
)
from scrm.telemetry import get_logger
from scrm.tools.article_fetcher import fetch_article
from scrm.tools.bigquery_tool import BigQueryTool

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
You investigate one finding for a supply chain risk team. The input is JSON with a
monitored site and a finding: what the verifier concluded (verdict, scope, reason, quote),
its risk scores if any, the date, and the last date you may search (search_until).
Findings are either high risk or "unverifiable" (the source article could not be read).

Your job:
1. Corroborate: find other, independent reports of the same disruption.
2. Status: decide whether the disruption is still ongoing or has been resolved, using
   the most recent reports up to search_until.

Tools (at most {MAX_TOOL_CALLS} calls in total, so plan them):
- search_news(keywords, this_site_only): keyword search over news near monitored sites.
  Use 2-4 distinctive words likely to appear in headlines or organisation names, in the
  language of local coverage when useful (e.g. "haven staking" for Dutch). Results come
  back newest first as IDs R1, R2, ... with headline words, source, date and place.
- read_article(result_id): read an article by ID. F0 is the finding's own article.

Return:
- summary: at most three sentences in English with what the evidence shows.
- status: "ongoing", "resolved" or "unclear".
- confidence: "high" (several independent reports agree and are recent), "medium" (some
  support), or "low" (little or conflicting evidence).
- corroborating_ids: IDs of results that independently support the finding (not F0).
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
    event_id: str
    start: date
    end: date
    fetch: Fetcher = fetch_article
    results: dict[str, str] = field(default_factory=dict)  # result ID -> URL (F0 preset)
    searched: int = 0  # search results registered so far, numbered R1, R2, ...
    calls: int = 0
    refused: int = 0

    def before_tool(self, tool: Any, args: dict[str, Any], tool_context: Any) -> dict | None:
        """ADK before_tool_callback: count and log every call, refuse past the budget."""
        self.calls += 1
        allowed = self.calls <= MAX_TOOL_CALLS
        log.info(
            "investigator.tool_call",
            extra={
                "event_id": self.event_id,
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
            results.append(
                {
                    "id": result_id,
                    "title": clustering.slug_title(row["url"]),
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
        return {
            "status": "ok",
            "title": article.title,
            "text": article.text[:MAX_ARTICLE_CHARS],
            "tool_calls_left": self.remaining(),
        }


def build_agent(settings: Settings, toolkit: Toolkit) -> LlmAgent:
    """Return the agent for one finding. Constructing it makes no network calls."""
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
    )


# --------------------------------------------------------------------------------------
# Running an investigation
# --------------------------------------------------------------------------------------


def needs_investigation(verified: VerifiedEvent, score: RiskScore | None) -> bool:
    if verified.verdict.value == "unverifiable":
        return True
    return score is not None and score.severity * score.impact >= RISK_THRESHOLD


Investigate = Callable[[InvestigatorInput, Toolkit], Awaitable[InvestigationJudgement]]


async def _gemini_investigate(
    payload: InvestigatorInput, toolkit: Toolkit
) -> InvestigationJudgement:
    agent = build_agent(get_settings(), toolkit)
    config = RunConfig(max_llm_calls=MAX_LLM_CALLS)
    return await run_structured(agent, payload, InvestigationJudgement, run_config=config)


async def run(
    verified: VerifiedEvent,
    score: RiskScore | None,
    site: Site,
    search_until: date,
    tool: BigQueryTool,
    *,
    investigate: Investigate = _gemini_investigate,
    fetch: Fetcher = fetch_article,
) -> Investigation | None:
    """Investigate one finding. Returns None if the investigation itself failed.

    Call inside ``ctx.bind()`` so every tool-call log line carries the run ID.
    """
    event = verified.event
    trigger: Literal["high_risk", "unverifiable"] = (
        "unverifiable" if verified.verdict.value == "unverifiable" else "high_risk"
    )
    toolkit = Toolkit(
        tool=tool,
        site=site,
        event_id=event.event_id,
        start=event.event_date - timedelta(days=LOOKBACK_DAYS),
        end=search_until,
        fetch=fetch,
        results={"F0": str(verified.evidence_url)},
    )
    payload = InvestigatorInput(
        site=site,
        finding_id="F0",
        title=event.title,
        event_date=event.event_date,
        search_until=search_until,
        verdict=verified.verdict,
        scope=verified.scope,
        reason=verified.reason,
        quote=verified.quote,
        severity=score.severity if score else None,
        impact=score.impact if score else None,
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
        log.exception("investigator.failed", extra={"event_id": event.event_id})
        return None
    cited = [toolkit.results[i] for i in judgement.corroborating_ids if i in toolkit.results]
    unknown = [i for i in judgement.corroborating_ids if i not in toolkit.results]
    if unknown:
        log.warning("investigator.unknown_ids", extra={"event_id": event.event_id, "ids": unknown})
    own = str(verified.evidence_url)
    result = Investigation(
        event_id=event.event_id,
        site_id=site.site_id,
        source_url=verified.evidence_url,
        trigger=trigger,
        summary=judgement.summary,
        status=judgement.status,
        confidence=judgement.confidence,
        corroborating_urls=list(dict.fromkeys(u for u in cited if u != own)),
        tool_calls=min(toolkit.calls, MAX_TOOL_CALLS),
        hit_limit=hit_limit or toolkit.refused > 0,
    )
    log.info(
        "investigator.done",
        extra={
            "event_id": event.event_id,
            "trigger": trigger,
            "status": result.status,
            "confidence": result.confidence,
            "corroborating": len(result.corroborating_urls),
            "tool_calls": result.tool_calls,
        },
    )
    return result
