"""Root agent: answers a plain-language question by choosing sites and dates, then running
the pipeline.

Example: "What are the risks to our chip supply this week?" The agent looks at the
monitored sites, picks the relevant ones (for example the semiconductor and memory fabs)
and a date range ("this week" = the last 7 days), runs the pipeline once, and answers in
a few sentences. The brief itself, with every source link, is attached by Python from the
pipeline result; the model never retypes it, so it cannot alter claims or links.

The pipeline is exposed as a tool rather than as LLM sub-agents: its steps are fixed and
deterministic (see orchestrator.py), so a model has nothing to decide inside it.

Tools (closures over one question):
    list_sites()                                   monitored sites with type, company and
                                                   component
    run_risk_brief(site_ids, start_date, end_date) runs orchestrator.run_pipeline; returns
                                                   a compact summary of the claims (no URLs)
run_risk_brief validates its arguments (known sites, at most ``MAX_SITES``, at most
``MAX_DAYS`` days, not after today) and runs at most once per question; a
before_tool_callback caps and logs every tool call.

Input:  RootRequest
Output: RootResult (the model's RootAnswer plus the PipelineResult it produced)
"""

import asyncio
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from datetime import date
from typing import Any

from google.adk.agents import LlmAgent, RunConfig

from scrm.agents import orchestrator, query_agent
from scrm.agents._adk import run_structured
from scrm.config import Settings, get_settings
from scrm.schemas import (
    BriefRequest,
    DateRange,
    PipelineResult,
    RootAnswer,
    RootRequest,
    RootResult,
    Site,
)
from scrm.telemetry import RunContext, get_logger
from scrm.tools.bigquery_tool import BigQueryTool

NAME = "root_agent"
INPUT_SCHEMA = RootRequest
OUTPUT_SCHEMA = RootResult
DESCRIPTION = (__doc__ or "").splitlines()[0]

MAX_SITES = 6
MAX_DAYS = 31
MAX_TOOL_CALLS = 4
MAX_LLM_CALLS = 8

log = get_logger(__name__)

INSTRUCTION = f"""\
You answer supply chain risk questions for a laptop maker's risk team. The input is JSON
with the question and today's date.

Steps:
1. Call list_sites to see the monitored sites: factories (with the company and the
   component they supply), ports and shipping chokepoints.
2. Choose the sites the question is about, at most {MAX_SITES}. For a component or
   supplier question ("chip supply", "displays", "batteries") pick the factories that
   make it; add ports or chokepoints only if the question is about shipping or routes.
   For a place or route question, pick the sites there.
3. Choose the date range, ending today unless the question says otherwise: "this week"
   or "lately" = the last 7 days including today; "this month" = the last 30 days. At
   most {MAX_DAYS} days.
4. Call run_risk_brief once with those site_ids and dates (YYYY-MM-DD).
5. Answer from what run_risk_brief returned, never from your own knowledge:
   - summary: two to four sentences answering the question. Name the highest risks with
     their status (ongoing, resolved, unclear) and confidence, and say plainly if
     nothing was confirmed for a site.
   - rationale: one sentence on why you chose these sites and dates.
   - site_ids, start_date, end_date: what you ran.
The full brief with sources is attached to your answer automatically; do not list URLs.
"""

PipelineRunner = Callable[[BriefRequest, RunContext], Awaitable[PipelineResult]]


@dataclass
class RootToolkit:
    """Per-question tool state: the sites, the one pipeline result, the call budget."""

    tool: BigQueryTool
    ctx: RunContext
    today: date
    run_pipeline: PipelineRunner
    sites: dict[str, Site] | None = None
    result: PipelineResult | None = None
    calls: int = 0

    def before_tool(self, tool: Any, args: dict[str, Any], tool_context: Any) -> dict | None:
        """ADK before_tool_callback: count and log every call, refuse past the budget."""
        self.calls += 1
        allowed = self.calls <= MAX_TOOL_CALLS
        log.info(
            "root_agent.tool_call",
            extra={"call": self.calls, "tool": tool.name, "tool_args": args, "allowed": allowed},
        )
        if allowed:
            return None
        return {"status": "error", "message": "Tool call limit reached. Give your answer now."}

    async def _load_sites(self) -> dict[str, Site]:
        if self.sites is None:
            loaded = await asyncio.to_thread(query_agent.load_sites, self.tool, [])
            self.sites = {s.site_id: s for s in loaded}
        return self.sites

    async def list_sites(self) -> dict:
        """List every monitored site.

        Returns:
            Sites with site_id, name, type (factory, port or chokepoint), company,
            component supplied, city and country.
        """
        sites = await self._load_sites()
        return {
            "status": "ok",
            "sites": [
                {
                    "site_id": s.site_id,
                    "name": s.site_name,
                    "type": s.site_type.value,
                    "company": s.company,
                    "component": s.component,
                    "city": s.city,
                    "country": s.country,
                }
                for s in sites.values()
            ],
        }

    def _check(self, site_ids: list[str], start: date, end: date, known: set[str]) -> str | None:
        """Why these arguments are refused, or None if they are fine."""
        unknown = [s for s in site_ids if s not in known]
        if unknown:
            return f"Unknown site_ids {unknown}; use IDs from list_sites."
        if not site_ids or len(site_ids) > MAX_SITES:
            return f"Choose between 1 and {MAX_SITES} sites."
        if start > end:
            return "start_date must be on or before end_date."
        if end > self.today:
            return f"end_date cannot be after today ({self.today})."
        if (end - start).days + 1 > MAX_DAYS:
            return f"The date range can be at most {MAX_DAYS} days."
        return None

    async def run_risk_brief(self, site_ids: list[str], start_date: str, end_date: str) -> dict:
        """Run the risk pipeline for the chosen sites and dates. Call it once.

        Args:
            site_ids: Site IDs from list_sites, at most six.
            start_date: First day, YYYY-MM-DD.
            end_date: Last day, YYYY-MM-DD, not after today.

        Returns:
            A summary of the brief: each claim with its site, scope (direct or indirect),
            status, confidence and risk, plus the sites with nothing confirmed.
        """
        if self.result is not None:
            return {"status": "error", "message": "The brief has already been run. Answer now."}
        try:
            start, end = date.fromisoformat(start_date), date.fromisoformat(end_date)
        except ValueError:
            return {"status": "error", "message": "Dates must be YYYY-MM-DD."}
        site_ids = list(dict.fromkeys(site_ids))
        problem = self._check(site_ids, start, end, set(await self._load_sites()))
        if problem:
            return {"status": "error", "message": problem}
        request = BriefRequest(site_ids=site_ids, date_range=DateRange(start=start, end=end))
        self.result = await self.run_pipeline(request, self.ctx)
        return summarise(self.result)


def summarise(result: PipelineResult) -> dict:
    """What the model sees of the brief: claims and their review outcome, no URLs."""
    items = result.brief.items
    with_claims = {i.site_id for i in items}
    return {
        "status": "ok",
        "date_range": f"{result.brief.date_range.start} to {result.brief.date_range.end}",
        "claims": [
            {
                "site_id": i.site_id,
                "scope": i.scope,
                "status": i.status,
                "confidence": i.confidence,
                "risk": i.risk,
                "claim": i.claim,
                "contradicted": i.contradictions,
            }
            for i in items
        ],
        "sites_with_nothing_confirmed": [s for s in result.brief.site_ids if s not in with_claims],
        "unverifiable_stories_investigated": sum(
            i.trigger == "unverifiable" for i in result.investigations
        ),
    }


def build_agent(settings: Settings, toolkit: RootToolkit) -> LlmAgent:
    """Return the agent for one question. Constructing it makes no network calls."""
    return LlmAgent(
        name=NAME,
        model=settings.gemini_model,
        description=DESCRIPTION,
        instruction=INSTRUCTION,
        input_schema=RootRequest,
        output_schema=RootAnswer,
        output_key=NAME,
        tools=[toolkit.list_sites, toolkit.run_risk_brief],
        before_tool_callback=toolkit.before_tool,
    )


Ask = Callable[[RootRequest, RootToolkit], Awaitable[RootAnswer]]


async def _gemini_ask(payload: RootRequest, toolkit: RootToolkit) -> RootAnswer:
    agent = build_agent(get_settings(), toolkit)
    config = RunConfig(max_llm_calls=MAX_LLM_CALLS)
    return await run_structured(agent, payload, RootAnswer, run_config=config)


async def run(
    question: str,
    ctx: RunContext,
    *,
    today: date,
    tool: BigQueryTool | None = None,
    ask: Ask = _gemini_ask,
    run_pipeline: PipelineRunner = orchestrator.run_pipeline,
) -> RootResult:
    """Answer one question within run ``ctx.run_id``."""
    tool = tool or BigQueryTool.from_settings(get_settings())
    toolkit = RootToolkit(tool=tool, ctx=ctx, today=today, run_pipeline=run_pipeline)
    with ctx.bind():
        log.info("root_agent.start", extra={"question": question, "today": str(today)})
        answer = await ask(RootRequest(question=question, today=today), toolkit)
        if toolkit.result is not None:
            # Report what actually ran, whatever the model wrote in these fields.
            ran = toolkit.result.brief
            answer = answer.model_copy(
                update={
                    "site_ids": ran.site_ids,
                    "start_date": ran.date_range.start,
                    "end_date": ran.date_range.end,
                }
            )
        log.info(
            "root_agent.done",
            extra={"site_ids": answer.site_ids, "ran_pipeline": toolkit.result is not None},
        )
        return RootResult(answer=answer, pipeline=toolkit.result, model_calls=dict(ctx.model_calls))
