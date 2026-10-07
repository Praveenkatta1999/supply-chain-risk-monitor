"""Verifier: reads the source article and decides whether an event really affects the site.

Steps for one candidate:
1. Fetch the article (tools/article_fetcher.py). Dead links, paywalls and other fetch
   failures become verdict "unverifiable" without calling the model: nothing is guessed.
2. Ask Gemini to classify the article with the scope rubric (structured output,
   ``VerifierJudgement``), with a one-sentence reason and a verbatim supporting quote:
       direct        the site itself is affected
       indirect      a connected route, supplier region or hinterland is affected
       not_relevant  no real, current disruption that touches this site
       unverifiable  the text is not the story (consent wall, teaser) or too truncated
   Direct and indirect map to verdict "yes" (with ``scope``), not_relevant to "no".
3. Check the quote really appears in the article. A "yes" whose quote cannot be found
   is downgraded to "unverifiable", so a hallucinated quote never reaches the brief.

Input:  VerificationRequest
Output: VerifiedEvent
"""

import asyncio
import re
from collections.abc import Awaitable, Callable

from google.adk.agents import LlmAgent
from pydantic import HttpUrl

from scrm.agents._adk import run_structured
from scrm.config import Settings, get_settings
from scrm.schemas import (
    Article,
    FetchFailure,
    Scope,
    Verdict,
    VerificationRequest,
    VerifiedEvent,
    VerifierInput,
    VerifierJudgement,
)
from scrm.telemetry import RunContext, get_logger
from scrm.tools.article_fetcher import fetch_article

NAME = "verifier"
INPUT_SCHEMA = VerificationRequest
OUTPUT_SCHEMA = VerifiedEvent
DESCRIPTION = (__doc__ or "").splitlines()[0]

MAX_ARTICLE_CHARS = 20_000
DEFAULT_CONCURRENCY = 5

log = get_logger(__name__)

INSTRUCTION = """\
You verify news for a supply chain risk team. The input is JSON with a monitored site
(a port, factory or shipping chokepoint, with coordinates and a radius in km) and the
text of one news article that a keyword search linked to that site.

Decide whether the article describes a REAL, CURRENT disruption that touches THIS site:
something that stops, slows or threatens operations or cargo flows. Examples: strikes or
blockades, closures, accidents, fires, severe weather, floods or low water, cyberattacks,
congestion, sanctions, conflict or security incidents, outages or production problems.

Classify it with exactly one assessment:

"direct": the site itself is affected. The disruption happens at the site, at its
  operator or company's facilities there, or on its own terminals, berths, fabs, lanes
  or approaches. Example: a strike at the port's container terminals; a fire at the fab.

"indirect": a connected route, supplier region or hinterland is affected, and the
  article makes the link to the site's cargo flows or operations plausible. Example: low
  water on the river that carries the port's inland barge traffic; attacks on shipping
  in the sea lane that feeds the canal; a regional power shortage that hits the fab's area.

"not_relevant": anything else, for example:
- a different place with a similar name, or the site only mentioned in passing;
- historical events, commemorations, feature stories, nothing happening now;
- general politics, markets, crime, business trends or opinion with no operational effect;
- a disruption elsewhere with no plausible link to this site.

"unverifiable": article_text is not the story itself (a cookie or consent wall, paywall
  teaser, error page, list of headlines) or is too truncated to judge. Never guess.

Fields:
- assessment: "direct", "indirect", "not_relevant" or "unverifiable".
- reason: exactly one sentence in English explaining the assessment.
- quote: a short excerpt (one or two sentences) copied VERBATIM from article_text, in
  its original language, that best supports the assessment. Do not translate, paraphrase
  or join separate passages. Required for "direct" and "indirect"; null for "unverifiable".
"""

_VERDICT_AND_SCOPE: dict[str, tuple[Verdict, Scope | None]] = {
    "direct": (Verdict.YES, Scope.DIRECT),
    "indirect": (Verdict.YES, Scope.INDIRECT),
    "not_relevant": (Verdict.NO, Scope.NOT_RELEVANT),
    "unverifiable": (Verdict.UNVERIFIABLE, None),
}


def build_agent(settings: Settings) -> LlmAgent:
    """Return the ADK agent definition. Constructing it makes no network calls."""
    return LlmAgent(
        name=NAME,
        model=settings.gemini_model,
        description=DESCRIPTION,
        instruction=INSTRUCTION,
        input_schema=VerifierInput,
        output_schema=VerifierJudgement,
        output_key=NAME,
    )


Fetcher = Callable[[HttpUrl], Awaitable[Article | FetchFailure]]
Judge = Callable[[VerifierInput], Awaitable[VerifierJudgement]]


async def _gemini_judge(payload: VerifierInput) -> VerifierJudgement:
    return await run_structured(build_agent(get_settings()), payload, VerifierJudgement)


# Curly quotes -> straight quotes, so a model that "fixes" punctuation still matches.
_CURLY_QUOTES = str.maketrans({0x2018: "'", 0x2019: "'", 0x201C: '"', 0x201D: '"'})


def _normalise(text: str) -> str:
    text = text.translate(_CURLY_QUOTES)
    return re.sub(r"\s+", " ", text).strip().casefold()


def quote_in_text(quote: str, text: str) -> bool:
    """True if ``quote`` appears in ``text``, ignoring whitespace, case and curly quotes."""
    return bool(quote.strip()) and _normalise(quote) in _normalise(text)


def _unverifiable(request: VerificationRequest, reason: str) -> VerifiedEvent:
    return VerifiedEvent(
        event=request.event,
        verdict=Verdict.UNVERIFIABLE,
        reason=reason,
        evidence_url=request.event.source_url,
    )


async def run(
    request: VerificationRequest,
    ctx: RunContext,
    *,
    fetch: Fetcher = fetch_article,
    judge: Judge = _gemini_judge,
) -> VerifiedEvent:
    """Verify one candidate event against its source article."""
    with ctx.bind():
        url = request.event.source_url
        article = await fetch(url)
        if isinstance(article, FetchFailure):
            result = _unverifiable(
                request, f"Article could not be read ({article.reason}: {article.detail})."
            )
        else:
            result = await _judge_article(request, article, judge)
        log.info(
            "verifier.done",
            extra={
                "event_id": request.event.event_id,
                "url": str(url),
                "verdict": result.verdict,
                "scope": result.scope,
                "reason": result.reason,
            },
        )
        return result


async def _judge_article(
    request: VerificationRequest, article: Article, judge: Judge
) -> VerifiedEvent:
    text = article.text[:MAX_ARTICLE_CHARS]
    payload = VerifierInput(
        site=request.site,
        event_date=request.event.event_date,
        article_url=article.url,
        article_title=article.title,
        article_text=text,
    )
    try:
        judgement = await judge(payload)
    except Exception as exc:  # one bad model call must not sink the whole batch
        log.exception("verifier.model_failed", extra={"event_id": request.event.event_id})
        return _unverifiable(request, f"Verifier model call failed ({type(exc).__name__}).")

    verdict, scope = _VERDICT_AND_SCOPE[judgement.assessment]
    quote = judgement.quote
    if quote and not quote_in_text(quote, text):
        log.warning(
            "verifier.quote_not_found",
            extra={"event_id": request.event.event_id, "assessment": judgement.assessment},
        )
        if verdict is Verdict.YES:
            return _unverifiable(request, "Model's supporting quote was not found in the article.")
        quote = None
    if verdict is Verdict.YES and not quote:
        return _unverifiable(request, "Model found a disruption but gave no supporting quote.")
    return VerifiedEvent(
        event=request.event,
        verdict=verdict,
        scope=scope,
        reason=judgement.reason,
        quote=quote,
        evidence_url=article.url,
    )


async def run_many(
    requests: list[VerificationRequest],
    ctx: RunContext,
    *,
    limit: asyncio.Semaphore | None = None,
    fetch: Fetcher = fetch_article,
    judge: Judge = _gemini_judge,
) -> list[VerifiedEvent]:
    """Verify many candidates concurrently, returning results in input order.

    ``limit`` caps how many run at once; pass a shared one to cap work across callers.
    """
    semaphore = limit or asyncio.Semaphore(DEFAULT_CONCURRENCY)

    async def one(request: VerificationRequest) -> VerifiedEvent:
        async with semaphore:
            return await run(request, ctx, fetch=fetch, judge=judge)

    with ctx.bind():
        return list(await asyncio.gather(*(one(r) for r in requests)))
