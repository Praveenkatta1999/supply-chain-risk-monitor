import asyncio
from datetime import UTC, datetime

from scrm.agents import verifier
from scrm.config import Settings
from scrm.schemas import (
    Article,
    FetchFailure,
    FetchFailureReason,
    Verdict,
    VerificationRequest,
    VerifierJudgement,
)
from scrm.telemetry import RunContext

TEXT = "Dockworkers walked out on Monday.\nThe strike has halted all container handling."


def article(url) -> Article:
    return Article(url=url, title="Strike", text=TEXT, fetched_at=datetime.now(UTC))


async def fetch_ok(url):
    return article(url)


async def fetch_dead(url):
    return FetchFailure(url=url, reason=FetchFailureReason.DEAD_LINK, detail="HTTP 404")


def judge_returning(verdict: Verdict, quote: str | None):
    calls = []

    async def judge(payload):
        calls.append(payload)
        return VerifierJudgement(verdict=verdict, reason="Because.", quote=quote)

    judge.calls = calls
    return judge


def verify(request, fetch, judge):
    return asyncio.run(verifier.run(request, RunContext.new(), fetch=fetch, judge=judge))


def test_verifier_builds_with_structured_output():
    agent = verifier.build_agent(Settings())
    assert agent.name == "verifier"
    assert agent.output_schema is VerifierJudgement


def test_quote_matching_ignores_whitespace_case_and_curly_quotes():
    assert verifier.quote_in_text("the  STRIKE has\nhalted", TEXT)
    curly = "it" + chr(0x2019) + "s closed"
    assert verifier.quote_in_text(curly, "It's closed today")
    assert not verifier.quote_in_text("the port is open", TEXT)
    assert not verifier.quote_in_text("  ", TEXT)


def test_dead_link_is_unverifiable_without_calling_the_model(event, site):
    judge = judge_returning(Verdict.YES, "halted")
    result = verify(VerificationRequest(event=event, site=site), fetch_dead, judge)
    assert result.verdict is Verdict.UNVERIFIABLE
    assert "dead_link" in result.reason
    assert judge.calls == []


def test_yes_with_real_quote_is_kept(event, site):
    judge = judge_returning(Verdict.YES, "The strike has halted all container handling.")
    result = verify(VerificationRequest(event=event, site=site), fetch_ok, judge)
    assert result.verdict is Verdict.YES
    assert result.quote == "The strike has halted all container handling."
    assert result.evidence_url == event.source_url
    assert judge.calls[0].site == site


def test_yes_with_invented_quote_is_downgraded(event, site):
    judge = judge_returning(Verdict.YES, "The port is closed for a month.")
    result = verify(VerificationRequest(event=event, site=site), fetch_ok, judge)
    assert result.verdict is Verdict.UNVERIFIABLE
    assert "quote" in result.reason


def test_no_with_invented_quote_keeps_verdict_but_drops_quote(event, site):
    judge = judge_returning(Verdict.NO, "Not in the article.")
    result = verify(VerificationRequest(event=event, site=site), fetch_ok, judge)
    assert result.verdict is Verdict.NO
    assert result.quote is None


def test_model_failure_is_unverifiable(event, site):
    async def broken(payload):
        raise RuntimeError("quota")

    result = verify(VerificationRequest(event=event, site=site), fetch_ok, broken)
    assert result.verdict is Verdict.UNVERIFIABLE


def test_run_many_preserves_order(event, site):
    other = event.model_copy(update={"event_id": "gkg-456"})
    requests = [VerificationRequest(event=e, site=site) for e in (event, other)]
    results = asyncio.run(
        verifier.run_many(
            requests, RunContext.new(), fetch=fetch_ok, judge=judge_returning(Verdict.NO, None)
        )
    )
    assert [r.event.event_id for r in results] == ["gkg-123", "gkg-456"]
