import asyncio
from datetime import UTC, datetime

from scrm.agents import verifier
from scrm.config import Settings
from scrm.schemas import (
    Article,
    FetchFailure,
    FetchFailureReason,
    Scope,
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


def judge_returning(assessment: str, quote: str | None):
    calls = []

    async def judge(payload):
        calls.append(payload)
        return VerifierJudgement(assessment=assessment, reason="Because.", quote=quote)

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
    judge = judge_returning("direct", "halted")
    result = verify(VerificationRequest(event=event, site=site), fetch_dead, judge)
    assert result.verdict is Verdict.UNVERIFIABLE
    assert "dead_link" in result.reason
    assert judge.calls == []


def test_yes_with_real_quote_is_kept(event, site):
    judge = judge_returning("direct", "The strike has halted all container handling.")
    result = verify(VerificationRequest(event=event, site=site), fetch_ok, judge)
    assert result.verdict is Verdict.YES
    assert result.quote == "The strike has halted all container handling."
    assert result.evidence_url == event.source_url
    assert judge.calls[0].site == site


def test_yes_with_invented_quote_is_downgraded(event, site):
    judge = judge_returning("direct", "The port is closed for a month.")
    result = verify(VerificationRequest(event=event, site=site), fetch_ok, judge)
    assert result.verdict is Verdict.UNVERIFIABLE
    assert "quote" in result.reason


def test_no_with_invented_quote_keeps_verdict_but_drops_quote(event, site):
    judge = judge_returning("not_relevant", "Not in the article.")
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
            requests, RunContext.new(), fetch=fetch_ok, judge=judge_returning("not_relevant", None)
        )
    )
    assert [r.event.event_id for r in results] == ["gkg-123", "gkg-456"]


def test_rubric_maps_to_verdict_and_scope(event, site):
    request = VerificationRequest(event=event, site=site)
    quote = "The strike has halted all container handling."
    direct = verify(request, fetch_ok, judge_returning("direct", quote))
    indirect = verify(request, fetch_ok, judge_returning("indirect", quote))
    unrelated = verify(request, fetch_ok, judge_returning("not_relevant", None))
    unreadable = verify(request, fetch_ok, judge_returning("unverifiable", None))
    assert (direct.verdict, direct.scope) == (Verdict.YES, Scope.DIRECT)
    assert (indirect.verdict, indirect.scope) == (Verdict.YES, Scope.INDIRECT)
    assert (unrelated.verdict, unrelated.scope) == (Verdict.NO, Scope.NOT_RELEVANT)
    assert (unreadable.verdict, unreadable.scope) == (Verdict.UNVERIFIABLE, None)


def test_indirect_with_invented_quote_is_downgraded(event, site):
    judge = judge_returning("indirect", "The river is closed.")
    result = verify(VerificationRequest(event=event, site=site), fetch_ok, judge)
    assert result.verdict is Verdict.UNVERIFIABLE and result.scope is None
