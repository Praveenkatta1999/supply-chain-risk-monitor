from datetime import UTC, date, datetime

import pytest
from pydantic import ValidationError

from scrm.schemas import BriefItem, DateRange, RiskBrief, RiskScore


def test_date_range_rejects_reversed():
    with pytest.raises(ValidationError):
        DateRange(start=date(2026, 9, 2), end=date(2026, 9, 1))


def test_risk_score_bounds():
    with pytest.raises(ValidationError):
        RiskScore(
            event_id="e",
            site_id="s",
            severity=6,
            impact=1,
            reason="x",
            source_url="https://example.com",
        )


def test_brief_item_requires_source_url():
    with pytest.raises(ValidationError):
        BriefItem(site_id="s", claim="Strike", severity=3, impact=3, source_urls=[])


def test_brief_markdown_must_link_every_source(date_range):
    item = BriefItem(
        site_id="SUP-01",
        claim="Strike",
        severity=3,
        impact=4,
        source_urls=["https://example.com/a"],
    )
    common = dict(
        run_id="r1",
        generated_at=datetime.now(UTC),
        date_range=date_range,
        site_ids=["SUP-01"],
        items=[item],
    )
    RiskBrief(**common, markdown="- Strike ([source](https://example.com/a))")
    with pytest.raises(ValidationError, match="missing source links"):
        RiskBrief(**common, markdown="- Strike")


def test_yes_verdict_requires_a_quote(event):
    from scrm.schemas import Verdict, VerifiedEvent

    common = dict(event=event, reason="Strike at the port.", evidence_url=event.source_url)
    with pytest.raises(ValidationError, match="quote"):
        VerifiedEvent(**common, verdict=Verdict.YES)
    VerifiedEvent(**common, verdict=Verdict.UNVERIFIABLE)


def test_brief_item_risk_is_severity_times_impact():
    item = BriefItem(site_id="s", claim="c", severity=3, impact=4, source_urls=["https://a.x/"])
    assert item.risk == 12
    unscored = BriefItem(site_id="s", claim="c", source_urls=["https://a.x/"])
    assert unscored.risk == 0


def test_brief_markdown_must_link_supporting_sources(date_range):
    item = BriefItem(
        site_id="s",
        claim="c",
        source_urls=["https://example.com/a"],
        supporting_urls=["https://example.com/b"],
    )
    common = dict(
        run_id="r1",
        generated_at=datetime.now(UTC),
        date_range=date_range,
        site_ids=["s"],
        items=[item],
    )
    with pytest.raises(ValidationError, match=r"example.com/b"):
        RiskBrief(**common, markdown="- c ([source](https://example.com/a))")


def test_scope_must_agree_with_verdict(event):
    from scrm.schemas import Scope, Verdict, VerifiedEvent

    common = dict(event=event, reason="r", evidence_url=event.source_url)
    VerifiedEvent(**common, verdict=Verdict.YES, scope=Scope.INDIRECT, quote="q")
    VerifiedEvent(**common, verdict=Verdict.NO, scope=Scope.NOT_RELEVANT)
    VerifiedEvent(**common, verdict=Verdict.YES, quote="q")  # legacy runs have no scope
    with pytest.raises(ValidationError, match="contradicts"):
        VerifiedEvent(**common, verdict=Verdict.NO, scope=Scope.DIRECT)
