import asyncio
import re

from scrm.agents import entity_resolver
from scrm.schemas import EntityResolutionRequest, Site, SiteType
from scrm.telemetry import RunContext


def make_site(site_id, company, site_name="Somewhere", site_type=SiteType.FACTORY) -> Site:
    return Site(
        site_id=site_id,
        site_name=site_name,
        company=company,
        site_type=site_type,
        country="X",
        lat=0,
        lon=0,
        radius_km=50,
    )


TSMC = make_site("S01", "TSMC")
FOXCONN = make_site("S02", "Foxconn (Hon Hai)")
ROTTERDAM = make_site("P05", None, "Port of Rotterdam", SiteType.PORT)


def test_aliases_include_company_and_configured_names():
    assert entity_resolver.aliases_for(TSMC) == ("tsmc", "taiwan semiconductor")
    # "Foxconn (Hon Hai)" holds two names and both must count.
    assert set(entity_resolver.aliases_for(FOXCONN)) == {"foxconn", "hon hai"}
    assert "havenbedrijf rotterdam" in entity_resolver.aliases_for(ROTTERDAM)


def test_match_is_whole_word_and_case_insensitive():
    names = [
        "Taiwan Semiconductor Manufacturing Co",
        "Tsmc",
        "Tsmc-Arizona",
        "Atsmcorp",  # not a whole-word match
        "Ministry Of Economic Affairs",
    ]
    matched = [m.raw_name for m in entity_resolver.match(names, TSMC)]
    assert matched == ["Taiwan Semiconductor Manufacturing Co", "Tsmc", "Tsmc-Arizona"]


def test_bigquery_pattern_matches_like_python():
    pattern = re.compile(entity_resolver.bigquery_pattern(TSMC))
    assert pattern.search("taiwan semiconductor manufacturing co,12;other,3")
    assert pattern.search("tsmc,5")
    assert not pattern.search("atsmcorp,5")


def test_site_without_company_or_aliases_has_no_pattern():
    assert entity_resolver.bigquery_pattern(make_site("X99", None)) is None


def test_run_returns_matches_for_the_site():
    request = EntityResolutionRequest(raw_names=["Hon Hai Precision", "Apple"], site=FOXCONN)
    result = asyncio.run(entity_resolver.run(request, RunContext.new()))
    assert [(m.raw_name, m.alias, m.site_id) for m in result.matches] == [
        ("Hon Hai Precision", "hon hai", "S02")
    ]
