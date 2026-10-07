import asyncio
from datetime import date

import pytest

from scrm.agents import query_agent
from scrm.schemas import QueryRequest, Site
from scrm.telemetry import RunContext


class FakeTool:
    """Answers each query by matching the table name in the SQL."""

    allowed_project = "proj"
    allowed_dataset = "scrm"

    def __init__(self, tables: dict[str, list[dict]]):
        self.tables = tables
        self.bytes_processed = 0
        self.statements: list[str] = []

    def fork(self):
        return self

    def run_query(self, sql, params=None):
        self.bytes_processed += 100
        self.statements.append(sql)
        if "@org_pattern" in sql:  # the entity retrieval path
            return self.tables.get("entity", [])
        for name, rows in self.tables.items():
            if f".scrm.{name}`" in sql:
                return rows
        raise AssertionError(f"unexpected SQL: {sql}")


SITE_ROW = {
    "site_id": "P05",
    "site_name": "Port of Rotterdam",
    "company": None,
    "site_type": "port",
    "component": None,
    "city": "Rotterdam",
    "country": "Netherlands",
    "lat": 51.95,
    "lon": 4.14,
    "radius_km": 50,
}
SITE = Site.model_validate(SITE_ROW)
KINDERDIJK = "Kinderdijk, Zuid-Holland, Netherlands"


def gkg_row(gkg_id, url, themes="", tone=0.0, distance_km=25.0, place="Rotterdam", day=10):
    return {
        "gkg_id": gkg_id,
        "event_date": date(2026, 9, day),
        "url": url,
        "themes": themes,
        "organizations": "Port Authority,12;Port Authority,40",
        "tone": tone,
        "place": place,
        "distance_km": distance_km,
        "source": "x.example",
    }


def coded_row(event_id, url, root="14", tone=-5.0, distance_km=10.0):
    return {
        "event_id": event_id,
        "event_date": date(2026, 9, 11),
        "actor1": "DOCKWORKER",
        "actor2": None,
        "event_root_code": root,
        "avg_tone": tone,
        "place": "Rotterdam",
        "distance_km": distance_km,
        "source_url": url,
    }


def test_parse_gdelt_list_strips_offsets_and_dedupes():
    assert query_agent.parse_gdelt_list("MARITIME,10;WB_167_PORTS,20;MARITIME,30;") == [
        "MARITIME",
        "WB_167_PORTS",
    ]
    assert query_agent.parse_gdelt_list(None) == []


def test_port_themes_negative_tone_and_proximity_rank_higher():
    strike = query_agent.candidate_from_gkg(
        gkg_row("a", "https://x.example/strike", "STRIKE,1;WB_167_PORTS,2", -6, 5), SITE
    )
    school = query_agent.candidate_from_gkg(
        gkg_row("b", "https://x.example/school", "EDUCATION,1", -1, 45), SITE
    )
    assert strike.event.relevance > school.event.relevance


def test_articles_naming_the_site_outrank_ones_only_inside_the_radius():
    named = query_agent.candidate_from_gkg(
        gkg_row("a", "https://x.example/staking-in-de-rotterdamse-haven", distance_km=30), SITE
    )
    nearby = query_agent.candidate_from_gkg(
        gkg_row("b", "https://x.example/drugs-arrest-in-russia", distance_km=5), SITE
    )
    assert named.signals.site == 1.0 and nearby.signals.site == 0.0
    assert named.event.relevance > nearby.event.relevance


def test_cluster_and_rank_keeps_one_representative_per_story():
    slug = "schepen-botsen-op-elkaar-bij-kinderdijk-flinke-schade"
    rows = [
        gkg_row("k1", f"https://a.example/{slug}", "MARITIME,1", -4, 30, KINDERDIJK),
        gkg_row("k2", f"https://b.example/{slug}-xq", "MARITIME,1", -3, 30, KINDERDIJK, day=11),
        gkg_row("k3", f"https://c.example/{slug}", "MARITIME,1", -2, 30, KINDERDIJK, day=12),
        gkg_row("dup", f"https://a.example/{slug}", "", 0, 49, KINDERDIJK),  # same URL as k1
        gkg_row("other", "https://d.example/unrelated-story-about-tulips"),
    ]
    events = query_agent.cluster_and_rank(
        [query_agent.candidate_from_gkg(r, SITE) for r in rows], limit=20
    )
    story = next(e for e in events if e.cluster_size == 3)
    assert story.event_id == "k1"  # best-scoring member represents the cluster
    assert [str(u) for u in story.supporting_urls] == [
        f"https://b.example/{slug}-xq",
        f"https://c.example/{slug}",
    ]
    assert len(events) == 2


def test_cluster_size_counts_toward_relevance():
    slug = "rhine-water-levels-fall-to-new-record-low"
    lone = query_agent.candidate_from_gkg(
        gkg_row("x", f"https://a.example/{slug}", "MARITIME,1"), SITE
    )
    copies = [
        query_agent.candidate_from_gkg(
            gkg_row(f"c{i}", f"https://p{i}.example/{slug}-{w}", "MARITIME,1"), SITE
        )
        for i, w in enumerate(["barges", "fuel", "shipping", "drought"])
    ]
    clustered = query_agent.cluster_and_rank(copies, limit=1)[0]
    assert clustered.cluster_size == 4
    assert clustered.relevance > lone.event.relevance


def test_limit_applies_to_clusters():
    rows = [gkg_row(f"g{i}", f"https://x.example/story-number-{w}") for i, w in enumerate("abc")]
    events = query_agent.cluster_and_rank(
        [query_agent.candidate_from_gkg(r, SITE) for r in rows], limit=2
    )
    assert len(events) == 2


def test_run_merges_both_tables_and_skips_bad_urls(date_range):
    tool = FakeTool(
        {
            "sites": [SITE_ROW],
            "gkg_near_sites": [
                gkg_row(
                    "g1", "https://x.example/port-of-rotterdam-closed", "WB_167_PORTS,1", -5, 5
                ),
                gkg_row("g2", "not a url"),
            ],
            "events_near_sites": [coded_row(7, "https://x.example/protest")],
        }
    )
    request = QueryRequest(site_ids=["P05"], date_range=date_range, limit=20)
    result = asyncio.run(query_agent.run(request, RunContext.new(), tool=tool))
    assert [e.event_id for e in result.events] == ["g1", "7"]
    assert {e.source_table for e in result.events} == {"gkg_near_sites", "events_near_sites"}
    assert result.bytes_processed == 400  # sites, gkg, events, entity path
    assert len(result.sql) == 4


def test_preloaded_sites_skip_the_sites_query(date_range):
    tool = FakeTool({"gkg_near_sites": [], "events_near_sites": []})
    request = QueryRequest(site_ids=["P05"], date_range=date_range)
    result = asyncio.run(query_agent.run(request, RunContext.new(), tool=tool, sites=[SITE]))
    assert result.events == [] and len(result.sql) == 3


def test_unknown_site_is_an_error():
    with pytest.raises(ValueError, match="unknown site_ids"):
        query_agent.load_sites(FakeTool({"sites": []}), ["NOPE"])


def test_empty_site_list_loads_all_sites():
    assert query_agent.load_sites(FakeTool({"sites": [SITE_ROW]}), []) == [SITE]


def test_entity_path_adds_company_articles_from_other_sites(date_range):
    tsmc = SITE.model_copy(
        update={"site_id": "S01", "site_name": "TSMC Hsinchu fabs", "company": "TSMC"}
    )
    row = gkg_row("e1", "https://x.example/tsmc-fab-outage-halts-output", distance_km=3)
    row["organizations"] = "Taiwan Semiconductor Manufacturing Co,10"
    tool = FakeTool({"gkg_near_sites": [], "events_near_sites": [], "entity": [row]})
    request = QueryRequest(site_ids=["S01"], date_range=date_range)
    result = asyncio.run(query_agent.run(request, RunContext.new(), tool=tool, sites=[tsmc]))
    (event,) = result.events
    assert event.retrieval == "entity"
    assert event.matched_entities == ["Taiwan Semiconductor Manufacturing Co"]
    assert event.distance_km is None  # distance was to another site
    assert event.title == "tsmc fab outage halts output"


def test_entity_match_is_a_strong_ranking_signal():
    tsmc = SITE.model_copy(
        update={"site_id": "S01", "site_name": "TSMC Hsinchu fabs", "company": "TSMC"}
    )
    named = gkg_row("a", "https://x.example/a-story", distance_km=40)
    named["organizations"] = "Tsmc,1"
    nearby = gkg_row("b", "https://x.example/b-story", distance_km=1)
    named_c = query_agent.candidate_from_gkg(named, tsmc)
    nearby_c = query_agent.candidate_from_gkg(nearby, tsmc)
    assert named_c.signals.site == 1.0 and nearby_c.signals.site == 0.0
    assert named_c.event.relevance > nearby_c.event.relevance


def test_site_without_aliases_skips_the_entity_query(date_range):
    unnamed = SITE.model_copy(update={"site_id": "X99", "site_name": "Somewhere", "company": None})
    tool = FakeTool({"gkg_near_sites": [], "events_near_sites": []})
    request = QueryRequest(site_ids=["X99"], date_range=date_range)
    asyncio.run(query_agent.run(request, RunContext.new(), tool=tool, sites=[unnamed]))
    assert not any("@org_pattern" in sql for sql in tool.statements)
