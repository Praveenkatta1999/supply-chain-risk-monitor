import asyncio
from datetime import date

import pytest

from scrm.agents import query_agent
from scrm.schemas import Event, QueryRequest
from scrm.telemetry import RunContext


class FakeTool:
    """Answers each query by matching the table name in the SQL."""

    allowed_project = "proj"
    allowed_dataset = "scrm"

    def __init__(self, tables: dict[str, list[dict]]):
        self.tables = tables
        self.bytes_processed = 0

    def run_query(self, sql, params=None):
        self.bytes_processed += 100
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


def gkg_row(gkg_id, url, themes="", tone=0.0, distance_km=25.0):
    return {
        "gkg_id": gkg_id,
        "event_date": date(2026, 9, 10),
        "url": url,
        "themes": themes,
        "organizations": "Port Authority,12;Port Authority,40",
        "tone": tone,
        "distance_km": distance_km,
    }


def coded_row(event_id, url, root="14", tone=-5.0, distance_km=10.0):
    return {
        "event_id": event_id,
        "event_date": date(2026, 9, 11),
        "actor1": "DOCKWORKER",
        "actor2": None,
        "event_root_code": root,
        "avg_tone": tone,
        "distance_km": distance_km,
        "source_url": url,
    }


def test_parse_gdelt_list_strips_offsets_and_dedupes():
    assert query_agent.parse_gdelt_list("MARITIME,10;WB_167_PORTS,20;MARITIME,30;") == [
        "MARITIME",
        "WB_167_PORTS",
    ]
    assert query_agent.parse_gdelt_list(None) == []


def test_component_scores_are_bounded():
    assert query_agent.theme_score(["WB_167_PORTS", "STRIKE", "MARITIME"]) == 1.0
    assert query_agent.theme_score(["EDUCATION"]) == 0.0
    assert query_agent.tone_score(-20) == 1.0
    assert query_agent.tone_score(3) == 0.0
    assert query_agent.distance_score(0, 50) == 1.0
    assert query_agent.distance_score(60, 50) == 0.0


def test_port_themes_negative_tone_and_proximity_rank_higher():
    site = query_agent.Site.model_validate(SITE_ROW)
    strike = query_agent.event_from_gkg(
        gkg_row("a", "https://x.example/strike", "STRIKE,1;WB_167_PORTS,2", -6, 5), site
    )
    school = query_agent.event_from_gkg(
        gkg_row("b", "https://x.example/school", "EDUCATION,1", -1, 45), site
    )
    assert strike.relevance > school.relevance


def make_event(event_id: str, url: str, relevance: float) -> Event:
    return Event(
        event_id=event_id,
        source_table="gkg_near_sites",
        site_id="P05",
        event_date=date(2026, 9, 10),
        source_url=url,
        relevance=relevance,
    )


def test_rank_and_dedupe_keeps_best_row_per_url_and_limits():
    events = [
        make_event("low", "https://x.example/same", 0.2),
        make_event("high", "https://x.example/same", 0.9),
        make_event("other", "https://x.example/other", 0.5),
        make_event("third", "https://x.example/third", 0.1),
    ]
    top = query_agent.rank_and_dedupe(events, limit=2)
    assert [e.event_id for e in top] == ["high", "other"]


def test_run_merges_both_tables_and_skips_bad_urls(date_range):
    tool = FakeTool(
        {
            "sites": [SITE_ROW],
            "gkg_near_sites": [
                gkg_row("g1", "https://x.example/port", "WB_167_PORTS,1", -5, 5),
                gkg_row("g2", "not a url"),
            ],
            "events_near_sites": [coded_row(7, "https://x.example/protest")],
        }
    )
    request = QueryRequest(site_ids=["P05"], date_range=date_range, limit=20)
    result = asyncio.run(query_agent.run(request, RunContext.new(), tool=tool))
    assert [e.event_id for e in result.events] == ["g1", "7"]
    assert {e.source_table for e in result.events} == {"gkg_near_sites", "events_near_sites"}
    assert result.bytes_processed == 300
    assert len(result.sql) == 3


def test_unknown_site_is_an_error():
    with pytest.raises(ValueError, match="unknown site_ids"):
        query_agent.load_sites(FakeTool({"sites": []}), ["NOPE"])
