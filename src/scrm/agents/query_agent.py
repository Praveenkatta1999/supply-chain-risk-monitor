"""Query agent: pulls candidate disruptions near a site from BigQuery and ranks them.

Deterministic, not LLM-driven: the SQL is fixed and parameterised, so it is auditable,
cheap and cannot be steered into an unsafe query. All SQL runs through BigQueryTool,
which dry-runs first and refuses anything over the byte limit or outside ``scrm``.

Ranking combines three signals, each scaled to 0..1:
    themes    how strongly the article's GDELT themes relate to ports and shipping
    tone      how negative the coverage is (GDELT tone, roughly -10..+10)
    distance  how close the event is to the site, relative to the site's radius
Candidates are deduplicated by URL (keeping the best-ranked row) and the top N returned.

Input:  QueryRequest
Output: QueryResult
"""

from collections.abc import Callable
from datetime import date
from typing import Any

from google.cloud import bigquery
from pydantic import ValidationError

from scrm.config import get_settings
from scrm.schemas import Event, QueryRequest, QueryResult, Site
from scrm.telemetry import RunContext, get_logger
from scrm.tools.bigquery_tool import BigQueryTool

NAME = "query_agent"
INPUT_SCHEMA = QueryRequest
OUTPUT_SCHEMA = QueryResult

log = get_logger(__name__)

WEIGHTS = {"themes": 0.5, "tone": 0.3, "distance": 0.2}

# GDELT GKG themes that signal a port or shipping disruption. Strong themes are about
# ports and ships directly; weaker ones are disruptive but not shipping-specific.
PORT_THEMES: dict[str, float] = {
    "WB_167_PORTS": 1.0,
    "MARITIME_INCIDENT": 1.0,
    "MARITIME_INCIDENT_SELF_IDENTIFIED": 1.0,
    "MANMADE_DISASTER_AGROUND": 1.0,
    "BLOCKADE": 1.0,
    "STRIKE": 1.0,
    "MARITIME": 0.6,
    "WB_1803_TRANSPORT_INFRASTRUCTURE": 0.6,
    "WB_135_TRANSPORT": 0.4,
    "TRAFFIC": 0.4,
    "WB_698_TRADE": 0.3,
    "ECON_TRADE_DISPUTE": 0.5,
    "SANCTIONS": 0.5,
    "CYBER_ATTACK": 0.6,
    "NATURAL_DISASTER_FLOOD": 0.5,
    "NATURAL_DISASTER_DROUGHTS": 0.4,
    "NATURAL_DISASTER_STORM": 0.5,
    "PROTEST": 0.4,
    "CLOSURE": 0.4,
}
THEME_SATURATION = 2.0  # summed theme weight at which the theme score reaches 1.0

# CAMEO root codes in scrm.events_near_sites carry no themes; give them a fixed,
# modest relevance (protests include strikes and blockades, the rest is violence).
CAMEO_ROOT_RELEVANCE: dict[str, float] = {"14": 0.4, "17": 0.2, "18": 0.2, "19": 0.2, "20": 0.2}

GKG_SQL = """
SELECT gkg_id, DATE(published_at) AS event_date, url, themes, organizations, tone, distance_km
FROM `{table}`
WHERE site_id = @site_id
  AND published_at >= TIMESTAMP(@start_date)
  AND published_at < TIMESTAMP(DATE_ADD(@end_date, INTERVAL 1 DAY))
  AND url IS NOT NULL
"""

EVENTS_SQL = """
SELECT event_id, event_date, actor1, actor2, event_root_code, avg_tone, distance_km, source_url
FROM `{table}`
WHERE site_id = @site_id
  AND event_date BETWEEN @start_date AND @end_date
  AND source_url IS NOT NULL
"""

SITE_SQL = """
SELECT site_id, site_name, company, site_type, component, city, country, lat, lon, radius_km
FROM `{table}`
WHERE site_id IN UNNEST(@site_ids)
"""


# ---------------------------------------------------------------------------
# Scoring (pure functions)
# ---------------------------------------------------------------------------


def parse_gdelt_list(raw: str | None) -> list[str]:
    """Parse a GKG 'NAME,offset;NAME,offset;' field into unique names, in order."""
    names = (part.split(",", 1)[0].strip() for part in (raw or "").split(";"))
    return list(dict.fromkeys(name for name in names if name))


def theme_score(themes: list[str]) -> float:
    total = sum(PORT_THEMES.get(theme, 0.0) for theme in set(themes))
    return min(1.0, total / THEME_SATURATION)


def tone_score(tone: float | None) -> float:
    return 0.0 if tone is None else min(1.0, max(0.0, -tone / 10))


def distance_score(distance_km: float | None, radius_km: float) -> float:
    return 0.0 if distance_km is None else min(1.0, max(0.0, 1 - distance_km / radius_km))


def relevance(theme: float, tone: float, distance: float) -> float:
    score = WEIGHTS["themes"] * theme + WEIGHTS["tone"] * tone + WEIGHTS["distance"] * distance
    return round(score, 4)


# ---------------------------------------------------------------------------
# Row -> Event
# ---------------------------------------------------------------------------


def event_from_gkg(row: dict[str, Any], site: Site) -> Event:
    themes = parse_gdelt_list(row["themes"])
    return Event(
        event_id=row["gkg_id"],
        source_table="gkg_near_sites",
        site_id=site.site_id,
        event_date=row["event_date"],
        organizations=parse_gdelt_list(row["organizations"]),
        themes=themes,
        tone=row["tone"],
        distance_km=row["distance_km"],
        source_url=row["url"],
        relevance=relevance(
            theme_score(themes),
            tone_score(row["tone"]),
            distance_score(row["distance_km"], site.radius_km),
        ),
    )


def event_from_coded_event(row: dict[str, Any], site: Site) -> Event:
    root = row["event_root_code"] or ""
    return Event(
        event_id=str(row["event_id"]),
        source_table="events_near_sites",
        site_id=site.site_id,
        event_date=row["event_date"],
        actors=[a for a in (row["actor1"], row["actor2"]) if a],
        themes=[f"CAMEO_ROOT_{root}"] if root else [],
        tone=row["avg_tone"],
        distance_km=row["distance_km"],
        source_url=row["source_url"],
        relevance=relevance(
            CAMEO_ROOT_RELEVANCE.get(root, 0.0),
            tone_score(row["avg_tone"]),
            distance_score(row["distance_km"], site.radius_km),
        ),
    )


def rank_and_dedupe(events: list[Event], limit: int) -> list[Event]:
    """Sort by relevance (desc), keep the best event per URL, return the top ``limit``."""
    ordered = sorted(events, key=lambda e: e.relevance or 0.0, reverse=True)
    best_per_url: dict[str, Event] = {}
    for event in ordered:
        best_per_url.setdefault(str(event.source_url), event)
    return list(best_per_url.values())[:limit]


# ---------------------------------------------------------------------------
# BigQuery
# ---------------------------------------------------------------------------


def _table(tool: BigQueryTool, name: str) -> str:
    return f"{tool.allowed_project}.{tool.allowed_dataset}.{name}"


def _date_params(site_id: str, start: date, end: date) -> list[bigquery.ScalarQueryParameter]:
    return [
        bigquery.ScalarQueryParameter("site_id", "STRING", site_id),
        bigquery.ScalarQueryParameter("start_date", "DATE", start),
        bigquery.ScalarQueryParameter("end_date", "DATE", end),
    ]


def load_sites(tool: BigQueryTool, site_ids: list[str]) -> list[Site]:
    """Fetch site rows from scrm.sites, in the order requested."""
    sql = SITE_SQL.format(table=_table(tool, "sites"))
    rows = tool.run_query(sql, [bigquery.ArrayQueryParameter("site_ids", "STRING", site_ids)])
    by_id = {row["site_id"]: Site.model_validate(row) for row in rows}
    missing = [s for s in site_ids if s not in by_id]
    if missing:
        raise ValueError(f"unknown site_ids: {missing}")
    return [by_id[s] for s in site_ids]


RowParser = Callable[[dict[str, Any], Site], Event]


def _to_events(rows: list[dict[str, Any]], site: Site, build: RowParser) -> list[Event]:
    events = []
    for row in rows:
        try:
            events.append(build(row, site))
        except ValidationError as exc:
            # Mostly malformed URLs in GDELT; skip the row rather than fail the run.
            log.warning("query.row_skipped", extra={"site_id": site.site_id, "error": str(exc)})
    return events


def _candidates_for_site(
    tool: BigQueryTool, site: Site, request: QueryRequest, sql_log: list[str]
) -> list[Event]:
    params = _date_params(site.site_id, request.date_range.start, request.date_range.end)
    gkg_sql = GKG_SQL.format(table=_table(tool, "gkg_near_sites"))
    events_sql = EVENTS_SQL.format(table=_table(tool, "events_near_sites"))
    gkg_rows = tool.run_query(gkg_sql, params)
    event_rows = tool.run_query(events_sql, params)
    sql_log += [gkg_sql, events_sql]
    candidates = _to_events(gkg_rows, site, event_from_gkg) + _to_events(
        event_rows, site, event_from_coded_event
    )
    top = rank_and_dedupe(candidates, request.limit)
    log.info(
        "query.site_done",
        extra={
            "site_id": site.site_id,
            "gkg_rows": len(gkg_rows),
            "event_rows": len(event_rows),
            "returned": len(top),
        },
    )
    return top


async def run(
    request: QueryRequest, ctx: RunContext, *, tool: BigQueryTool | None = None
) -> QueryResult:
    """Return the top-ranked candidate events for each requested site."""
    tool = tool or BigQueryTool.from_settings(get_settings())
    with ctx.bind():
        log.info("query.start", extra={"site_ids": request.site_ids, "limit": request.limit})
        bytes_before = tool.bytes_processed
        sites = load_sites(tool, request.site_ids)
        sql_log: list[str] = [SITE_SQL.format(table=_table(tool, "sites"))]
        events = [
            event for site in sites for event in _candidates_for_site(tool, site, request, sql_log)
        ]
        bytes_processed = tool.bytes_processed - bytes_before
        log.info("query.done", extra={"events": len(events), "bytes_processed": bytes_processed})
        return QueryResult(events=events, sql=sql_log, bytes_processed=bytes_processed)
