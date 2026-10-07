"""Query agent: pulls candidate disruptions near a site from BigQuery and ranks them.

Deterministic, not LLM-driven: the SQL is fixed and parameterised, so it is auditable,
cheap and cannot be steered into an unsafe query. All SQL runs through BigQueryTool,
which dry-runs first and refuses anything over the byte limit or outside ``scrm``.

Steps for each site:
1. Two retrieval paths:
   - location: every article (gkg_near_sites) and coded event (events_near_sites)
     geocoded inside the site's radius in the window;
   - entity: articles that name the site's company or an alias (entity_resolver.py) but
     were geocoded near a different monitored site.
2. Score each one on themes, site match, tone and distance (ranking.py). An entity
   match sets the site signal to 1.0.
3. Keep the best row per URL, then group rows about the same story (clustering.py).
4. Each cluster is represented by its best-scoring article; the others become
   ``supporting_urls``. The number of independent write-ups in the cluster (syndicated
   copies count once) feeds the ranking.
5. Return the top N clusters.

Input:  QueryRequest
Output: QueryResult
"""

import asyncio
from collections.abc import Callable
from dataclasses import dataclass
from datetime import date
from typing import Any, Literal
from urllib.parse import urlsplit

from google.cloud import bigquery
from pydantic import ValidationError

from scrm.agents import clustering, entity_resolver, ranking
from scrm.config import get_settings
from scrm.schemas import Event, QueryRequest, QueryResult, Site
from scrm.telemetry import RunContext, get_logger
from scrm.tools.bigquery_tool import BigQueryTool

NAME = "query_agent"
INPUT_SCHEMA = QueryRequest
OUTPUT_SCHEMA = QueryResult

MAX_SUPPORTING_URLS = 10

log = get_logger(__name__)

GKG_SQL = """
SELECT gkg_id, DATE(published_at) AS event_date, source, url, themes, organizations, tone,
  place, distance_km
FROM `{table}`
WHERE site_id = @site_id
  AND published_at >= TIMESTAMP(@start_date)
  AND published_at < TIMESTAMP(DATE_ADD(@end_date, INTERVAL 1 DAY))
  AND url IS NOT NULL
"""

# Second retrieval path: articles naming the site's company, geocoded near other sites.
ENTITY_SQL = """
SELECT gkg_id, DATE(published_at) AS event_date, source, url, themes, organizations, tone,
  place, distance_km
FROM `{table}`
WHERE site_id != @site_id
  AND published_at >= TIMESTAMP(@start_date)
  AND published_at < TIMESTAMP(DATE_ADD(@end_date, INTERVAL 1 DAY))
  AND url IS NOT NULL
  AND REGEXP_CONTAINS(LOWER(IFNULL(organizations, '')), @org_pattern)
ORDER BY published_at DESC
LIMIT {limit}
"""
ENTITY_ROW_LIMIT = 2000

EVENTS_SQL = """
SELECT event_id, event_date, actor1, actor2, event_root_code, avg_tone, place, distance_km,
  source_url
FROM `{table}`
WHERE site_id = @site_id
  AND event_date BETWEEN @start_date AND @end_date
  AND source_url IS NOT NULL
"""

SITE_SQL = """
SELECT site_id, site_name, company, site_type, component, city, country, lat, lon, radius_km
FROM `{table}`
WHERE @all_sites OR site_id IN UNNEST(@site_ids)
ORDER BY site_id
"""


@dataclass(frozen=True)
class Candidate:
    """An event plus the ranking signals it was scored on."""

    event: Event
    signals: ranking.Signals


def parse_gdelt_list(raw: str | None) -> list[str]:
    """Parse a GKG 'NAME,offset;NAME,offset;' field into unique names, in order."""
    names = (part.split(",", 1)[0].strip() for part in (raw or "").split(";"))
    return list(dict.fromkeys(name for name in names if name))


def _domain(url: str) -> str | None:
    host = urlsplit(url).hostname
    return host.removeprefix("www.") if host else None


def _signals(
    site: Site,
    theme: float,
    url: str,
    orgs: list[str],
    tone: float | None,
    place: str | None,
    distance_km: float | None,
) -> ranking.Signals:
    return ranking.Signals(
        themes=theme,
        site=ranking.site_match_score(site, urlsplit(url).path, orgs, place),
        tone=ranking.tone_score(tone),
        distance=ranking.distance_score(distance_km, site.radius_km),
    )


def candidate_from_gkg(
    row: dict[str, Any], site: Site, retrieval: Literal["location", "entity"] = "location"
) -> Candidate:
    themes = parse_gdelt_list(row["themes"])
    orgs = parse_gdelt_list(row["organizations"])
    # For entity-path rows the stored distance is to another site, so it says nothing here.
    distance = row["distance_km"] if retrieval == "location" else None
    signals = _signals(
        site, ranking.theme_score(themes), row["url"], orgs, row["tone"], row["place"], distance
    )
    event = Event(
        event_id=row["gkg_id"],
        source_table="gkg_near_sites",
        site_id=site.site_id,
        event_date=row["event_date"],
        title=clustering.slug_title(row["url"]),
        source=row.get("source") or _domain(row["url"]),
        organizations=orgs,
        themes=themes,
        tone=row["tone"],
        place=row["place"],
        distance_km=distance,
        source_url=row["url"],
        relevance=signals.relevance(),
        matched_entities=[m.raw_name for m in entity_resolver.match(orgs, site)],
        retrieval=retrieval,
    )
    return Candidate(event, signals)


def candidate_from_entity_row(row: dict[str, Any], site: Site) -> Candidate:
    return candidate_from_gkg(row, site, retrieval="entity")


def candidate_from_coded_event(row: dict[str, Any], site: Site) -> Candidate:
    root = row["event_root_code"] or ""
    theme = ranking.CAMEO_ROOT_RELEVANCE.get(root, 0.0)
    signals = _signals(
        site, theme, row["source_url"], [], row["avg_tone"], row["place"], row["distance_km"]
    )
    event = Event(
        event_id=str(row["event_id"]),
        source_table="events_near_sites",
        site_id=site.site_id,
        event_date=row["event_date"],
        title=clustering.slug_title(row["source_url"]),
        source=_domain(row["source_url"]),
        actors=[a for a in (row["actor1"], row["actor2"]) if a],
        themes=[f"CAMEO_ROOT_{root}"] if root else [],
        tone=row["avg_tone"],
        place=row["place"],
        distance_km=row["distance_km"],
        source_url=row["source_url"],
        relevance=signals.relevance(),
    )
    return Candidate(event, signals)


def best_per_url(candidates: list[Candidate]) -> list[Candidate]:
    """Keep the highest-relevance candidate for each URL, ordered by relevance (desc)."""
    ordered = sorted(candidates, key=lambda c: c.event.relevance or 0.0, reverse=True)
    best: dict[str, Candidate] = {}
    for candidate in ordered:
        best.setdefault(str(candidate.event.source_url), candidate)
    return list(best.values())


def cluster_and_rank(candidates: list[Candidate], limit: int) -> list[Event]:
    """Group same-story candidates, represent each group by its best member, rank groups."""
    unique = best_per_url(candidates)
    items = [
        clustering.Item(str(c.event.source_url), c.event.place, c.event.event_date) for c in unique
    ]
    representatives = []
    for members in clustering.cluster(items):
        group = [unique[i] for i in members]  # already in relevance order
        best, others = group[0], group[1:]
        reports = clustering.independent_reports([str(c.event.source_url) for c in group])
        representatives.append(
            best.event.model_copy(
                update={
                    "relevance": best.signals.relevance(cluster_size=reports),
                    "cluster_size": len(group),
                    "supporting_urls": [o.event.source_url for o in others][:MAX_SUPPORTING_URLS],
                }
            )
        )
    representatives.sort(key=lambda e: e.relevance or 0.0, reverse=True)
    return representatives[:limit]


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


def load_sites(tool: BigQueryTool, site_ids: list[str] | None = None) -> list[Site]:
    """Fetch sites from scrm.sites in the order requested; all sites if ``site_ids`` is empty."""
    params = [
        bigquery.ScalarQueryParameter("all_sites", "BOOL", not site_ids),
        bigquery.ArrayQueryParameter("site_ids", "STRING", site_ids or []),
    ]
    rows = tool.run_query(SITE_SQL.format(table=_table(tool, "sites")), params)
    by_id = {row["site_id"]: Site.model_validate(row) for row in rows}
    if not site_ids:
        return list(by_id.values())
    missing = [s for s in site_ids if s not in by_id]
    if missing:
        raise ValueError(f"unknown site_ids: {missing}")
    return [by_id[s] for s in site_ids]


RowParser = Callable[[dict[str, Any], Site], Candidate]


def _parse_rows(rows: list[dict[str, Any]], site: Site, parse: RowParser) -> list[Candidate]:
    candidates = []
    for row in rows:
        try:
            candidates.append(parse(row, site))
        except ValidationError as exc:
            # Mostly malformed URLs in GDELT; skip the row rather than fail the run.
            log.warning("query.row_skipped", extra={"site_id": site.site_id, "error": str(exc)})
    return candidates


async def _candidates_for_site(
    tool: BigQueryTool, site: Site, request: QueryRequest, sql_log: list[str]
) -> list[Event]:
    params = _date_params(site.site_id, request.date_range.start, request.date_range.end)
    gkg_sql = GKG_SQL.format(table=_table(tool, "gkg_near_sites"))
    events_sql = EVENTS_SQL.format(table=_table(tool, "events_near_sites"))
    queries = [
        asyncio.to_thread(tool.run_query, gkg_sql, params),
        asyncio.to_thread(tool.run_query, events_sql, params),
    ]
    sql_log += [gkg_sql, events_sql]
    pattern = entity_resolver.bigquery_pattern(site)
    if pattern:
        entity_sql = ENTITY_SQL.format(table=_table(tool, "gkg_near_sites"), limit=ENTITY_ROW_LIMIT)
        entity_params = [*params, bigquery.ScalarQueryParameter("org_pattern", "STRING", pattern)]
        queries.append(asyncio.to_thread(tool.run_query, entity_sql, entity_params))
        sql_log.append(entity_sql)
    # BigQuery calls block, so run them in threads to let other sites proceed meanwhile.
    gkg_rows, event_rows, *rest = await asyncio.gather(*queries)
    entity_rows = rest[0] if rest else []
    candidates = (
        _parse_rows(gkg_rows, site, candidate_from_gkg)
        + _parse_rows(event_rows, site, candidate_from_coded_event)
        + _parse_rows(entity_rows, site, candidate_from_entity_row)
    )
    top = cluster_and_rank(candidates, request.limit)
    log.info(
        "query.site_done",
        extra={
            "site_id": site.site_id,
            "gkg_rows": len(gkg_rows),
            "event_rows": len(event_rows),
            "entity_rows": len(entity_rows),
            "returned": len(top),
            "returned_via_entity": sum(e.retrieval == "entity" for e in top),
            "articles_covered": sum(e.cluster_size for e in top),
        },
    )
    return top


async def run(
    request: QueryRequest,
    ctx: RunContext,
    *,
    tool: BigQueryTool | None = None,
    sites: list[Site] | None = None,
) -> QueryResult:
    """Return the top-ranked story clusters for each requested site.

    Pass ``sites`` if they are already loaded, to skip the scrm.sites lookup.
    """
    # A fork shares the client but counts bytes for this call only, even when several
    # sites run concurrently on the same tool.
    tool = (tool or BigQueryTool.from_settings(get_settings())).fork()
    with ctx.bind():
        log.info("query.start", extra={"site_ids": request.site_ids, "limit": request.limit})
        sql_log: list[str] = []
        if sites is None:
            sites = await asyncio.to_thread(load_sites, tool, request.site_ids)
            sql_log.append(SITE_SQL.format(table=_table(tool, "sites")))
        events = [
            event
            for site in sites
            for event in await _candidates_for_site(tool, site, request, sql_log)
        ]
        log.info(
            "query.done", extra={"events": len(events), "bytes_processed": tool.bytes_processed}
        )
        return QueryResult(events=events, sql=sql_log, bytes_processed=tool.bytes_processed)
