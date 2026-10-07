"""Entity resolver: matches GDELT organisation names to a site's company and its aliases.

Deterministic, not LLM-driven: company names and their common aliases are a short,
reviewable table (``SITE_ALIASES``), and matching is whole-word and case-insensitive, so
"Taiwan Semiconductor Manufacturing Co" and "Tsmc" both resolve to S01. Ports and
chokepoints match on their operators ("Havenbedrijf Rotterdam", "Suez Canal Authority").

Matches are used twice by query_agent:
1. As a second retrieval path: articles that name the company but were geocoded near a
   different monitored site (the scrm tables only hold articles near some site).
2. As the strongest ranking signal: a match sets the site signal to 1.0.

Input:  EntityResolutionRequest
Output: EntityResolutionResult
"""

import re

from scrm.schemas import EntityMatch, EntityResolutionRequest, EntityResolutionResult, Site
from scrm.telemetry import RunContext, get_logger

NAME = "entity_resolver"
INPUT_SCHEMA = EntityResolutionRequest
OUTPUT_SCHEMA = EntityResolutionResult

log = get_logger(__name__)

# Common names for each site's company or operator, beyond scrm.sites.company. Keep them
# specific: "Samsung" alone would match Samsung Electronics, not the Samsung SDI plant.
SITE_ALIASES: dict[str, tuple[str, ...]] = {
    "S01": ("TSMC", "Taiwan Semiconductor"),
    "S02": ("Foxconn", "Hon Hai"),
    "S03": ("Quanta Computer",),
    "S04": ("BOE Technology", "Beijing Oriental Electronics"),
    "S05": ("Samsung SDI",),
    "S06": ("LG Display",),
    "S07": ("SK Hynix", "Hynix"),
    "S08": ("Murata Manufacturing", "Murata"),
    "S09": ("Intel",),
    "S10": ("Compal Electronics", "Compal"),
    "S11": ("Infineon",),
    "S12": ("Western Digital",),
    "P01": ("Port of Kaohsiung", "Kaohsiung Port", "Taiwan International Ports"),
    "P02": ("Port of Shanghai", "Shanghai International Port", "SIPG"),
    "P03": ("Port of Busan", "Busan Port Authority"),
    "P04": ("Port of Singapore", "PSA International", "Maritime and Port Authority of Singapore"),
    "P05": ("Port of Rotterdam", "Havenbedrijf Rotterdam", "Rotterdam Port Authority"),
    "P06": ("Port of Los Angeles", "Port of Long Beach"),
    "C01": ("Suez Canal Authority", "Suez Canal"),
    "C02": ("Bab el-Mandeb", "Bab al-Mandab"),
    "C03": ("Strait of Malacca", "Malacca Strait"),
    "C04": ("Panama Canal Authority", "Panama Canal"),
}


def normalise(name: str) -> str:
    """Lowercase and reduce to letters and digits separated by single spaces."""
    return " ".join(re.findall(r"[^\W_]+", name.lower()))


def aliases_for(site: Site) -> tuple[str, ...]:
    """The site's company plus its configured aliases, normalised and deduplicated."""
    names = [*SITE_ALIASES.get(site.site_id, ())]
    if site.company:
        # "Foxconn (Hon Hai)" holds two names; split on brackets and slashes.
        names += re.split(r"[()/]", site.company)
    normalised = (normalise(n) for n in names)
    return tuple(dict.fromkeys(n for n in normalised if n))


def bigquery_pattern(site: Site) -> str | None:
    """An RE2 pattern matching any alias as whole words in LOWER(organizations)."""
    aliases = aliases_for(site)
    if not aliases:
        return None
    words = (r"[^a-z0-9]+".join(re.escape(w) for w in alias.split()) for alias in aliases)
    return r"\b(?:" + "|".join(words) + r")\b"


def match(raw_names: list[str], site: Site) -> list[EntityMatch]:
    """Organisation names that contain one of the site's aliases as whole words."""
    aliases = aliases_for(site)
    matches = []
    for raw in raw_names:
        padded = f" {normalise(raw)} "
        alias = next((a for a in aliases if f" {a} " in padded), None)
        if alias:
            matches.append(EntityMatch(raw_name=raw, site_id=site.site_id, alias=alias))
    return matches


async def run(request: EntityResolutionRequest, ctx: RunContext) -> EntityResolutionResult:
    """Resolve organisation names for one site within run ``ctx.run_id``."""
    with ctx.bind():
        matches = match(request.raw_names, request.site)
        log.info(
            "entity_resolver.done",
            extra={"site_id": request.site.site_id, "matches": len(matches)},
        )
        return EntityResolutionResult(matches=matches)
