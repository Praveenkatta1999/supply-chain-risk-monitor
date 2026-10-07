"""Ranking signals for candidate events. Pure functions, no I/O.

Each signal is scaled to 0..1 and combined with fixed weights:
    themes    how strongly the article's GDELT themes relate to ports, shipping and disruption
    site      whether the article names the site itself, not just a place inside its radius
    cluster   how many independent articles cover the same story (see clustering.py)
    tone      how negative the coverage is (GDELT tone, roughly -10..+10)
    distance  how close the geocoded place is to the site, relative to the site's radius

Lessons from the P05 runs that shaped this:
- Being geocoded inside the radius, or naming the site's city, is weak evidence: articles
  about Rotterdam the city (housing, drug cases, a travel guide) were all false positives.
  Only naming the site, its operator or its company (or an alias from entity_resolver)
  earns site credit, and it is the strongest single signal after themes.
- Wide coverage alone is not relevance: big political stories in a nearby city got many
  articles. So the cluster signal amplifies relevance rather than adding to it: it is
  scaled by the stronger of the theme and site signals.
"""

import re
from dataclasses import dataclass
from functools import lru_cache
from math import log2

from scrm.agents import entity_resolver
from scrm.schemas import Site

WEIGHTS = {"themes": 0.40, "site": 0.30, "cluster": 0.10, "tone": 0.10, "distance": 0.10}

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

# Words that describe a facility. A site name next to one of these ("port of rotterdam",
# "rotterdamse haven", "suez canal") identifies the site rather than just its city.
FACILITY_WORDS = (
    "port|ports|harbou?r|haven|havens|havenbedrijf|hafen|terminals?|canal|kanal|channel"
    "|strait|fabs?|plant|factory|campus"
)
# Words in site names that are not distinctive on their own.
GENERIC_NAME_WORDS = {
    "port", "ports", "of", "the", "and", "fab", "fabs", "plant", "campus", "complex",
    "assembly", "test", "strait", "canal",
}  # fmt: skip


@dataclass(frozen=True)
class Signals:
    themes: float
    site: float
    tone: float
    distance: float

    def relevance(self, cluster_size: int = 1) -> float:
        coverage = cluster_score(cluster_size) * max(self.themes, self.site)
        score = (
            WEIGHTS["themes"] * self.themes
            + WEIGHTS["site"] * self.site
            + WEIGHTS["cluster"] * coverage
            + WEIGHTS["tone"] * self.tone
            + WEIGHTS["distance"] * self.distance
        )
        return round(score, 4)


def theme_score(themes: list[str]) -> float:
    total = sum(PORT_THEMES.get(theme, 0.0) for theme in set(themes))
    return min(1.0, total / THEME_SATURATION)


def tone_score(tone: float | None) -> float:
    return 0.0 if tone is None else min(1.0, max(0.0, -tone / 10))


def distance_score(distance_km: float | None, radius_km: float) -> float:
    return 0.0 if distance_km is None else min(1.0, max(0.0, 1 - distance_km / radius_km))


def cluster_score(cluster_size: int) -> float:
    """0 for a lone article, rising to 1.0 at 8 articles on the same story."""
    return min(1.0, log2(max(cluster_size, 1)) / 3)


# ---------------------------------------------------------------------------
# Site match
# ---------------------------------------------------------------------------


def _phrase(text: str) -> str:
    """Lowercase and turn any run of non-letters (hyphens, slashes) into one space."""
    return " ".join(re.findall(r"[^\W\d_]+", text.lower()))


def name_phrases(site_name: str) -> list[str]:
    """Distinctive parts of a site name: 'Ports of Los Angeles and Long Beach' ->
    ['los angeles', 'long beach'], 'Port of Rotterdam' -> ['rotterdam']."""
    parts = re.split(r"\band\b", site_name.lower())
    phrases = [
        " ".join(w for w in _phrase(p).split() if w not in GENERIC_NAME_WORDS) for p in parts
    ]
    return [p for p in phrases if p]


@lru_cache(maxsize=64)
def _identity_pattern(site_name: str, aliases: tuple[str, ...]) -> re.Pattern[str]:
    alternatives = []
    for name in name_phrases(site_name):
        n = re.escape(name)
        # "<name>... <facility>" within two words, or "<facility> of <name>"
        alternatives.append(rf"\b{n}\w*(?: \w+)? (?:{FACILITY_WORDS})\b")
        alternatives.append(rf"\b(?:{FACILITY_WORDS}) (?:of |van )?{n}\b")
    # The company and its aliases (entity_resolver.SITE_ALIASES) count on their own.
    alternatives += [rf"\b{re.escape(_phrase(alias))}\b" for alias in aliases if _phrase(alias)]
    return re.compile("|".join(alternatives))


def site_match_score(site: Site, slug: str, organizations: list[str], place: str | None) -> float:
    """1.0 if the URL slug, an organisation or the place names the site itself, else 0.0."""
    identity = _identity_pattern(site.site_name, entity_resolver.aliases_for(site))
    # Match each place component separately, so "South Channel, Shanghai" does not read
    # as "channel shanghai".
    places = (place or "").split(",")
    texts = [_phrase(slug), *(_phrase(o) for o in organizations), *(_phrase(p) for p in places)]
    return 1.0 if any(identity.search(text) for text in texts) else 0.0
