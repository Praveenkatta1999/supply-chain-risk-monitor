"""Group candidate articles that describe the same story. Pure functions, no I/O.

GDELT has no headlines, so the URL slug stands in for the title: news URLs almost always
carry the headline ("schepen-botsen-op-elkaar-bij-kinderdijk-..."). Two articles are the
same story when all of these hold:
    - same geocoded place (GDELT FullName), and both have one;
    - published within MAX_DAYS_APART days of each other;
    - their slugs share at least MIN_SHARED_TOKENS distinctive words, and those make up
      at least MIN_OVERLAP of the shorter slug's words (overlap coefficient).
Words that are common across the site's slugs ("rotterdam", "shanghai", "china") do not
count as shared, or every story about a city would merge.

Algorithm (input must be in rank order, best first):
1. Syndicated copies, identical slug words at the same place, are grouped directly.
2. Leader clustering over those groups: each group joins the highest-ranked existing
   cluster whose leader is the same story, or starts a new cluster. Comparing against the
   leader only, not every member, stops chains of loosely related stories from merging.
"""

import re
from collections import Counter, defaultdict
from dataclasses import dataclass
from datetime import date
from urllib.parse import urlsplit

MAX_DAYS_APART = 3
MIN_SHARED_TOKENS = 3
MIN_OVERLAP = 0.35
# A word in more than this share of a site's distinct slugs (and at least
# MIN_COMMON_COUNT of them) is too common to suggest two articles are the same story.
COMMON_TOKEN_SHARE = 0.01
MIN_COMMON_COUNT = 20
MIN_TOKEN_LENGTH = 3
STOPWORDS = {
    "the", "and", "for", "with", "from", "that", "this", "after", "over", "into", "news",
    "article", "articles", "html", "htm", "php", "aspx", "amp", "index", "story",
    "van", "het", "een", "der", "die", "das", "und", "des", "les", "una", "del", "que",
}  # fmt: skip


@dataclass(frozen=True)
class Item:
    """The fields clustering needs from one candidate."""

    url: str
    place: str | None
    day: date


def _headline_words(url: str) -> list[str]:
    """All words of the URL path segment that looks most like a headline."""
    segments = [s for s in urlsplit(url).path.split("/") if s]
    best: list[str] = []
    for segment in segments:
        words = [w for w in re.split(r"[^a-z]+", segment.lower().rsplit(".", 1)[0]) if w]
        if len(words) > len(best):
            best = words
    return best


def slug_tokens(url: str) -> frozenset[str]:
    """Distinctive words from the URL's headline segment, for comparing stories."""
    words = _headline_words(url)
    return frozenset(w for w in words if len(w) >= MIN_TOKEN_LENGTH and w not in STOPWORDS)


def slug_title(url: str) -> str | None:
    """A readable stand-in for the headline ("rhine water levels fall ..."), if any."""
    words = _headline_words(url)
    real_words = [w for w in words if len(w) >= MIN_TOKEN_LENGTH]  # not hex IDs like "f a bf"
    return " ".join(words) if len(real_words) >= MIN_SHARED_TOKENS else None


def independent_reports(urls: list[str]) -> int:
    """Distinct write-ups among same-story URLs: syndicated copies (same slug) count once."""
    return len({slug_tokens(url) or url for url in urls})


def same_story(a: Item, b: Item, tokens_a: frozenset[str], tokens_b: frozenset[str]) -> bool:
    if a.place is None or a.place != b.place:
        return False
    if abs((a.day - b.day).days) > MAX_DAYS_APART:
        return False
    shared = len(tokens_a & tokens_b)
    smaller = min(len(tokens_a), len(tokens_b))
    return shared >= MIN_SHARED_TOKENS and shared / smaller >= MIN_OVERLAP


def _syndication_groups(items: list[Item], tokens: list[frozenset[str]]) -> list[list[int]]:
    """Group items with identical slug words at the same place, within the date window."""
    groups: list[list[int]] = []
    by_key: dict[tuple[frozenset[str], str], int] = {}
    for i, (item, words) in enumerate(zip(items, tokens, strict=True)):
        key = (words, item.place) if item.place and len(words) >= MIN_SHARED_TOKENS else None
        g = by_key.get(key) if key else None
        if g is not None and same_story(items[groups[g][0]], item, words, words):
            groups[g].append(i)
            continue
        if key:
            by_key[key] = len(groups)
        groups.append([i])
    return groups


def _common_tokens(token_sets: list[frozenset[str]]) -> set[str]:
    counts = Counter(word for words in token_sets for word in words)
    limit = max(MIN_COMMON_COUNT, COMMON_TOKEN_SHARE * len(token_sets))
    return {word for word, n in counts.items() if n > limit}


def cluster(items: list[Item]) -> list[list[int]]:
    """Return clusters as lists of indices into ``items`` (best first), in rank order."""
    tokens = [slug_tokens(item.url) for item in items]
    groups = _syndication_groups(items, tokens)
    common = _common_tokens([tokens[g[0]] for g in groups])
    distinct = [tokens[g[0]] - common for g in groups]

    clusters: list[list[int]] = []  # item indices; clusters[c][0] is the leader
    leader_group: list[int] = []  # group index of each cluster's leader
    index: dict[str, list[int]] = defaultdict(list)  # word -> clusters whose leader has it
    for g, members in enumerate(groups):
        item, words = items[members[0]], distinct[g]
        candidates = sorted({c for word in words for c in index[word]})
        match = next(
            (
                c
                for c in candidates
                if same_story(items[clusters[c][0]], item, distinct[leader_group[c]], words)
            ),
            None,
        )
        if match is None:
            for word in words:
                index[word].append(len(clusters))
            clusters.append(list(members))
            leader_group.append(g)
        else:
            clusters[match].extend(members)
    return clusters
