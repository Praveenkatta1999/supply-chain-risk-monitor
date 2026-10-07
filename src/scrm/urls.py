"""URL identity: when two URLs point at the same article.

GDELT and news sites mix ``http://`` and ``https://`` for the same page, so anywhere the
pipeline asks "is this the same article?" (deduplication, matching scores and evidence to
claims, excluding a story's own sources from its corroboration) it compares
``article_key(url)`` rather than the raw string. Links shown in a brief are left as found.
"""

from urllib.parse import urlsplit

from pydantic import HttpUrl


def article_key(url: HttpUrl | str) -> str:
    """The URL without its scheme, host lowercased: http and https compare equal."""
    parts = urlsplit(str(url))
    key = f"//{(parts.netloc or '').lower()}{parts.path}"
    return f"{key}?{parts.query}" if parts.query else key


def unique_urls[U: (HttpUrl, str)](urls: list[U]) -> list[U]:
    """Drop later URLs that point at the same article as an earlier one, keeping order."""
    seen: dict[str, U] = {}
    for url in urls:
        seen.setdefault(article_key(url), url)
    return list(seen.values())
