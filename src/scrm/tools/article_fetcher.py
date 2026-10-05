"""Fetch a news article by URL and return its readable text.

Failures (dead links, paywalls, timeouts, oversized or non-HTML responses) come back as a
``FetchFailure`` the verifier can act on; this module never raises into the pipeline.
Text extraction is deliberately simple (paragraphs and headings via the standard library
HTML parser), which works for most news sites without adding a dependency.
"""

from datetime import UTC, datetime
from html.parser import HTMLParser
from typing import ClassVar

import httpx
from pydantic import HttpUrl

from scrm.schemas import Article, FetchFailure, FetchFailureReason
from scrm.telemetry import get_logger

log = get_logger(__name__)

DEFAULT_TIMEOUT_S = 10.0
MAX_BYTES = 2_000_000
# Pages with less article text than this are almost always paywalls, consent walls or
# JavaScript-rendered shells, so they are treated as unreadable rather than guessed at.
MIN_TEXT_CHARS = 400
USER_AGENT = "scrm-risk-monitor/0.1 (supply chain research prototype)"

_PAYWALL_STATUSES = {401, 402, 403, 451}
_DEAD_STATUSES = {404, 410}


class _TextExtractor(HTMLParser):
    """Collect the text of <title>, headings and paragraphs, skipping page chrome."""

    _SKIP: ClassVar = frozenset(
        {"script", "style", "noscript", "nav", "header", "footer", "aside", "form", "svg"}
    )
    _BLOCKS: ClassVar = frozenset({"p", "h1", "h2", "h3", "li"})

    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.title: str | None = None
        self.blocks: list[str] = []
        self._skip_depth = 0
        self._in_title = False
        self._current: list[str] | None = None

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        if tag in self._SKIP:
            self._skip_depth += 1
        elif tag == "title":
            self._in_title = True
        elif tag in self._BLOCKS and self._skip_depth == 0:
            self._flush()
            self._current = []

    def handle_endtag(self, tag: str) -> None:
        if tag in self._SKIP and self._skip_depth:
            self._skip_depth -= 1
        elif tag == "title":
            self._in_title = False
        elif tag in self._BLOCKS:
            self._flush()

    def handle_data(self, data: str) -> None:
        if self._in_title and self.title is None:
            self.title = " ".join(data.split()) or None
        elif self._current is not None and self._skip_depth == 0:
            self._current.append(data)

    def close(self) -> None:
        super().close()
        self._flush()

    def _flush(self) -> None:
        if self._current is not None:
            text = " ".join("".join(self._current).split())
            if text:
                self.blocks.append(text)
        self._current = None


def extract_text(html: str) -> tuple[str | None, str]:
    """Return (title, body text) from an HTML page."""
    parser = _TextExtractor()
    parser.feed(html)
    parser.close()
    return parser.title, "\n".join(parser.blocks)


async def fetch_article(
    url: HttpUrl | str,
    *,
    client: httpx.AsyncClient | None = None,
    timeout_s: float = DEFAULT_TIMEOUT_S,
    max_bytes: int = MAX_BYTES,
) -> Article | FetchFailure:
    """Download ``url`` and return the article text, or a ``FetchFailure`` explaining why not."""
    url = str(url)
    if client is None:
        async with _new_client(timeout_s) as own_client:
            return await _fetch(own_client, url, max_bytes)
    return await _fetch(client, url, max_bytes)


def _new_client(timeout_s: float) -> httpx.AsyncClient:
    return httpx.AsyncClient(
        timeout=timeout_s, follow_redirects=True, headers={"User-Agent": USER_AGENT}
    )


async def _fetch(client: httpx.AsyncClient, url: str, max_bytes: int) -> Article | FetchFailure:
    def fail(reason: FetchFailureReason, detail: str) -> FetchFailure:
        log.info("article.fetch_failed", extra={"url": url, "reason": reason, "detail": detail})
        return FetchFailure(url=url, reason=reason, detail=detail)

    try:
        async with client.stream("GET", url) as response:
            if response.status_code in _PAYWALL_STATUSES:
                return fail(FetchFailureReason.PAYWALL, f"HTTP {response.status_code}")
            if response.status_code in _DEAD_STATUSES:
                return fail(FetchFailureReason.DEAD_LINK, f"HTTP {response.status_code}")
            if not response.is_success:
                return fail(FetchFailureReason.HTTP_ERROR, f"HTTP {response.status_code}")
            content_type = response.headers.get("content-type", "")
            if "html" not in content_type:
                return fail(FetchFailureReason.NOT_HTML, content_type or "no content-type")
            body = bytearray()
            async for chunk in response.aiter_bytes():
                body.extend(chunk)
                if len(body) > max_bytes:
                    return fail(FetchFailureReason.TOO_LARGE, f"over {max_bytes:,} bytes")
            html = body.decode(response.encoding or "utf-8", errors="replace")
    except httpx.TimeoutException as exc:
        return fail(FetchFailureReason.TIMEOUT, type(exc).__name__)
    except httpx.HTTPError as exc:
        return fail(FetchFailureReason.DEAD_LINK, f"{type(exc).__name__}: {exc}")

    title, text = extract_text(html)
    if len(text) < MIN_TEXT_CHARS:
        return fail(FetchFailureReason.PAYWALL, f"only {len(text)} chars of article text")
    return Article(url=url, title=title, text=text, fetched_at=datetime.now(UTC))
