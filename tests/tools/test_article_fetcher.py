import asyncio

import httpx

from scrm.schemas import Article, FetchFailure, FetchFailureReason
from scrm.tools.article_fetcher import extract_text, fetch_article

PARAGRAPH = "Dockworkers at the port walked out on Monday, halting container handling. " * 8
ARTICLE_HTML = f"""
<html><head><title>Port strike</title><script>var x = 1;</script></head>
<body><nav><p>Home | News</p></nav>
<h1>Strike halts port</h1><p>{PARAGRAPH}</p>
<footer><p>Copyright</p></footer></body></html>
"""


def fetch_with(handler, **kwargs) -> Article | FetchFailure:
    async def go():
        async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
            return await fetch_article("https://news.example/a", client=client, **kwargs)

    return asyncio.run(go())


def html_response(body: str, status: int = 200) -> httpx.Response:
    return httpx.Response(status, text=body, headers={"content-type": "text/html; charset=utf-8"})


def test_extract_text_keeps_paragraphs_and_skips_chrome():
    title, text = extract_text(ARTICLE_HTML)
    assert title == "Port strike"
    assert text.startswith("Strike halts port\nDockworkers")
    assert "Home" not in text and "Copyright" not in text and "var x" not in text


def test_returns_article_on_success():
    result = fetch_with(lambda request: html_response(ARTICLE_HTML))
    assert isinstance(result, Article)
    assert result.title == "Port strike"
    assert "halting container handling" in result.text


def test_404_is_dead_link():
    result = fetch_with(lambda request: html_response("gone", status=404))
    assert isinstance(result, FetchFailure)
    assert result.reason is FetchFailureReason.DEAD_LINK


def test_403_is_paywall():
    result = fetch_with(lambda request: html_response("subscribe", status=403))
    assert result.reason is FetchFailureReason.PAYWALL


def test_too_little_text_is_paywall():
    result = fetch_with(lambda request: html_response("<p>Subscribe to read more.</p>"))
    assert result.reason is FetchFailureReason.PAYWALL


def test_non_html_is_refused():
    result = fetch_with(
        lambda request: httpx.Response(
            200, content=b"%PDF", headers={"content-type": "application/pdf"}
        )
    )
    assert result.reason is FetchFailureReason.NOT_HTML


def test_size_cap():
    result = fetch_with(lambda request: html_response(ARTICLE_HTML), max_bytes=100)
    assert result.reason is FetchFailureReason.TOO_LARGE


def test_timeout_is_reported_not_raised():
    def handler(request):
        raise httpx.ReadTimeout("slow", request=request)

    assert fetch_with(handler).reason is FetchFailureReason.TIMEOUT


def test_connection_error_is_dead_link():
    def handler(request):
        raise httpx.ConnectError("no such host", request=request)

    assert fetch_with(handler).reason is FetchFailureReason.DEAD_LINK
