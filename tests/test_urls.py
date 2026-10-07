from scrm.urls import article_key, unique_urls


def test_http_and_https_are_the_same_article():
    assert article_key("http://www.Example.com/a?b=1") == article_key(
        "https://www.example.com/a?b=1"
    )
    assert article_key("https://example.com/a") != article_key("https://example.com/b")


def test_unique_urls_keeps_the_first_of_each_article():
    urls = ["http://x.example/a", "https://x.example/a", "https://x.example/b"]
    assert unique_urls(urls) == ["http://x.example/a", "https://x.example/b"]
