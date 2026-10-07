from datetime import date

from scrm.agents import clustering
from scrm.agents.clustering import Item

KINDERDIJK = "Kinderdijk, Zuid-Holland, Netherlands"
SLUG = "schepen-botsen-op-elkaar-bij-kinderdijk-flinke-schade-aan-duits-schip"


def item(url, place=KINDERDIJK, day=10) -> Item:
    return Item(url=url, place=place, day=date(2026, 9, day))


def test_slug_tokens_pick_the_headline_segment():
    url = f"https://www.hetkontakt.nl/krimpenerwaard/nieuws/478097/{SLUG}-xq9070"
    assert clustering.slug_tokens(url) == {
        "schepen", "botsen", "elkaar", "bij", "kinderdijk", "flinke", "schade", "aan",
        "duits", "schip",
    }  # fmt: skip
    assert clustering.slug_tokens("https://blog.udn.com/jefnjil/192520027") == {"jefnjil"}


def test_same_story_on_regional_editions_clusters():
    items = [
        item(f"https://www.hetkontakt.nl/krimpenerwaard/nieuws/478097/{SLUG}-xq9070"),
        item(f"https://www.hetkontakt.nl/klaroen/nieuws/478092/{SLUG}"),
        item(f"https://www.hetkontakt.nl/alblasserwaard/nieuws/478093/{SLUG}-vza397", day=12),
        item("https://example.nl/nieuws/brug-over-de-noord-blijft-open"),
    ]
    assert clustering.cluster(items) == [[0, 1, 2], [3]]


def test_different_place_or_far_apart_dates_do_not_cluster():
    url = f"https://a.example/{SLUG}"
    assert clustering.cluster([item(url), item(f"https://b.example/{SLUG}", place="Delft")]) == [
        [0],
        [1],
    ]
    assert clustering.cluster([item(url), item(f"https://b.example/{SLUG}", day=14)]) == [
        [0],
        [1],
    ]
    assert clustering.cluster([item(url, place=None), item(f"https://b.example/{SLUG}", None)]) == [
        [0],
        [1],
    ]


def test_syndicated_copies_cluster_even_when_their_words_are_common():
    slug = "discover-top-experiences-in-the-netherlands-second-city"
    copies = [item(f"https://paper{i}.example/story/9351538/{slug}") for i in range(80)]
    clusters = clustering.cluster(copies)
    assert len(clusters) == 1 and len(clusters[0]) == 80
    assert clustering.independent_reports([c.url for c in copies]) == 1


def test_leader_clustering_does_not_chain_loosely_related_stories():
    # b shares three words with a, c shares three with b, but c and a share none.
    a = item("https://x.example/alpha-bravo-charlie-delta")
    b = item("https://x.example/alpha-bravo-charlie-echo-foxtrot-golf")
    c = item("https://x.example/echo-foxtrot-golf-hotel-india-juliet")
    assert clustering.cluster([a, b, c]) == [[0, 1], [2]]


def test_words_common_across_a_site_do_not_count_as_shared():
    # Without the common-word rule these two would match: 3 shared words, overlap 0.75.
    stories = [
        item("https://x.example/rotterdam-haven-nieuws-staking"),
        item("https://y.example/rotterdam-haven-nieuws-brand"),
    ]
    assert clustering.cluster(stories) == [[0, 1]]
    # 30 distinct slugs make "rotterdam", "haven" and "nieuws" common at this site.
    words = [a + b + c for a in "pqr" for b in "stu" for c in "vwxyz"][:30]
    filler = [item(f"https://z.example/rotterdam-haven-nieuws-{w}", day=20) for w in words]
    clusters = clustering.cluster(stories + filler)
    assert [0] in clusters and [1] in clusters


def test_slug_title_is_readable_headline_or_none():
    url = "https://www.hellenicshippingnews.com/rhine-water-levels-fall-to-new-record-low/"
    assert clustering.slug_title(url) == "rhine water levels fall to new record low"
    assert clustering.slug_title("https://www.netscape.com/f2a-bf3c-f9eb") is None
    assert clustering.slug_title("https://www.setn.com/news/1906280") is None
