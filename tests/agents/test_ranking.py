import pytest

from scrm.agents import ranking
from scrm.schemas import Site, SiteType


def make_site(site_name, city, company=None, site_type=SiteType.PORT) -> Site:
    return Site(
        site_id="X",
        site_name=site_name,
        company=company,
        site_type=site_type,
        city=city,
        country="NL",
        lat=0,
        lon=0,
        radius_km=50,
    )


ROTTERDAM = make_site("Port of Rotterdam", "Rotterdam")
SHANGHAI = make_site("Port of Shanghai", "Shanghai")
TSMC = make_site("TSMC Hsinchu fabs", "Hsinchu", company="TSMC", site_type=SiteType.FACTORY)


def test_name_phrases_drop_generic_words():
    assert ranking.name_phrases("Port of Rotterdam") == ["rotterdam"]
    assert ranking.name_phrases("Ports of Los Angeles and Long Beach") == [
        "los angeles",
        "long beach",
    ]
    assert ranking.name_phrases("Suez Canal") == ["suez"]


@pytest.mark.parametrize(
    ("site", "slug", "orgs", "place", "expected"),
    [
        # The site itself is named: in the headline, an organisation or the place.
        (ROTTERDAM, "/artikel/805432-rotterdamse-haven-te-afhankelijk", [], None, 1.0),
        (ROTTERDAM, "/news/strike-shuts-port-of-rotterdam", [], None, 1.0),
        (ROTTERDAM, "/x", ["Havenbedrijf Rotterdam"], None, 1.0),
        (SHANGHAI, "/x", ["Shanghai International Port Group"], None, 1.0),
        (TSMC, "/tech/tsmc-halts-production", [], None, 1.0),
        # The city alone is not the site: in the headline, an organisation or the place.
        (ROTTERDAM, "/nieuws/pompenburg-rotterdam-gesloopt", [], None, 0.0),
        (ROTTERDAM, "/news/drug-arrest-in-russia", ["Erasmus University Rotterdam"], None, 0.0),
        (ROTTERDAM, "/x", [], "Rotterdam, Zuid-Holland, Netherlands", 0.0),
        # Place components are matched separately: not "channel shanghai".
        (SHANGHAI, "/x", [], "South Channel, Shanghai, China", 0.0),
    ],
)
def test_site_match_score(site, slug, orgs, place, expected):
    assert ranking.site_match_score(site, slug, orgs, place) == expected


def test_component_scores_are_bounded():
    assert ranking.theme_score(["WB_167_PORTS", "STRIKE", "MARITIME"]) == 1.0
    assert ranking.theme_score(["EDUCATION"]) == 0.0
    assert ranking.tone_score(-20) == 1.0
    assert ranking.tone_score(3) == 0.0
    assert ranking.distance_score(0, 50) == 1.0
    assert ranking.distance_score(60, 50) == 0.0
    assert ranking.cluster_score(1) == 0.0
    assert ranking.cluster_score(8) == ranking.cluster_score(100) == 1.0


def test_site_match_outranks_mere_proximity():
    about_site = ranking.Signals(themes=0.3, site=1.0, tone=0.3, distance=0.2)
    just_nearby = ranking.Signals(themes=0.3, site=0.0, tone=0.3, distance=1.0)
    assert about_site.relevance() > just_nearby.relevance()


def test_cluster_size_raises_relevance():
    signals = ranking.Signals(themes=0.5, site=0.0, tone=0.5, distance=0.5)
    assert signals.relevance(cluster_size=4) > signals.relevance(cluster_size=1)


def test_wide_coverage_of_an_irrelevant_story_earns_nothing():
    politics = ranking.Signals(themes=0.0, site=0.0, tone=0.8, distance=0.8)
    assert politics.relevance(cluster_size=50) == politics.relevance(cluster_size=1)


def test_weights_sum_to_one():
    assert sum(ranking.WEIGHTS.values()) == pytest.approx(1.0)
