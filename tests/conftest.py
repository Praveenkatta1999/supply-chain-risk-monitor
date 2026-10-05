from datetime import date

import pytest

from scrm.schemas import DateRange, Event, Site, SiteType


@pytest.fixture
def site() -> Site:
    return Site(
        site_id="SUP-01",
        site_name="Example Assembly Plant",
        company="Example Co",
        site_type=SiteType.FACTORY,
        component="mainboard",
        city="Zhengzhou",
        country="CN",
        lat=34.75,
        lon=113.62,
        radius_km=50,
    )


@pytest.fixture
def date_range() -> DateRange:
    return DateRange(start=date(2026, 9, 1), end=date(2026, 9, 7))


@pytest.fixture
def event() -> Event:
    return Event(
        event_id="gkg-123",
        source_table="gkg_near_sites",
        site_id="SUP-01",
        event_date=date(2026, 9, 3),
        title="Workers strike at plant",
        themes=["STRIKE"],
        source_url="https://example.com/news/strike",
    )
