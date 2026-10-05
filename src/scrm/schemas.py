"""All Pydantic models shared across agents, tools and the API.

Rule: every agent returns one of these models, never free text. Models that end up in a
brief carry the source URL they came from, so every claim can be traced to an article.
"""

from datetime import date, datetime
from enum import StrEnum
from typing import Annotated, Literal, Self

from pydantic import BaseModel, ConfigDict, Field, HttpUrl, model_validator

Score = Annotated[int, Field(ge=1, le=5)]


class _Model(BaseModel):
    model_config = ConfigDict(extra="forbid")


# ---------------------------------------------------------------------------
# Core domain
# ---------------------------------------------------------------------------


class SiteType(StrEnum):
    FACTORY = "factory"
    PORT = "port"
    CHOKEPOINT = "chokepoint"


class Site(_Model):
    """One row of scrm.sites."""

    site_id: str
    site_name: str
    company: str | None = None
    site_type: SiteType
    component: str | None = None
    city: str | None = None
    country: str
    lat: float = Field(ge=-90, le=90)
    lon: float = Field(ge=-180, le=180)
    radius_km: float = Field(gt=0)


class DateRange(_Model):
    start: date
    end: date

    @model_validator(mode="after")
    def _start_before_end(self) -> Self:
        if self.start > self.end:
            raise ValueError("start must be on or before end")
        return self


class Event(_Model):
    """A candidate disruption near a site, from scrm.events_near_sites or scrm.gkg_near_sites."""

    event_id: str
    source_table: Literal["events_near_sites", "gkg_near_sites"]
    site_id: str
    event_date: date
    title: str | None = None
    actors: list[str] = Field(default_factory=list)
    organizations: list[str] = Field(default_factory=list)
    themes: list[str] = Field(default_factory=list)
    tone: float | None = None
    distance_km: float | None = Field(default=None, ge=0)
    source_url: HttpUrl
    relevance: float | None = Field(
        default=None, ge=0, le=1, description="Ranking score set by query_agent."
    )


class Article(_Model):
    """A fetched source article."""

    url: HttpUrl
    title: str | None = None
    text: str
    fetched_at: datetime


class FetchFailureReason(StrEnum):
    DEAD_LINK = "dead_link"  # 404/410, DNS failure, connection refused
    PAYWALL = "paywall"  # 401/402/403, or a page with too little article text
    TIMEOUT = "timeout"
    TOO_LARGE = "too_large"
    NOT_HTML = "not_html"
    HTTP_ERROR = "http_error"  # any other non-2xx status


class FetchFailure(_Model):
    """Why an article could not be fetched. Returned, never raised, by article_fetcher."""

    url: HttpUrl
    reason: FetchFailureReason
    detail: str


class Verdict(StrEnum):
    YES = "yes"  # a real disruption that affects this site
    NO = "no"  # a false positive
    UNVERIFIABLE = "unverifiable"  # the article could not be read; nothing was guessed


class VerifiedEvent(_Model):
    """An event after the verifier has read (or failed to read) its source article."""

    event: Event
    verdict: Verdict
    reason: str = Field(min_length=1, description="One sentence explaining the verdict.")
    quote: str | None = Field(
        default=None, description="Verbatim text from the article supporting the verdict."
    )
    evidence_url: HttpUrl

    @model_validator(mode="after")
    def _yes_needs_a_quote(self) -> Self:
        if self.verdict is Verdict.YES and not self.quote:
            raise ValueError("a 'yes' verdict must quote the article")
        return self


class RiskScore(_Model):
    """Severity and impact for one verified event at one site."""

    event_id: str
    site_id: str
    severity: Score
    impact: Score
    reason: str = Field(min_length=1, max_length=500)
    source_url: HttpUrl


class BriefItem(_Model):
    """One claim in the brief. A claim without a source URL is invalid."""

    site_id: str
    claim: str = Field(min_length=1)
    severity: Score | None = None  # None until risk_scorer is implemented
    impact: Score | None = None
    source_urls: list[HttpUrl] = Field(min_length=1)


class RiskBrief(_Model):
    """The daily risk brief."""

    run_id: str
    generated_at: datetime
    date_range: DateRange
    site_ids: list[str]
    items: list[BriefItem]
    markdown: str

    @model_validator(mode="after")
    def _every_source_is_linked(self) -> Self:
        urls = {str(url) for item in self.items for url in item.source_urls}
        missing = sorted(url for url in urls if url not in self.markdown)
        if missing:
            raise ValueError(f"markdown is missing source links: {missing}")
        return self


# ---------------------------------------------------------------------------
# Agent inputs and outputs
# ---------------------------------------------------------------------------


class BriefRequest(_Model):
    """Orchestrator input (also the POST /brief body). Empty site_ids means all sites."""

    site_ids: list[str] = Field(default_factory=list)
    date_range: DateRange


class QueryRequest(_Model):
    site_ids: list[str] = Field(min_length=1)
    date_range: DateRange
    limit: int = Field(default=20, ge=1, le=200, description="Top N candidates per site.")


class QueryResult(_Model):
    events: list[Event]
    sql: list[str] = Field(description="Every statement that was executed, for auditing.")
    bytes_processed: int = Field(ge=0)


class EntityResolutionRequest(_Model):
    raw_names: list[str]
    candidate_companies: list[str]


class EntityMatch(_Model):
    raw_name: str
    company: str | None = Field(description="Matched company, or None if no confident match.")
    confidence: float = Field(ge=0, le=1)
    reason: str


class EntityResolutionResult(_Model):
    matches: list[EntityMatch]


class VerificationRequest(_Model):
    event: Event
    site: Site


class VerifierInput(_Model):
    """What the verifier's LLM sees: the site, the candidate, and the article text."""

    site: Site
    event_date: date
    article_url: HttpUrl
    article_title: str | None
    article_text: str


class VerifierJudgement(_Model):
    """The verifier LLM's structured answer, before the quote is checked against the text."""

    verdict: Verdict
    reason: str = Field(description="One sentence, in English.")
    quote: str | None = Field(
        description="A short verbatim excerpt from article_text, in its original language."
    )


class ScoringRequest(_Model):
    verified_event: VerifiedEvent
    site: Site


class ReportRequest(_Model):
    run_id: str
    date_range: DateRange
    sites: list[Site]
    verified_events: list[VerifiedEvent]
    scores: list[RiskScore] = Field(default_factory=list)  # unused until risk_scorer exists


class Finding(_Model):
    """One confirmed event, as the report writer's LLM sees it."""

    site_id: str
    event_date: date
    source_url: HttpUrl
    reason: str
    quote: str


class ReportDraftInput(_Model):
    date_range: DateRange
    sites: list[Site]
    findings: list[Finding]


class DraftItem(_Model):
    site_id: str
    claim: str = Field(description="One or two sentences, in English, stating the disruption.")
    source_urls: list[HttpUrl] = Field(
        min_length=1, description="source_url values of the findings this claim is based on."
    )


class ReportDraft(_Model):
    """The report writer LLM's output; Python renders it to Markdown."""

    items: list[DraftItem]
