"""All Pydantic models shared across agents, tools and the API.

Rule: every agent returns one of these models, never free text. Models that end up in a
brief carry the source URL they came from, so every claim can be traced to an article.
"""

from datetime import date, datetime
from enum import StrEnum
from typing import Annotated, Literal, Self

from pydantic import BaseModel, ConfigDict, Field, HttpUrl, model_validator

Score = Annotated[int, Field(ge=1, le=5)]
Status = Literal["ongoing", "resolved", "unclear"]
Confidence = Literal["high", "medium", "low"]


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
    title: str | None = Field(default=None, description="Headline words from the URL slug.")
    source: str | None = Field(default=None, description="Publisher domain.")
    actors: list[str] = Field(default_factory=list)
    organizations: list[str] = Field(default_factory=list)
    themes: list[str] = Field(default_factory=list)
    tone: float | None = None
    place: str | None = Field(default=None, description="GDELT's geocoded place name.")
    distance_km: float | None = Field(default=None, ge=0)
    source_url: HttpUrl
    relevance: float | None = Field(
        default=None, ge=0, le=1, description="Ranking score set by query_agent."
    )
    cluster_size: int = Field(
        default=1, ge=1, description="Articles in this story cluster, including this one."
    )
    supporting_urls: list[HttpUrl] = Field(
        default_factory=list,
        description="Other articles in the same story cluster; not verified individually.",
    )
    matched_entities: list[str] = Field(
        default_factory=list,
        description="Organisation names that entity_resolver matched to this site's company.",
    )
    retrieval: Literal["location", "entity"] = Field(
        default="location",
        description="'location': geocoded inside this site's radius. 'entity': names this "
        "site's company but was geocoded near another monitored site.",
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


class Scope(StrEnum):
    """How a disruption relates to the site (the verifier's rubric)."""

    DIRECT = "direct"  # the site itself is affected
    INDIRECT = "indirect"  # a connected route, supplier region or hinterland is affected
    NOT_RELEVANT = "not_relevant"  # no real disruption that touches this site


class VerifiedEvent(_Model):
    """An event after the verifier has read (or failed to read) its source article.

    ``verdict`` is "yes" for direct and indirect disruptions; ``scope`` says which. Scope
    is None when the article could not be judged (and for runs before the rubric existed).
    """

    event: Event
    verdict: Verdict
    scope: Scope | None = None
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

    @model_validator(mode="after")
    def _scope_matches_verdict(self) -> Self:
        if self.scope is None:
            return self
        relevant = self.scope in (Scope.DIRECT, Scope.INDIRECT)
        if relevant != (self.verdict is Verdict.YES):
            raise ValueError(f"scope {self.scope} contradicts verdict {self.verdict}")
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
    """One claim in the brief. A claim without a source URL is invalid.

    ``status``, ``confidence`` and ``contradictions`` come from the reviewer's final pass;
    ``revisions`` counts how many times the report writer rewrote the claim on its request.
    """

    item_id: str | None = Field(default=None, description="I1, I2, ... within one brief.")
    site_id: str
    claim: str = Field(min_length=1)
    scope: Scope | None = None
    status: Status | None = None
    confidence: Confidence | None = None
    contradictions: list[str] = Field(
        default_factory=list, description="Statements the evidence contradicts (reviewer)."
    )
    revisions: int = Field(default=0, ge=0)
    severity: Score | None = None
    impact: Score | None = None
    source_urls: list[HttpUrl] = Field(min_length=1, description="Verified sources.")
    supporting_urls: list[HttpUrl] = Field(
        default_factory=list, description="Same-story articles that were not verified."
    )

    @property
    def risk(self) -> int:
        """severity x impact (1..25), 0 if unscored. Used to order the brief."""
        return (self.severity or 0) * (self.impact or 0)


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
        urls = {
            str(url) for item in self.items for url in [*item.source_urls, *item.supporting_urls]
        }
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
    raw_names: list[str] = Field(description="Organisation names as GDELT extracted them.")
    site: Site


class EntityMatch(_Model):
    raw_name: str
    site_id: str
    alias: str = Field(description="The company name or alias that matched.")


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

    assessment: Literal["direct", "indirect", "not_relevant", "unverifiable"]
    reason: str = Field(description="One sentence, in English.")
    # Defaults to None because the model sometimes omits the field instead of sending null.
    quote: str | None = Field(
        default=None,
        description="A short verbatim excerpt from article_text, in its original language.",
    )


class TriageCandidate(_Model):
    """One candidate story as the triage model sees it: metadata only, no URL."""

    candidate_id: str
    title: str | None
    source: str | None
    event_date: date
    place: str | None
    themes: list[str]
    organizations: list[str]
    matched_entities: list[str]
    cluster_size: int


class TriageInput(_Model):
    site: Site
    candidates: list[TriageCandidate]


class TriagePick(_Model):
    candidate_id: str
    reason: str = Field(description="One line, in English.")


class TriageDraft(_Model):
    """The triage model's answer: the candidates worth sending to the verifier."""

    picks: list[TriagePick]


class TriageSelection(_Model):
    event_id: str
    reason: str


class TriageResult(_Model):
    """Which candidates triage sent to the verifier, and why."""

    site_id: str
    considered: int = Field(ge=0)
    selected: list[TriageSelection]
    fallback: bool = Field(
        default=False, description="True if the triage call failed and top-ranked were used."
    )


class ScoringRequest(_Model):
    verified_event: VerifiedEvent
    site: Site


class ScorerInput(_Model):
    """What the risk scorer's LLM sees: the site and the verifier's evidence."""

    site: Site
    event_date: date
    scope: Scope | None
    reason: str
    quote: str
    cluster_size: int


class RiskJudgement(_Model):
    """The risk scorer LLM's structured answer."""

    severity: Score = Field(description="How serious the event is in itself, 1-5.")
    impact: Score = Field(description="How much it disrupts this specific site, 1-5.")
    reason: str = Field(description="One sentence, in English.")


class InvestigatorSource(_Model):
    """One of the story's own sources, as the investigator sees it (no URL)."""

    source_id: str = Field(description="Result ID for read_article: F1, F2, ...")
    title: str | None
    event_date: date
    verdict: Verdict
    scope: Scope | None
    reason: str
    quote: str | None


class InvestigatorInput(_Model):
    """What the investigator model is asked to check: one claim, or one unverifiable story."""

    site: Site
    subject: Literal["claim", "unverifiable_story"]
    claim: str
    sources: list[InvestigatorSource]
    search_until: date
    severity: Score | None
    impact: Score | None


Stance = Literal["supports", "contradicts", "context"]


class EvidenceSnippet(_Model):
    """A verbatim excerpt the investigator cites, by result ID (no URL)."""

    result_id: str
    quote: str = Field(description="Verbatim from the article text or the result's title.")
    stance: Stance = Field(description="Does it support or contradict the claim, or add context?")


class InvestigationJudgement(_Model):
    """The investigator model's structured answer."""

    summary: str = Field(description="At most three sentences, in English.")
    status: Status
    confidence: Confidence
    corroborating_ids: list[str] = Field(
        default_factory=list, description="Result IDs (R1, R2, ...) that corroborate it."
    )
    evidence: list[EvidenceSnippet] = Field(
        default_factory=list, description="Up to 5 verbatim excerpts, supporting or not."
    )


class InvestigationEvidence(_Model):
    """An excerpt checked to appear verbatim in what the investigator read, with its URL."""

    url: HttpUrl
    quote: str
    stance: Stance


class Investigation(_Model):
    """One investigated story (a brief claim or an unverifiable story), sources mapped to URLs.

    Claims are identified by ``item_id``; unverifiable stories by ``event_id``.
    ``source_url`` is the first of ``source_urls`` (kept for runs saved before claims were
    investigated as a whole).
    """

    item_id: str | None = None
    event_id: str | None = None
    site_id: str
    source_url: HttpUrl
    source_urls: list[HttpUrl] = Field(default_factory=list)
    trigger: Literal["high_risk", "unverifiable"]
    summary: str
    status: Status
    confidence: Confidence
    corroborating_urls: list[HttpUrl] = Field(default_factory=list)
    tool_calls: int = Field(ge=0)
    hit_limit: bool = False
    evidence: list[InvestigationEvidence] = Field(default_factory=list)


class ReportRequest(_Model):
    run_id: str
    date_range: DateRange
    sites: list[Site]
    verified_events: list[VerifiedEvent]
    scores: list[RiskScore] = Field(default_factory=list)
    investigations: list[Investigation] = Field(default_factory=list)
    considered: dict[str, int] = Field(
        default_factory=dict, description="Candidate stories triaged, per site_id."
    )


class Finding(_Model):
    """One confirmed event, as the report writer's LLM sees it.

    The model cites findings by ``finding_id`` and never sees or copies URLs; Python maps
    IDs back to sources, so a model cannot cite a URL that was not verified.
    """

    finding_id: str
    site_id: str
    scope: Scope | None
    event_date: date
    reason: str
    quote: str
    severity: Score | None = None
    impact: Score | None = None


class ReportDraftInput(_Model):
    date_range: DateRange
    sites: list[Site]
    findings: list[Finding]


class DraftItem(_Model):
    site_id: str
    claim: str = Field(description="One or two sentences, in English, stating the disruption.")
    finding_ids: list[str] = Field(
        min_length=1, description="finding_id values of the findings this claim is based on."
    )


class ReportDraft(_Model):
    """The report writer LLM's output; Python renders it to Markdown."""

    items: list[DraftItem]


class ReviewEvidence(_Model):
    reason: str
    quote: str | None


class ReviewSnippet(_Model):
    """An investigator excerpt as the reviewer sees it (no URL)."""

    stance: Stance
    quote: str


class ReviewItem(_Model):
    """One claim as the reviewer sees it, with all the evidence behind it (no URLs)."""

    item_id: str
    site_id: str
    scope: Scope | None
    claim: str
    severity: Score | None
    impact: Score | None
    verifier_evidence: list[ReviewEvidence]
    investigation_summary: str | None
    investigation_status: Status | None
    investigation_confidence: Confidence | None
    investigation_evidence: list[ReviewSnippet] = Field(default_factory=list)


class ReviewInput(_Model):
    items: list[ReviewItem]


class ClaimReview(_Model):
    item_id: str
    status: Status = Field(description="Is the disruption ongoing, resolved, or unclear?")
    confidence: Confidence
    contradicted: list[str] = Field(
        default_factory=list,
        description="Statements in the claim that the evidence contradicts, quoted briefly.",
    )
    corrections: list[str] = Field(
        default_factory=list,
        description="Specific edits for the report writer; empty if the claim is fine.",
    )


class ReviewDraft(_Model):
    """The reviewer model's answer: one review per claim."""

    reviews: list[ClaimReview]


class RevisionItem(_Model):
    item_id: str
    claim: str
    corrections: list[str]
    verifier_evidence: list[ReviewEvidence]
    investigation_summary: str | None


class RevisionInput(_Model):
    items: list[RevisionItem]


class RevisedClaim(_Model):
    item_id: str
    claim: str = Field(description="The corrected claim, one or two sentences in English.")


class RevisionDraft(_Model):
    """The report writer's rewrite of the claims the reviewer asked to correct."""

    claims: list[RevisedClaim]


class ReviewRound(_Model):
    """One pass of the critic loop: the reviews, and which claims were then rewritten."""

    round: int = Field(ge=1)
    reviews: list[ClaimReview]
    revised_item_ids: list[str] = Field(default_factory=list)


class SiteRun(_Model):
    """Everything the pipeline produced for one site, for auditing and evals."""

    site: Site
    candidates: list[Event]
    triage: TriageResult | None = None
    verified_events: list[VerifiedEvent]
    scores: list[RiskScore]
    investigations: list[Investigation] = Field(default_factory=list)
    bytes_processed: int = Field(ge=0)


class PipelineResult(_Model):
    """Orchestrator output with the per-site detail behind the brief."""

    brief: RiskBrief
    site_runs: list[SiteRun]
    investigations: list[Investigation] = Field(default_factory=list)
    review_rounds: list[ReviewRound] = Field(default_factory=list)
    model_calls: dict[str, int] = Field(description="Model calls per agent name.")


# ---------------------------------------------------------------------------
# Root agent
# ---------------------------------------------------------------------------


class RootRequest(_Model):
    """A plain-language question for the root agent."""

    question: str = Field(min_length=1)
    today: date


class RootAnswer(_Model):
    """The root agent's answer. The brief itself is attached by Python, not retyped."""

    summary: str = Field(description="Two to four sentences answering the question.")
    rationale: str = Field(description="One sentence: why these sites and dates.")
    site_ids: list[str]
    start_date: date
    end_date: date


class RootResult(_Model):
    answer: RootAnswer
    pipeline: PipelineResult | None = Field(
        default=None, description="The brief and its detail, if the pipeline ran."
    )
    model_calls: dict[str, int]
