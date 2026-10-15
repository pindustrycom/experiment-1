from __future__ import annotations

import uuid
from datetime import date, datetime
from typing import Any, Generic, Literal, TypeVar

from pydantic import BaseModel, ConfigDict, EmailStr, Field, field_validator

T = TypeVar("T")


class Page(BaseModel, Generic[T]):
    items: list[T]
    total: int
    page: int
    page_size: int


class ErrorBody(BaseModel):
    code: str
    message: str
    details: Any | None = None


class ErrorResponse(BaseModel):
    error: ErrorBody


# ---------------------------------------------------------------- auth
class LoginRequest(BaseModel):
    email: str = Field(max_length=255)
    password: str = Field(max_length=256)


class UserOut(BaseModel):
    model_config = ConfigDict(from_attributes=True)
    id: int
    email: str
    role: str


class ChangePassword(BaseModel):
    current_password: str
    new_password: str = Field(min_length=12, max_length=256)


# ---------------------------------------------------------------- companies / pairs
class CompanyOut(BaseModel):
    model_config = ConfigDict(from_attributes=True)
    id: int
    ticker: str
    name: str
    cik: str | None
    sector: str | None
    industry: str | None
    sic_code: str | None
    listing_status: str


class LatestScore(BaseModel):
    value: float | None
    status: str
    as_of: datetime
    change_7d: float | None = None
    model_version: str | None = None


class PairOut(BaseModel):
    id: int
    uid: uuid.UUID
    slug: str
    company_a: CompanyOut
    company_b: CompanyOut
    industry: str | None
    status: str
    monitoring_interval_minutes: int
    alert_threshold: float
    alert_change_threshold: float
    notes: str | None
    created_at: datetime
    archived_at: datetime | None
    watchlist_ids: list[int] = []
    si_v1: LatestScore | None = None
    si_v2: LatestScore | None = None
    direction: str | None = None
    confidence: float | None = None
    explanation: str | None = None
    last_update: datetime | None = None
    is_demo: bool = False


class PairCreate(BaseModel):
    ticker_a: str = Field(min_length=1, max_length=10)
    ticker_b: str = Field(min_length=1, max_length=10)
    name_a: str | None = Field(default=None, max_length=255)
    name_b: str | None = Field(default=None, max_length=255)
    industry: str | None = Field(default=None, max_length=100)
    monitoring_interval_minutes: int = Field(default=60, ge=15, le=10080)
    alert_threshold: float = Field(default=70.0, ge=0, le=100)
    alert_change_threshold: float = Field(default=10.0, ge=0, le=100)
    notes: str | None = Field(default=None, max_length=5000)
    watchlist_ids: list[int] = []

    @field_validator("ticker_a", "ticker_b")
    @classmethod
    def _upper(cls, v: str) -> str:
        return v.strip().upper()


class PairUpdate(BaseModel):
    industry: str | None = Field(default=None, max_length=100)
    status: Literal["active", "disabled", "archived"] | None = None
    monitoring_interval_minutes: int | None = Field(default=None, ge=15, le=10080)
    alert_threshold: float | None = Field(default=None, ge=0, le=100)
    alert_change_threshold: float | None = Field(default=None, ge=0, le=100)
    notes: str | None = Field(default=None, max_length=5000)
    watchlist_ids: list[int] | None = None


class WatchlistIn(BaseModel):
    name: str = Field(min_length=1, max_length=100)
    description: str | None = Field(default=None, max_length=2000)
    pair_ids: list[int] = []


class WatchlistOut(BaseModel):
    id: int
    name: str
    description: str | None
    is_default: bool
    pair_ids: list[int]


# ---------------------------------------------------------------- scores
class ComponentOut(BaseModel):
    model_config = ConfigDict(from_attributes=True)
    component: str
    value: float | None
    available: bool
    weight: float
    explanation: str
    inputs: dict
    sources: list


class ScoreOut(BaseModel):
    id: int
    index_name: str
    model_version: str
    as_of: datetime
    computed_at: datetime
    value: float | None
    status: str
    coverage: float
    direction: str
    confidence: float | None
    explanation: str
    trigger: str
    supporting_event_ids: list[int]
    is_demo: bool
    components: list[ComponentOut] | None = None
    confidence_breakdown: dict | None = None
    direction_detail: dict | None = None


# ---------------------------------------------------------------- news
class EvidenceOut(BaseModel):
    quote: str
    supports: str
    verified: bool


class EventOut(BaseModel):
    id: int
    pair_id: int
    pair_slug: str
    is_material: bool
    event_type: str
    affected_tickers: list[str]
    scope: str
    relative_direction: str
    severity: float
    asymmetry: float
    expected_persistence_days: int
    extraction_confidence: float
    source_reliability: float
    is_new_information: bool
    thread_key: str
    summary: str
    company_exposure: dict
    evidence: list[EvidenceOut]
    evidence_verified_ratio: float
    uncertainties: list[str]
    model_name: str
    prompt_version: str
    analyzed_at: datetime
    changed_index: bool = False
    index_effect: dict | None = None


class DocumentOut(BaseModel):
    id: int
    provider: str
    doc_type: str
    url: str
    title: str
    publisher: str | None
    published_at: datetime
    ingested_at: datetime
    source_reliability: float
    filing_form: str | None
    filing_items: list[str]


class ArticleOut(BaseModel):
    id: int
    headline: str
    first_published_at: datetime
    first_ingested_at: datetime
    document_count: int
    distinct_publishers: int
    tickers: list[str]
    analysis_status: str
    analysis_error: str | None
    is_demo: bool
    documents: list[DocumentOut]
    events: list[EventOut]


# ---------------------------------------------------------------- predictions
class OutcomeOut(BaseModel):
    end_date: date
    divergence: float
    threshold: float
    diverged: bool
    log_return_a: float
    log_return_b: float
    paper_trades: dict
    evaluated_at: datetime
    is_total_return: bool


class PredictionOut(BaseModel):
    id: int
    prediction_date: date
    data_cutoff: datetime
    horizon_days: int
    maturity_date: date
    target_version: str
    target_definition: dict
    si_v1: float | None
    si_v2: float | None
    si_v2_status: str
    si_v2_model_version: str | None
    weights_version: str | None
    confidence: float | None
    direction: str
    baselines: dict
    base_date: date
    price_provider: str
    is_total_return: bool
    is_demo: bool
    status: Literal["pending", "resolved"]
    outcome: OutcomeOut | None = None
    error: float | None = None


# ---------------------------------------------------------------- jobs / settings
class JobRunOut(BaseModel):
    model_config = ConfigDict(from_attributes=True)
    id: int
    job_name: str
    trigger: str
    status: str
    params: dict
    created_at: datetime
    started_at: datetime | None
    finished_at: datetime | None
    duration_ms: int | None
    attempts: int
    error: str | None
    result: dict


class ScheduledJobOut(BaseModel):
    model_config = ConfigDict(from_attributes=True)
    name: str
    description: str
    cron: str
    timezone: str
    trading_days_only: bool
    enabled: bool
    max_attempts: int
    next_run: datetime | None = None
    last_run: JobRunOut | None = None


class RunJobRequest(BaseModel):
    job: str = Field(max_length=60)
    params: dict = {}


class ScheduleUpdate(BaseModel):
    cron: str | None = Field(default=None, max_length=100)
    enabled: bool | None = None


class SettingsUpdate(BaseModel):
    section: str
    value: dict


class UserCreate(BaseModel):
    email: EmailStr
    password: str = Field(min_length=12, max_length=256)
    role: Literal["admin", "viewer"] = "viewer"
