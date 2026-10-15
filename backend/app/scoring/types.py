from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date, datetime

import pandas as pd


@dataclass
class ComponentResult:
    code: str
    value: float | None
    available: bool
    explanation: str
    inputs: dict = field(default_factory=dict)
    sources: list = field(default_factory=list)
    signed: float | None = None  # >0 favours company A (used for directional bias only)

    @classmethod
    def unavailable(cls, code: str, reason: str, **inputs) -> ComponentResult:
        return cls(code=code, value=None, available=False, explanation=f"Unavailable: {reason}", inputs=inputs)


@dataclass
class MarketData:
    """Adjusted closes for A and B (columns 'a','b') and optionally the benchmark, all <= cutoff."""

    prices: pd.DataFrame
    benchmark: pd.Series | None = None
    is_total_return: bool = True
    provider: str = ""

    @property
    def last_date(self) -> date | None:
        return None if self.prices.empty else pd.Timestamp(self.prices.index[-1]).date()


@dataclass
class EventEvidence:
    id: int
    thread_key: str
    event_type: str
    scope: str
    relative_direction: str
    affected_tickers: list[str]
    severity: float
    asymmetry: float
    persistence_days: int
    source_reliability: float
    extraction_confidence: float
    evidence_verified_ratio: float
    published_at: datetime
    ingested_at: datetime
    is_material: bool
    distinct_publishers: int = 1
    official_source: bool = False
    summary: str = ""
    document_ids: list[int] = field(default_factory=list)


@dataclass
class NewsCoverage:
    """Whether news collection actually ran for each company before the cutoff."""

    fetch_days_ok_a: int  # distinct days in the last `days` with >=1 successful fetch
    fetch_days_ok_b: int
    days: int
    relevant_clusters: int  # story clusters mentioning either company in the lookback
    analyzed_clusters: int
    last_fetch_at: datetime | None = None


@dataclass
class Fact:
    metric: str
    period_end: date
    duration_days: int | None
    value: float
    filed_at: date
    accession: str
    form: str | None = None


@dataclass
class CompanyProfile:
    ticker: str
    sic_code: str | None = None
    facts: list[Fact] = field(default_factory=list)


@dataclass
class PairInputs:
    ticker_a: str
    ticker_b: str
    cutoff: datetime
    market: MarketData
    events: list[EventEvidence] | None  # None: news pipeline not configured / no data at all
    coverage: NewsCoverage | None
    profile_a: CompanyProfile
    profile_b: CompanyProfile
