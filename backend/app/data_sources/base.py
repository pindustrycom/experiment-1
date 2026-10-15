from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date, datetime


class ProviderError(Exception):
    """A provider call failed after retries (network, HTTP error, malformed payload)."""

    def __init__(self, provider: str, message: str, status: int | None = None, retryable: bool = True):
        super().__init__(f"{provider}: {message}")
        self.provider, self.status, self.retryable = provider, status, retryable


class ProviderNotConfigured(ProviderError):
    def __init__(self, provider: str, what: str):
        super().__init__(provider, f"not configured ({what})", retryable=False)


@dataclass
class PriceBar:
    date: date
    close: float
    adj_close: float
    volume: float | None = None
    dividend: float = 0.0
    split_factor: float = 1.0


@dataclass
class RawDocument:
    provider: str
    doc_type: str  # news | filing | press_release | rss
    url: str
    title: str
    published_at: datetime
    publisher: str | None = None
    content_text: str | None = None
    tickers: list[str] = field(default_factory=list)
    filing_form: str | None = None
    filing_items: list[str] = field(default_factory=list)
    accession: str | None = None
    raw: dict = field(default_factory=dict)


@dataclass
class FactRecord:
    metric: str
    period_start: date | None
    period_end: date
    value: float
    unit: str
    filed_at: date
    accession: str
    form: str | None
    fiscal_year: int | None
    fiscal_period: str | None
    tag: str


@dataclass
class CompanyReference:
    ticker: str
    cik: str | None
    name: str | None
    sic_code: str | None = None
    sic_description: str | None = None
