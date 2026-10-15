"""Prices and fundamentals (point-in-time)."""
from __future__ import annotations

from datetime import date, datetime

from sqlalchemy import Boolean, Date, DateTime, Float, ForeignKey, Index, String, UniqueConstraint, func
from sqlalchemy.orm import Mapped, mapped_column

from app.core.db import Base
from app.models.core import JSONType


class PriceObservation(Base):
    """One daily bar per (company, date, provider).

    `adj_close` is the provider's total-return adjusted close when `is_total_return` is true
    (splits and cash dividends reinvested on the ex-date). When a provider only supplies
    split-adjusted prices, `is_total_return` is false and research code reports that.
    """

    __tablename__ = "price_observations"
    id: Mapped[int] = mapped_column(primary_key=True)
    company_id: Mapped[int] = mapped_column(ForeignKey("companies.id", ondelete="RESTRICT"))
    date: Mapped[date] = mapped_column(Date)
    close: Mapped[float] = mapped_column(Float)
    adj_close: Mapped[float] = mapped_column(Float)
    volume: Mapped[float | None] = mapped_column(Float)
    dividend: Mapped[float] = mapped_column(Float, default=0.0)
    split_factor: Mapped[float] = mapped_column(Float, default=1.0)
    provider: Mapped[str] = mapped_column(String(30))
    is_total_return: Mapped[bool] = mapped_column(Boolean)
    is_demo: Mapped[bool] = mapped_column(Boolean, default=False)
    ingested_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())

    __table_args__ = (
        UniqueConstraint("company_id", "date", "provider", name="uq_price_company_date_provider"),
        Index("ix_price_company_date", "company_id", "date"),
    )


class CompanyFundamental(Base):
    """A single reported XBRL fact. `filed_at` is the SEC filing date, the point in time at which
    the value became public; scoring only uses facts with filed_at <= cutoff."""

    __tablename__ = "company_fundamentals"
    id: Mapped[int] = mapped_column(primary_key=True)
    company_id: Mapped[int] = mapped_column(ForeignKey("companies.id", ondelete="RESTRICT"), index=True)
    metric: Mapped[str] = mapped_column(String(60))
    period_start: Mapped[date | None] = mapped_column(Date)
    period_end: Mapped[date] = mapped_column(Date)
    duration_days: Mapped[int | None]
    fiscal_period: Mapped[str | None] = mapped_column(String(4))
    fiscal_year: Mapped[int | None]
    form: Mapped[str | None] = mapped_column(String(20))
    value: Mapped[float] = mapped_column(Float)
    unit: Mapped[str] = mapped_column(String(20))
    filed_at: Mapped[date] = mapped_column(Date)
    accession: Mapped[str] = mapped_column(String(30))
    provider: Mapped[str] = mapped_column(String(30), default="sec_companyfacts")
    is_demo: Mapped[bool] = mapped_column(Boolean, default=False)
    raw: Mapped[dict] = mapped_column(JSONType, default=dict)
    ingested_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())

    __table_args__ = (
        UniqueConstraint("company_id", "metric", "period_end", "duration_days", "accession", name="uq_fundamental_fact"),
        Index("ix_fund_company_metric_filed", "company_id", "metric", "filed_at"),
    )
