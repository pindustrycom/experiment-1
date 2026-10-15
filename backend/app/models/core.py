"""Users, companies, pairs, watchlists, settings, alerts."""
from __future__ import annotations

import uuid
from datetime import datetime

from sqlalchemy import (
    JSON,
    Boolean,
    CheckConstraint,
    DateTime,
    Float,
    ForeignKey,
    Integer,
    String,
    Text,
    func,
)
from sqlalchemy.dialects.postgresql import JSONB, UUID
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.core.db import Base
from app.models.base import TimestampMixin

JSONType = JSON().with_variant(JSONB(), "postgresql")


class User(TimestampMixin, Base):
    __tablename__ = "users"
    id: Mapped[int] = mapped_column(primary_key=True)
    email: Mapped[str] = mapped_column(String(255), unique=True, index=True)
    password_hash: Mapped[str] = mapped_column(String(255))
    role: Mapped[str] = mapped_column(String(20), default="admin")
    is_active: Mapped[bool] = mapped_column(Boolean, default=True)
    last_login_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    __table_args__ = (CheckConstraint("role in ('admin','viewer')", name="ck_users_role"),)


class Company(TimestampMixin, Base):
    __tablename__ = "companies"
    id: Mapped[int] = mapped_column(primary_key=True)
    ticker: Mapped[str] = mapped_column(String(16), unique=True, index=True)
    name: Mapped[str] = mapped_column(String(255))
    cik: Mapped[str | None] = mapped_column(String(10), index=True)
    sector: Mapped[str | None] = mapped_column(String(100))
    industry: Mapped[str | None] = mapped_column(String(150))
    sic_code: Mapped[str | None] = mapped_column(String(8))
    sic_description: Mapped[str | None] = mapped_column(String(255))
    aliases: Mapped[list] = mapped_column(JSONType, default=list)
    is_benchmark: Mapped[bool] = mapped_column(Boolean, default=False)
    listing_status: Mapped[str] = mapped_column(String(20), default="listed")
    updated_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now(), onupdate=func.now())


class Pair(TimestampMixin, Base):
    """A monitored pair. `pair_key` is the order-independent identity (sorted tickers) and is
    unique, so KO/PEP and PEP/KO can never coexist. `uid` is the stable public identifier."""

    __tablename__ = "pairs"
    id: Mapped[int] = mapped_column(primary_key=True)
    uid: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), unique=True, default=uuid.uuid4)
    slug: Mapped[str] = mapped_column(String(40), unique=True, index=True)
    pair_key: Mapped[str] = mapped_column(String(40), unique=True)
    company_a_id: Mapped[int] = mapped_column(ForeignKey("companies.id", ondelete="RESTRICT"), index=True)
    company_b_id: Mapped[int] = mapped_column(ForeignKey("companies.id", ondelete="RESTRICT"), index=True)
    industry: Mapped[str | None] = mapped_column(String(100), index=True)
    status: Mapped[str] = mapped_column(String(16), default="active", index=True)
    monitoring_interval_minutes: Mapped[int] = mapped_column(Integer, default=60)
    alert_threshold: Mapped[float] = mapped_column(Float, default=70.0)
    alert_change_threshold: Mapped[float] = mapped_column(Float, default=10.0)
    notes: Mapped[str | None] = mapped_column(Text)
    archived_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    updated_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now(), onupdate=func.now())

    company_a: Mapped[Company] = relationship(foreign_keys=[company_a_id], lazy="joined")
    company_b: Mapped[Company] = relationship(foreign_keys=[company_b_id], lazy="joined")

    __table_args__ = (
        CheckConstraint("company_a_id <> company_b_id", name="ck_pairs_distinct"),
        CheckConstraint("status in ('active','disabled','archived')", name="ck_pairs_status"),
        CheckConstraint("monitoring_interval_minutes >= 15", name="ck_pairs_interval"),
    )

    @property
    def tickers(self) -> tuple[str, str]:
        return self.company_a.ticker, self.company_b.ticker


class Watchlist(TimestampMixin, Base):
    __tablename__ = "watchlists"
    id: Mapped[int] = mapped_column(primary_key=True)
    name: Mapped[str] = mapped_column(String(100), unique=True)
    description: Mapped[str | None] = mapped_column(Text)
    is_default: Mapped[bool] = mapped_column(Boolean, default=False)
    pairs: Mapped[list[Pair]] = relationship(secondary="watchlist_pairs", lazy="selectin", order_by="Pair.slug")


class WatchlistPair(Base):
    __tablename__ = "watchlist_pairs"
    watchlist_id: Mapped[int] = mapped_column(ForeignKey("watchlists.id", ondelete="CASCADE"), primary_key=True)
    pair_id: Mapped[int] = mapped_column(ForeignKey("pairs.id", ondelete="CASCADE"), primary_key=True)


class AppSetting(Base):
    __tablename__ = "app_settings"
    key: Mapped[str] = mapped_column(String(100), primary_key=True)
    value: Mapped[dict | list | str | int | float | bool | None] = mapped_column(JSONType)
    updated_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now(), onupdate=func.now())
    updated_by: Mapped[int | None] = mapped_column(ForeignKey("users.id", ondelete="SET NULL"))


class Alert(TimestampMixin, Base):
    __tablename__ = "alerts"
    id: Mapped[int] = mapped_column(primary_key=True)
    pair_id: Mapped[int | None] = mapped_column(ForeignKey("pairs.id", ondelete="RESTRICT"), index=True)
    kind: Mapped[str] = mapped_column(String(40))
    severity: Mapped[str] = mapped_column(String(10), default="info")
    message: Mapped[str] = mapped_column(Text)
    payload: Mapped[dict] = mapped_column(JSONType, default=dict)
    acknowledged_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    dedup_key: Mapped[str | None] = mapped_column(String(200), unique=True)
    __table_args__ = (CheckConstraint("severity in ('info','warning','critical')", name="ck_alerts_severity"),)
