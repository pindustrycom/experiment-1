"""Model versions, index scores, components, predictions, outcomes, backtests (append-only)."""
from __future__ import annotations

from datetime import date, datetime

from sqlalchemy import (
    Boolean,
    CheckConstraint,
    Date,
    DateTime,
    Float,
    ForeignKey,
    Index,
    Integer,
    String,
    Text,
    UniqueConstraint,
    func,
)
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.core.db import Base
from app.models.core import JSONType


class ModelVersion(Base):
    """Immutable description of an index definition (formula + weights + parameters)."""

    __tablename__ = "model_versions"
    id: Mapped[int] = mapped_column(primary_key=True)
    index_name: Mapped[str] = mapped_column(String(20))  # si_v1 | si_v2
    version: Mapped[str] = mapped_column(String(40))
    weights: Mapped[dict] = mapped_column(JSONType)
    params: Mapped[dict] = mapped_column(JSONType)
    description: Mapped[str] = mapped_column(Text)
    content_hash: Mapped[str] = mapped_column(String(64))
    is_active: Mapped[bool] = mapped_column(Boolean, default=False)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())
    __table_args__ = (
        UniqueConstraint("index_name", "version", name="uq_model_version"),
        UniqueConstraint("index_name", "content_hash", name="uq_model_content"),
    )


class IndexScore(Base):
    __tablename__ = "index_scores"
    id: Mapped[int] = mapped_column(primary_key=True)
    pair_id: Mapped[int] = mapped_column(ForeignKey("pairs.id", ondelete="RESTRICT"))
    model_version_id: Mapped[int] = mapped_column(ForeignKey("model_versions.id", ondelete="RESTRICT"))
    index_name: Mapped[str] = mapped_column(String(20))
    as_of: Mapped[datetime] = mapped_column(DateTime(timezone=True))  # data cutoff
    computed_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())
    value: Mapped[float | None] = mapped_column(Float)
    status: Mapped[str] = mapped_column(String(20))  # complete | provisional | unavailable
    coverage: Mapped[float] = mapped_column(Float)
    direction: Mapped[str] = mapped_column(String(30))
    direction_detail: Mapped[dict] = mapped_column(JSONType, default=dict)
    confidence: Mapped[float | None] = mapped_column(Float)
    confidence_breakdown: Mapped[dict] = mapped_column(JSONType, default=dict)
    explanation: Mapped[str] = mapped_column(Text, default="")
    trigger: Mapped[str] = mapped_column(String(20))
    supporting_event_ids: Mapped[list] = mapped_column(JSONType, default=list)
    inputs_digest: Mapped[str] = mapped_column(String(64))
    is_demo: Mapped[bool] = mapped_column(Boolean, default=False)

    components: Mapped[list[ComponentCalculation]] = relationship(lazy="selectin", order_by="ComponentCalculation.component")

    __table_args__ = (
        CheckConstraint("status in ('complete','provisional','unavailable')", name="ck_score_status"),
        CheckConstraint("value is null or (value >= 0 and value <= 100)", name="ck_score_range"),
        UniqueConstraint("pair_id", "model_version_id", "as_of", "inputs_digest", name="uq_score_identity"),
        Index("ix_score_pair_index_asof", "pair_id", "index_name", "as_of"),
    )


class ComponentCalculation(Base):
    __tablename__ = "si_component_calculations"
    id: Mapped[int] = mapped_column(primary_key=True)
    score_id: Mapped[int] = mapped_column(ForeignKey("index_scores.id", ondelete="RESTRICT"), index=True)
    component: Mapped[str] = mapped_column(String(4))
    value: Mapped[float | None] = mapped_column(Float)
    available: Mapped[bool] = mapped_column(Boolean)
    weight: Mapped[float] = mapped_column(Float)
    inputs: Mapped[dict] = mapped_column(JSONType, default=dict)
    explanation: Mapped[str] = mapped_column(Text, default="")
    sources: Mapped[list] = mapped_column(JSONType, default=list)
    __table_args__ = (
        UniqueConstraint("score_id", "component", name="uq_component_score"),
        CheckConstraint("value is null or (value >= 0 and value <= 1)", name="ck_component_range"),
    )


class PredictionSnapshot(Base):
    """Immutable forward prediction. A database trigger rejects UPDATE and DELETE."""

    __tablename__ = "prediction_snapshots"
    id: Mapped[int] = mapped_column(primary_key=True)
    pair_id: Mapped[int] = mapped_column(ForeignKey("pairs.id", ondelete="RESTRICT"), index=True)
    prediction_date: Mapped[date] = mapped_column(Date, index=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())
    data_cutoff: Mapped[datetime] = mapped_column(DateTime(timezone=True))
    horizon_days: Mapped[int] = mapped_column(Integer)
    target_version: Mapped[str] = mapped_column(String(40))
    target_definition: Mapped[dict] = mapped_column(JSONType)
    si_v1: Mapped[float | None] = mapped_column(Float)
    si_v2: Mapped[float | None] = mapped_column(Float)
    si_v2_status: Mapped[str] = mapped_column(String(20))
    si_v1_score_id: Mapped[int | None] = mapped_column(ForeignKey("index_scores.id", ondelete="RESTRICT"))
    si_v2_score_id: Mapped[int | None] = mapped_column(ForeignKey("index_scores.id", ondelete="RESTRICT"))
    si_v1_model_version: Mapped[str | None] = mapped_column(String(40))
    si_v2_model_version: Mapped[str | None] = mapped_column(String(40))
    weights_version: Mapped[str | None] = mapped_column(String(40))
    components: Mapped[dict] = mapped_column(JSONType)
    baselines: Mapped[dict] = mapped_column(JSONType)  # correlation-only, previous-return-gap
    confidence: Mapped[float | None] = mapped_column(Float)
    direction: Mapped[str] = mapped_column(String(30))
    supporting_event_ids: Mapped[list] = mapped_column(JSONType, default=list)
    supporting_document_ids: Mapped[list] = mapped_column(JSONType, default=list)
    market_info: Mapped[dict] = mapped_column(JSONType)
    base_date: Mapped[date] = mapped_column(Date)  # last trading day whose close is t
    base_price_a: Mapped[float] = mapped_column(Float)
    base_price_b: Mapped[float] = mapped_column(Float)
    price_provider: Mapped[str] = mapped_column(String(30))
    is_total_return: Mapped[bool] = mapped_column(Boolean)
    is_demo: Mapped[bool] = mapped_column(Boolean, default=False)

    __table_args__ = (
        UniqueConstraint("pair_id", "prediction_date", "horizon_days", "target_version", name="uq_prediction_identity"),
        CheckConstraint("horizon_days > 0", name="ck_prediction_horizon"),
    )


class RealizedOutcome(Base):
    """Immutable outcome for a matured prediction (trigger rejects UPDATE/DELETE)."""

    __tablename__ = "realized_outcomes"
    id: Mapped[int] = mapped_column(primary_key=True)
    snapshot_id: Mapped[int] = mapped_column(ForeignKey("prediction_snapshots.id", ondelete="RESTRICT"), unique=True)
    evaluated_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())
    end_date: Mapped[date] = mapped_column(Date)
    end_price_a: Mapped[float] = mapped_column(Float)
    end_price_b: Mapped[float] = mapped_column(Float)
    log_return_a: Mapped[float] = mapped_column(Float)
    log_return_b: Mapped[float] = mapped_column(Float)
    divergence: Mapped[float] = mapped_column(Float)  # |ln(PA_t+H/PA_t) - ln(PB_t+H/PB_t)|
    threshold: Mapped[float] = mapped_column(Float)
    diverged: Mapped[bool] = mapped_column(Boolean)
    paper_trades: Mapped[dict] = mapped_column(JSONType, default=dict)
    price_provider: Mapped[str] = mapped_column(String(30))
    is_total_return: Mapped[bool] = mapped_column(Boolean)
    is_demo: Mapped[bool] = mapped_column(Boolean, default=False)


class BacktestRun(Base):
    __tablename__ = "backtest_runs"
    id: Mapped[int] = mapped_column(primary_key=True)
    kind: Mapped[str] = mapped_column(String(30))  # reported_historical | walk_forward
    label: Mapped[str] = mapped_column(String(200))
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())
    params: Mapped[dict] = mapped_column(JSONType, default=dict)
    metrics: Mapped[dict] = mapped_column(JSONType, default=dict)
    observations: Mapped[list] = mapped_column(JSONType, default=list)
    limitations: Mapped[list] = mapped_column(JSONType, default=list)
    status: Mapped[str] = mapped_column(String(20), default="complete")
    is_demo: Mapped[bool] = mapped_column(Boolean, default=False)
