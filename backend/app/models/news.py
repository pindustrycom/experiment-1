"""Source documents (verified facts), story clusters, and AI interpretations (kept separate)."""
from __future__ import annotations

from datetime import datetime

from sqlalchemy import (
    Boolean,
    CheckConstraint,
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


class NewsArticle(Base):
    """A deduplicated story: syndicated copies / repeated coverage of the same underlying item
    collapse into one NewsArticle so a single story cannot increase a score repeatedly."""

    __tablename__ = "news_articles"
    id: Mapped[int] = mapped_column(primary_key=True)
    cluster_key: Mapped[str] = mapped_column(String(64), unique=True)
    headline: Mapped[str] = mapped_column(Text)
    first_published_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), index=True)
    first_ingested_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())
    last_seen_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())
    document_count: Mapped[int] = mapped_column(Integer, default=1)
    distinct_publishers: Mapped[int] = mapped_column(Integer, default=1)
    tickers: Mapped[list] = mapped_column(JSONType, default=list)
    analysis_status: Mapped[str] = mapped_column(String(20), default="pending", index=True)
    analysis_error: Mapped[str | None] = mapped_column(Text)
    is_demo: Mapped[bool] = mapped_column(Boolean, default=False)

    documents: Mapped[list[SourceDocument]] = relationship(back_populates="article", lazy="selectin", order_by="SourceDocument.published_at")

    __table_args__ = (
        CheckConstraint(
            "analysis_status in ('pending','analyzed','skipped','failed','budget_blocked','not_configured')",
            name="ck_news_analysis_status",
        ),
    )


class SourceDocument(Base):
    __tablename__ = "source_documents"
    id: Mapped[int] = mapped_column(primary_key=True)
    article_id: Mapped[int | None] = mapped_column(ForeignKey("news_articles.id", ondelete="RESTRICT"), index=True)
    provider: Mapped[str] = mapped_column(String(30))
    doc_type: Mapped[str] = mapped_column(String(20))  # news | filing | press_release | rss
    url: Mapped[str] = mapped_column(Text)
    url_hash: Mapped[str] = mapped_column(String(64), unique=True)
    title: Mapped[str] = mapped_column(Text)
    publisher: Mapped[str | None] = mapped_column(String(200))
    published_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), index=True)
    ingested_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now(), index=True)
    content_text: Mapped[str | None] = mapped_column(Text)
    content_hash: Mapped[str | None] = mapped_column(String(64), index=True)
    tickers: Mapped[list] = mapped_column(JSONType, default=list)
    filing_form: Mapped[str | None] = mapped_column(String(20))
    filing_items: Mapped[list] = mapped_column(JSONType, default=list)
    accession: Mapped[str | None] = mapped_column(String(30))
    source_reliability: Mapped[float] = mapped_column(Float)
    is_demo: Mapped[bool] = mapped_column(Boolean, default=False)
    raw: Mapped[dict] = mapped_column(JSONType, default=dict)

    article: Mapped[NewsArticle | None] = relationship(back_populates="documents")

    __table_args__ = (
        CheckConstraint("doc_type in ('news','filing','press_release','rss')", name="ck_doc_type"),
        CheckConstraint("source_reliability >= 0 and source_reliability <= 1", name="ck_doc_reliability"),
    )


class ExtractedEvent(Base):
    """Claude's structured interpretation of one story for one pair. Numeric fields here are the
    model's *estimates*; the index is computed from them by deterministic, versioned rules
    (app.scoring.si_v2), and only when the quoted evidence was verified against the source text."""

    __tablename__ = "extracted_events"
    id: Mapped[int] = mapped_column(primary_key=True)
    article_id: Mapped[int] = mapped_column(ForeignKey("news_articles.id", ondelete="RESTRICT"), index=True)
    pair_id: Mapped[int] = mapped_column(ForeignKey("pairs.id", ondelete="RESTRICT"), index=True)
    document_id: Mapped[int] = mapped_column(ForeignKey("source_documents.id", ondelete="RESTRICT"))
    is_material: Mapped[bool] = mapped_column(Boolean)
    event_type: Mapped[str] = mapped_column(String(40), index=True)
    affected_tickers: Mapped[list] = mapped_column(JSONType, default=list)
    scope: Mapped[str] = mapped_column(String(20))  # company_specific | both_companies | industry_wide | unrelated
    relative_direction: Mapped[str] = mapped_column(String(30))
    severity: Mapped[float] = mapped_column(Float)
    asymmetry: Mapped[float] = mapped_column(Float)
    expected_persistence_days: Mapped[int] = mapped_column(Integer)
    extraction_confidence: Mapped[float] = mapped_column(Float)
    source_reliability: Mapped[float] = mapped_column(Float)
    is_new_information: Mapped[bool] = mapped_column(Boolean, default=True)
    related_event_id: Mapped[int | None] = mapped_column(ForeignKey("extracted_events.id", ondelete="SET NULL"))
    thread_key: Mapped[str] = mapped_column(String(64), index=True)
    summary: Mapped[str] = mapped_column(Text)
    company_exposure: Mapped[dict] = mapped_column(JSONType, default=dict)
    evidence: Mapped[list] = mapped_column(JSONType, default=list)  # [{quote, supports, verified}]
    evidence_verified_ratio: Mapped[float] = mapped_column(Float, default=0.0)
    uncertainties: Mapped[list] = mapped_column(JSONType, default=list)
    source_urls: Mapped[list] = mapped_column(JSONType, default=list)
    published_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), index=True)
    analyzed_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())
    model_name: Mapped[str] = mapped_column(String(80))
    prompt_version: Mapped[str] = mapped_column(String(20))
    schema_version: Mapped[str] = mapped_column(String(20))
    raw_response: Mapped[dict] = mapped_column(JSONType, default=dict)
    is_demo: Mapped[bool] = mapped_column(Boolean, default=False)

    __table_args__ = (
        UniqueConstraint("article_id", "pair_id", "prompt_version", name="uq_event_article_pair_prompt"),
        CheckConstraint("severity >= 0 and severity <= 1", name="ck_event_severity"),
        CheckConstraint("asymmetry >= 0 and asymmetry <= 1", name="ck_event_asymmetry"),
        CheckConstraint("extraction_confidence >= 0 and extraction_confidence <= 1", name="ck_event_conf"),
        CheckConstraint("expected_persistence_days >= 0", name="ck_event_persistence"),
        Index("ix_event_pair_published", "pair_id", "published_at"),
    )


class DocumentCompany(Base):
    __tablename__ = "document_companies"
    document_id: Mapped[int] = mapped_column(ForeignKey("source_documents.id", ondelete="CASCADE"), primary_key=True)
    company_id: Mapped[int] = mapped_column(ForeignKey("companies.id", ondelete="CASCADE"), primary_key=True)


class NewsFetchLog(Base):
    """Per-provider, per-ticker coverage record, used to decide whether 'no news' is a measured
    zero or missing data."""

    __tablename__ = "news_fetch_log"
    id: Mapped[int] = mapped_column(primary_key=True)
    provider: Mapped[str] = mapped_column(String(30))
    company_id: Mapped[int] = mapped_column(ForeignKey("companies.id", ondelete="CASCADE"))
    window_start: Mapped[datetime] = mapped_column(DateTime(timezone=True))
    window_end: Mapped[datetime] = mapped_column(DateTime(timezone=True))
    fetched_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())
    success: Mapped[bool] = mapped_column(Boolean)
    documents_found: Mapped[int] = mapped_column(Integer, default=0)
    error: Mapped[str | None] = mapped_column(Text)
    is_demo: Mapped[bool] = mapped_column(Boolean, default=False)
    __table_args__ = (Index("ix_newsfetch_company_end", "company_id", "window_end"),)
