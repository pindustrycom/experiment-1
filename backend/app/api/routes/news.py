from __future__ import annotations

from datetime import datetime

from fastapi import APIRouter, Depends, Query
from sqlalchemy import func, select
from sqlalchemy.orm import Session

from app.api.deps import current_user
from app.core.config import get_settings
from app.core.db import get_db
from app.models import ComponentCalculation, ExtractedEvent, IndexScore, NewsArticle, Pair, SourceDocument, User
from app.schemas.api import ArticleOut, DocumentOut, EventOut, Page

router = APIRouter(prefix="/news", tags=["news"])


def index_effect(db: Session, e: ExtractedEvent) -> dict | None:
    """Effect of an event on SI-v2: its weight in the first score computed after it was analysed,
    and the SI-v2 change versus the previous score."""
    first = db.scalar(select(IndexScore).where(IndexScore.pair_id == e.pair_id, IndexScore.index_name == "si_v2", IndexScore.as_of >= e.analyzed_at)
                      .order_by(IndexScore.as_of, IndexScore.id).limit(1))
    if first is None:
        return None
    prev = db.scalar(select(IndexScore).where(IndexScore.pair_id == e.pair_id, IndexScore.index_name == "si_v2", IndexScore.as_of < e.analyzed_at)
                     .order_by(IndexScore.as_of.desc(), IndexScore.id.desc()).limit(1))
    n = db.scalar(select(ComponentCalculation).where(ComponentCalculation.score_id == first.id, ComponentCalculation.component == "N"))
    weight = None
    if n is not None:
        for c in (n.inputs or {}).get("contributions", []):
            if c.get("event_id") == e.id:
                weight = c.get("weight")
    delta = None if (prev is None or prev.value is None or first.value is None) else round(first.value - prev.value, 3)
    return {"score_id": first.id, "score_as_of": first.as_of, "n_weight": weight, "si_v2_before": prev.value if prev else None,
            "si_v2_after": first.value, "si_v2_delta": delta, "counted": e.id in (first.supporting_event_ids or [])}


def event_out(db: Session, e: ExtractedEvent, slug: str) -> EventOut:
    eff = index_effect(db, e)
    return EventOut(
        id=e.id, pair_id=e.pair_id, pair_slug=slug, is_material=e.is_material, event_type=e.event_type, affected_tickers=e.affected_tickers,
        scope=e.scope, relative_direction=e.relative_direction, severity=e.severity, asymmetry=e.asymmetry,
        expected_persistence_days=e.expected_persistence_days, extraction_confidence=e.extraction_confidence, source_reliability=e.source_reliability,
        is_new_information=e.is_new_information, thread_key=e.thread_key, summary=e.summary, company_exposure=e.company_exposure or {},
        evidence=[{"quote": q.get("quote", ""), "supports": q.get("supports", ""), "verified": bool(q.get("verified"))} for q in (e.evidence or [])],
        evidence_verified_ratio=e.evidence_verified_ratio, uncertainties=e.uncertainties or [], model_name=e.model_name, prompt_version=e.prompt_version,
        analyzed_at=e.analyzed_at, changed_index=bool(eff and eff["counted"] and (eff["n_weight"] or 0) > 0), index_effect=eff,
    )


def article_page(db: Session, page: int, page_size: int, pair: Pair | None = None, ticker: str | None = None, event_type: str | None = None,
                 provider: str | None = None, material_only: bool = False, changed_only: bool = False, since: datetime | None = None,
                 until: datetime | None = None, status: str | None = None) -> Page[ArticleOut]:
    q = select(NewsArticle)
    if not get_settings().is_demo:
        q = q.where(NewsArticle.is_demo.is_(False))
    tick = None
    if pair is not None:
        tick = [pair.company_a.ticker, pair.company_b.ticker]
    elif ticker:
        tick = [ticker.upper()]
    if tick:
        from sqlalchemy import or_
        q = q.where(or_(*[NewsArticle.tickers.op("@>")(func.jsonb_build_array(t)) for t in tick]))
    if event_type or material_only or changed_only:
        ev = select(ExtractedEvent.article_id)
        if event_type:
            ev = ev.where(ExtractedEvent.event_type == event_type)
        if material_only or changed_only:
            ev = ev.where(ExtractedEvent.is_material.is_(True))
        if pair is not None:
            ev = ev.where(ExtractedEvent.pair_id == pair.id)
        q = q.where(NewsArticle.id.in_(ev))
    if provider:
        q = q.where(NewsArticle.id.in_(select(SourceDocument.article_id).where(SourceDocument.provider == provider)))
    if since:
        q = q.where(NewsArticle.first_published_at >= since)
    if until:
        q = q.where(NewsArticle.first_published_at <= until)
    if status:
        q = q.where(NewsArticle.analysis_status == status)
    total = db.scalar(select(func.count()).select_from(q.subquery()))
    arts = db.scalars(q.order_by(NewsArticle.first_published_at.desc()).offset((page - 1) * page_size).limit(page_size)).all()
    slugs = dict(db.execute(select(Pair.id, Pair.slug)).all())
    items = []
    for a in arts:
        evq = select(ExtractedEvent).where(ExtractedEvent.article_id == a.id)
        if pair is not None:
            evq = evq.where(ExtractedEvent.pair_id == pair.id)
        events = [event_out(db, e, slugs.get(e.pair_id, "?")) for e in db.scalars(evq.order_by(ExtractedEvent.id)).all()]
        if changed_only and not any(e.changed_index for e in events):
            continue
        items.append(ArticleOut(
            id=a.id, headline=a.headline, first_published_at=a.first_published_at, first_ingested_at=a.first_ingested_at,
            document_count=a.document_count, distinct_publishers=a.distinct_publishers, tickers=a.tickers, analysis_status=a.analysis_status,
            analysis_error=a.analysis_error, is_demo=a.is_demo,
            documents=[DocumentOut(id=d.id, provider=d.provider, doc_type=d.doc_type, url=d.url, title=d.title, publisher=d.publisher,
                                   published_at=d.published_at, ingested_at=d.ingested_at, source_reliability=d.source_reliability,
                                   filing_form=d.filing_form, filing_items=d.filing_items or []) for d in a.documents],
            events=events,
        ))
    return Page(items=items, total=total, page=page, page_size=page_size)


@router.get("", response_model=Page[ArticleOut])
def list_news(db: Session = Depends(get_db), _: User = Depends(current_user), page: int = Query(1, ge=1), page_size: int = Query(25, ge=1, le=100),
              pair: str | None = None, ticker: str | None = Query(None, max_length=10), event_type: str | None = None, provider: str | None = None,
              material_only: bool = False, changed_only: bool = False, since: datetime | None = None, until: datetime | None = None,
              status: str | None = None):
    p = None
    if pair:
        from app.api.routes.pairs import resolve_pair
        p = resolve_pair(db, pair)
    return article_page(db, page, page_size, p, ticker, event_type, provider, material_only, changed_only, since, until, status)
