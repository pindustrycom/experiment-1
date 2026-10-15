from __future__ import annotations

import uuid
from datetime import UTC, datetime, timedelta
from typing import Literal

import numpy as np
from fastapi import APIRouter, Depends, HTTPException, Query, status
from sqlalchemy import func, or_, select
from sqlalchemy.orm import Session

from app.api.deps import current_user, require_admin
from app.api.serializers import pair_out, prediction_out, score_out
from app.core.config import get_settings
from app.core.db import get_db
from app.models import (
    Company,
    ExtractedEvent,
    IndexScore,
    ModelVersion,
    NewsArticle,
    Pair,
    PredictionSnapshot,
    RealizedOutcome,
    User,
    Watchlist,
    WatchlistPair,
)
from app.research import stats
from app.research.market_calendar import last_completed_session, trading_days_between
from app.schemas.api import Page, PairCreate, PairOut, PairUpdate, PredictionOut, ScoreOut
from app.services import prices_service
from app.services.reference import PairExists, create_pair

router = APIRouter(prefix="/pairs", tags=["pairs"])


def resolve_pair(db: Session, ident: str) -> Pair:
    p = None
    if ident.isdigit():
        p = db.get(Pair, int(ident))
    else:
        try:
            p = db.scalar(select(Pair).where(Pair.uid == uuid.UUID(ident)))
        except ValueError:
            p = db.scalar(select(Pair).where(func.upper(Pair.slug) == ident.upper()))
            if p is None and "-" in ident:
                a, b = ident.upper().split("-", 1)
                p = db.scalar(select(Pair).where(Pair.pair_key == "|".join(sorted([a, b]))))
    if p is None:
        raise HTTPException(status.HTTP_404_NOT_FOUND, f"pair '{ident}' not found")
    return p


SORTS = {"slug", "si_v1", "si_v2", "change_7d", "confidence", "last_update"}


@router.get("", response_model=Page[PairOut])
def list_pairs(
    db: Session = Depends(get_db), _: User = Depends(current_user),
    status_: Literal["active", "disabled", "archived", "all", "monitored"] = Query("monitored", alias="status"),
    industry: str | None = None, watchlist_id: int | None = None, q: str | None = Query(None, max_length=50),
    sort: str = "si_v2", order: Literal["asc", "desc"] = "desc", page: int = Query(1, ge=1), page_size: int = Query(100, ge=1, le=500),
):
    stmt = select(Pair)
    if status_ == "monitored":
        stmt = stmt.where(Pair.status.in_(("active", "disabled")))
    elif status_ != "all":
        stmt = stmt.where(Pair.status == status_)
    if industry:
        stmt = stmt.where(Pair.industry == industry)
    if watchlist_id:
        stmt = stmt.join(WatchlistPair, WatchlistPair.pair_id == Pair.id).where(WatchlistPair.watchlist_id == watchlist_id)
    if q:
        like = f"%{q.upper()}%"
        ca, cb = Company.__table__.alias(), Company.__table__.alias()
        stmt = stmt.join(ca, ca.c.id == Pair.company_a_id).join(cb, cb.c.id == Pair.company_b_id).where(
            or_(func.upper(Pair.slug).like(like), func.upper(ca.c.name).like(like), func.upper(cb.c.name).like(like), func.upper(Pair.industry).like(like)))
    pairs = db.scalars(stmt).unique().all()
    items = pair_out(db, list(pairs))
    if sort not in SORTS:
        raise HTTPException(422, f"sort must be one of {sorted(SORTS)}")

    def key(p: PairOut):
        v = {"slug": p.slug, "si_v1": p.si_v1.value if p.si_v1 else None, "si_v2": p.si_v2.value if p.si_v2 else None,
             "change_7d": p.si_v2.change_7d if p.si_v2 else None, "confidence": p.confidence, "last_update": p.last_update}[sort]
        return (v is None, v if v is not None else 0)
    items.sort(key=key, reverse=False)
    if order == "desc":
        nonnull = [i for i in items if key(i)[0] is False][::-1]
        items = nonnull + [i for i in items if key(i)[0]]
    total = len(items)
    return Page(items=items[(page - 1) * page_size: page * page_size], total=total, page=page, page_size=page_size)


@router.post("", response_model=PairOut, status_code=201)
def add_pair(body: PairCreate, db: Session = Depends(get_db), _: User = Depends(require_admin)):
    try:
        from app.services.reference import get_or_create_company
        if body.name_a:
            get_or_create_company(db, body.ticker_a, body.name_a)
        if body.name_b:
            get_or_create_company(db, body.ticker_b, body.name_b)
        p = create_pair(db, body.ticker_a, body.ticker_b, body.industry, monitoring_interval_minutes=body.monitoring_interval_minutes,
                        alert_threshold=body.alert_threshold, alert_change_threshold=body.alert_change_threshold, notes=body.notes)
    except PairExists as e:
        db.rollback()
        raise HTTPException(status.HTTP_409_CONFLICT, {"message": str(e), "existing": e.pair.slug}) from e
    except ValueError as e:
        db.rollback()
        raise HTTPException(422, str(e)) from e
    for wid in body.watchlist_ids:
        if db.get(Watchlist, wid):
            db.add(WatchlistPair(watchlist_id=wid, pair_id=p.id))
    db.commit()
    return pair_out(db, [p])[0]


@router.get("/{pair_id}", response_model=PairOut)
def get_pair(pair_id: str, db: Session = Depends(get_db), _: User = Depends(current_user)):
    return pair_out(db, [resolve_pair(db, pair_id)])[0]


@router.patch("/{pair_id}", response_model=PairOut)
def update_pair(pair_id: str, body: PairUpdate, db: Session = Depends(get_db), _: User = Depends(require_admin)):
    p = resolve_pair(db, pair_id)
    data = body.model_dump(exclude_unset=True)
    wl = data.pop("watchlist_ids", None)
    for k, v in data.items():
        setattr(p, k, v)
    if "status" in data:
        p.archived_at = datetime.now(UTC) if data["status"] == "archived" else None
    if wl is not None:
        db.query(WatchlistPair).filter(WatchlistPair.pair_id == p.id).delete()
        for wid in wl:
            if db.get(Watchlist, wid):
                db.add(WatchlistPair(watchlist_id=wid, pair_id=p.id))
    db.commit()
    return pair_out(db, [p])[0]


@router.delete("/{pair_id}", status_code=200)
def remove_pair(pair_id: str, hard: bool = False, db: Session = Depends(get_db), _: User = Depends(require_admin)):
    """Removing a pair archives it: scores and predictions are kept forever. A hard delete is only
    allowed for pairs that never produced any score, event or prediction."""
    p = resolve_pair(db, pair_id)
    has_history = any(db.scalar(select(func.count()).select_from(m).where(m.pair_id == p.id)) for m in (IndexScore, PredictionSnapshot, ExtractedEvent))
    if hard:
        if has_history:
            raise HTTPException(status.HTTP_409_CONFLICT, "pair has historical scores/predictions; archive it instead")
        db.query(WatchlistPair).filter(WatchlistPair.pair_id == p.id).delete()
        db.delete(p)
        db.commit()
        return {"deleted": True}
    p.status, p.archived_at = "archived", datetime.now(UTC)
    db.commit()
    return {"archived": True, "slug": p.slug}


@router.get("/{pair_id}/scores", response_model=list[ScoreOut])
def pair_scores(pair_id: str, db: Session = Depends(get_db), _: User = Depends(current_user),
                index: Literal["si_v1", "si_v2", "all"] = "all", days: int = Query(365, ge=1, le=7300), detail: bool = False,
                limit: int = Query(2000, ge=1, le=10000)):
    p = resolve_pair(db, pair_id)
    q = select(IndexScore).where(IndexScore.pair_id == p.id, IndexScore.as_of >= datetime.now(UTC) - timedelta(days=days))
    if index != "all":
        q = q.where(IndexScore.index_name == index)
    rows = db.scalars(q.order_by(IndexScore.as_of, IndexScore.id).limit(limit)).all()
    versions = dict(db.execute(select(ModelVersion.id, ModelVersion.version)).all())
    return [score_out(s, versions[s.model_version_id], detail) for s in rows]


@router.get("/{pair_id}/scores/{score_id}", response_model=ScoreOut)
def score_detail(pair_id: str, score_id: int, db: Session = Depends(get_db), _: User = Depends(current_user)):
    p = resolve_pair(db, pair_id)
    s = db.get(IndexScore, score_id)
    if s is None or s.pair_id != p.id:
        raise HTTPException(404, "score not found")
    return score_out(s, db.get(ModelVersion, s.model_version_id).version, detail=True)


def _series(s):
    return [{"date": ix.strftime("%Y-%m-%d"), "value": None if (v is None or not np.isfinite(v)) else round(float(v), 6)} for ix, v in s.items()]


@router.get("/{pair_id}/prices")
def pair_prices(pair_id: str, db: Session = Depends(get_db), _: User = Depends(current_user), years: float = Query(3, gt=0, le=30)):
    """Price-relationship analytics. Hedge ratios are fitted only on trailing formation windows and
    z-scores are out-of-sample; cointegration is tested on log-price levels (Engle-Granger)."""
    p = resolve_pair(db, pair_id)
    now = datetime.now(UTC)
    bench = db.scalar(select(Company).where(Company.is_benchmark.is_(True)))
    md = prices_service.pair_market_data(db, p.company_a, p.company_b, now, bench, demo=None if get_settings().is_demo else False)
    df = md.prices
    if df.empty:
        return {"available": False, "reason": "no price data for both companies from a common provider", "provider": md.provider}
    rets = stats.log_returns(df)
    start = df.index[-1] - timedelta(days=int(365 * years))
    view = df[df.index >= start]
    norm = view / view.iloc[0] * 100
    rel = np.log(view["a"] / view["a"].iloc[0]) - np.log(view["b"] / view["b"].iloc[0])
    rc63 = stats.rolling_corr_series(rets, 63)
    rc252 = stats.rolling_corr_series(rets, 252)
    z = stats.rolling_oos_zscore(df, 252, 21)
    if not z.empty:
        z = z[z["date"] >= start]
    coint = {}
    for w in (252, 504):
        if len(df) >= w:
            seg = df.tail(w)
            fit = stats.fit_hedge_ratio(np.log(seg))
            coint[str(w)] = {"p_value": stats.engle_granger_pvalue(seg), "hedge_ratio": fit.beta if fit else None,
                             "window_start": str(seg.index[0].date()), "window_end": str(seg.index[-1].date())}
    last = df.index[-1].date()
    expected = last_completed_session(now)
    return {
        "available": True, "provider": md.provider, "is_total_return": md.is_total_return, "is_demo": md.provider == "demo",
        "ticker_a": p.company_a.ticker, "ticker_b": p.company_b.ticker,
        "last_date": str(last), "expected_last_session": str(expected), "stale": last < expected, "sessions_behind": max(0, len(trading_days_between(last, expected)) - 1),
        "normalized": [{"date": ix.strftime("%Y-%m-%d"), "a": round(float(r["a"]), 4), "b": round(float(r["b"]), 4)} for ix, r in norm.iterrows()],
        "relative_return": _series(rel),
        "rolling_corr_63": _series(rc63[rc63.index >= start]),
        "rolling_corr_252": _series(rc252[rc252.index >= start]),
        "zscore": [{"date": r["date"].strftime("%Y-%m-%d"), "z": round(r["z"], 4), "spread": round(r["spread"], 6), "beta": round(r["beta"], 4)} for _, r in z.iterrows()] if not z.empty else [],
        "cointegration": coint,
        "method": {
            "returns": "daily log returns of adjusted closes",
            "zscore": "hedge ratio re-fitted every 21 sessions on the preceding 252 sessions (OLS of log A on log B); applied out-of-sample",
            "cointegration": "Engle-Granger two-step test on log prices (statsmodels.coint, MacKinnon p-values)",
        },
    }


@router.get("/{pair_id}/news")
def pair_news(pair_id: str, db: Session = Depends(get_db), _: User = Depends(current_user), page: int = Query(1, ge=1), page_size: int = Query(25, ge=1, le=100),
              material_only: bool = False):
    from app.api.routes.news import article_page
    p = resolve_pair(db, pair_id)
    return article_page(db, page, page_size, pair=p, material_only=material_only)


@router.get("/{pair_id}/predictions", response_model=Page[PredictionOut])
def pair_predictions(pair_id: str, db: Session = Depends(get_db), _: User = Depends(current_user), horizon: int | None = None,
                     state: Literal["all", "pending", "resolved"] = "all", page: int = Query(1, ge=1), page_size: int = Query(50, ge=1, le=500)):
    p = resolve_pair(db, pair_id)
    q = select(PredictionSnapshot, RealizedOutcome).outerjoin(RealizedOutcome, RealizedOutcome.snapshot_id == PredictionSnapshot.id).where(PredictionSnapshot.pair_id == p.id)
    if horizon:
        q = q.where(PredictionSnapshot.horizon_days == horizon)
    if state == "pending":
        q = q.where(RealizedOutcome.id.is_(None))
    elif state == "resolved":
        q = q.where(RealizedOutcome.id.is_not(None))
    total = db.scalar(select(func.count()).select_from(q.subquery()))
    rows = db.execute(q.order_by(PredictionSnapshot.prediction_date.desc(), PredictionSnapshot.horizon_days).offset((page - 1) * page_size).limit(page_size)).all()
    return Page(items=[prediction_out(s, o) for s, o in rows], total=total, page=page, page_size=page_size)


@router.get("/{pair_id}/events")
def pair_events_timeline(pair_id: str, db: Session = Depends(get_db), _: User = Depends(current_user), days: int = Query(730, ge=1, le=7300)):
    """Event markers for the score-history chart."""
    p = resolve_pair(db, pair_id)
    rows = db.execute(select(ExtractedEvent, NewsArticle.headline).join(NewsArticle, NewsArticle.id == ExtractedEvent.article_id)
                      .where(ExtractedEvent.pair_id == p.id, ExtractedEvent.is_material.is_(True), ExtractedEvent.published_at >= datetime.now(UTC) - timedelta(days=days))
                      .order_by(ExtractedEvent.published_at)).all()
    return [{"id": e.id, "date": e.published_at, "event_type": e.event_type, "direction": e.relative_direction, "severity": e.severity,
             "scope": e.scope, "headline": h, "summary": e.summary, "article_id": e.article_id} for e, h in rows]
