"""ORM -> API shapes shared by several routers."""
from __future__ import annotations

from datetime import timedelta

from sqlalchemy import select
from sqlalchemy.orm import Session

from app.models import IndexScore, ModelVersion, Pair, WatchlistPair
from app.research.market_calendar import add_trading_days
from app.schemas.api import CompanyOut, LatestScore, PairOut, PredictionOut, ScoreOut


def latest_scores(db: Session, pair_ids: list[int]) -> dict[tuple[int, str], IndexScore]:
    if not pair_ids:
        return {}
    sub = (select(IndexScore.pair_id, IndexScore.index_name, IndexScore.id)
           .where(IndexScore.pair_id.in_(pair_ids))
           .distinct(IndexScore.pair_id, IndexScore.index_name)
           .order_by(IndexScore.pair_id, IndexScore.index_name, IndexScore.as_of.desc(), IndexScore.id.desc())).subquery()
    rows = db.scalars(select(IndexScore).where(IndexScore.id.in_(select(sub.c.id)))).all()
    return {(r.pair_id, r.index_name): r for r in rows}


def value_days_ago(db: Session, pair_id: int, index_name: str, ref: IndexScore, days: int = 7) -> float | None:
    v = db.scalar(select(IndexScore.value).where(IndexScore.pair_id == pair_id, IndexScore.index_name == index_name, IndexScore.value.is_not(None),
                                                 IndexScore.as_of <= ref.as_of - timedelta(days=days)).order_by(IndexScore.as_of.desc()).limit(1))
    return v


def _latest(db: Session, s: IndexScore | None, versions: dict[int, str]) -> LatestScore | None:
    if s is None:
        return None
    prev = value_days_ago(db, s.pair_id, s.index_name, s)
    return LatestScore(value=s.value, status=s.status, as_of=s.as_of, model_version=versions.get(s.model_version_id),
                       change_7d=None if (s.value is None or prev is None) else round(s.value - prev, 4))


def pair_out(db: Session, pairs: list[Pair]) -> list[PairOut]:
    ids = [p.id for p in pairs]
    latest = latest_scores(db, ids)
    versions = dict(db.execute(select(ModelVersion.id, ModelVersion.version)).all())
    wl: dict[int, list[int]] = {}
    for wid, pid in db.execute(select(WatchlistPair.watchlist_id, WatchlistPair.pair_id).where(WatchlistPair.pair_id.in_(ids))).all() if ids else []:
        wl.setdefault(pid, []).append(wid)
    out = []
    for p in pairs:
        s1, s2 = latest.get((p.id, "si_v1")), latest.get((p.id, "si_v2"))
        out.append(PairOut(
            id=p.id, uid=p.uid, slug=p.slug, company_a=CompanyOut.model_validate(p.company_a), company_b=CompanyOut.model_validate(p.company_b),
            industry=p.industry, status=p.status, monitoring_interval_minutes=p.monitoring_interval_minutes, alert_threshold=p.alert_threshold,
            alert_change_threshold=p.alert_change_threshold, notes=p.notes, created_at=p.created_at, archived_at=p.archived_at,
            watchlist_ids=sorted(wl.get(p.id, [])), si_v1=_latest(db, s1, versions), si_v2=_latest(db, s2, versions),
            direction=s2.direction if s2 else None, confidence=s2.confidence if s2 else None, explanation=s2.explanation if s2 else None,
            last_update=max([s.computed_at for s in (s1, s2) if s], default=None), is_demo=bool((s2 and s2.is_demo) or (s1 and s1.is_demo)),
        ))
    return out


def score_out(s: IndexScore, version: str, detail: bool = False) -> ScoreOut:
    return ScoreOut(
        id=s.id, index_name=s.index_name, model_version=version, as_of=s.as_of, computed_at=s.computed_at, value=s.value, status=s.status,
        coverage=s.coverage, direction=s.direction, confidence=s.confidence, explanation=s.explanation, trigger=s.trigger,
        supporting_event_ids=s.supporting_event_ids or [], is_demo=s.is_demo,
        components=[c for c in s.components] if detail else None,
        confidence_breakdown=s.confidence_breakdown if detail else None, direction_detail=s.direction_detail if detail else None,
    )


def prediction_out(snap, outcome) -> PredictionOut:
    o = None
    err = None
    if outcome is not None:
        o = dict(end_date=outcome.end_date, divergence=outcome.divergence, threshold=outcome.threshold, diverged=outcome.diverged,
                 log_return_a=outcome.log_return_a, log_return_b=outcome.log_return_b, paper_trades=outcome.paper_trades,
                 evaluated_at=outcome.evaluated_at, is_total_return=outcome.is_total_return)
        if snap.si_v2 is not None:
            err = round((snap.si_v2 / 100) - float(outcome.diverged), 4)  # pseudo-probability error (index is not calibrated)
    return PredictionOut(
        id=snap.id, prediction_date=snap.prediction_date, data_cutoff=snap.data_cutoff, horizon_days=snap.horizon_days,
        maturity_date=add_trading_days(snap.base_date, snap.horizon_days), target_version=snap.target_version,
        target_definition=snap.target_definition, si_v1=snap.si_v1, si_v2=snap.si_v2, si_v2_status=snap.si_v2_status,
        si_v2_model_version=snap.si_v2_model_version, weights_version=snap.weights_version, confidence=snap.confidence,
        direction=snap.direction, baselines=snap.baselines or {}, base_date=snap.base_date, price_provider=snap.price_provider,
        is_total_return=snap.is_total_return, is_demo=snap.is_demo, status="resolved" if outcome else "pending", outcome=o, error=err,
    )
