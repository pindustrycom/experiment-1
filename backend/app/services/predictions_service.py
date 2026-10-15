"""Immutable daily prediction snapshots and realized-outcome evaluation (forward test)."""
from __future__ import annotations

import logging
import math
from datetime import UTC, date, datetime

import pandas as pd
from sqlalchemy import select
from sqlalchemy.dialects.postgresql import insert
from sqlalchemy.orm import Session

from app.models import ComponentCalculation, ExtractedEvent, IndexScore, ModelVersion, Pair, PredictionSnapshot, RealizedOutcome, SourceDocument
from app.research.market_calendar import add_trading_days, is_trading_day, last_completed_session
from app.services import prices_service, scoring_service, settings_service

log = logging.getLogger(__name__)


def _components(db: Session, score: IndexScore | None) -> dict:
    if score is None:
        return {}
    rows = db.scalars(select(ComponentCalculation).where(ComponentCalculation.score_id == score.id)).all()
    return {r.component: {"value": r.value, "available": r.available, "weight": r.weight} for r in rows}


def snapshot_predictions(db: Session, now: datetime | None = None) -> dict:
    now = now or datetime.now(UTC)
    pdate = last_completed_session(now)
    cfg = settings_service.get_section(db, "predictions")
    out = {"prediction_date": str(pdate), "created": 0, "existing": 0, "skipped": []}
    for pair in db.scalars(select(Pair).where(Pair.status == "active").order_by(Pair.slug)).all():
        # Always score with the current cutoff first so the snapshot reflects data available now.
        scoring_service.score_pair(db, pair, now, trigger="snapshot")
        s1 = scoring_service.latest_score(db, pair.id, "si_v1", now)
        s2 = scoring_service.latest_score(db, pair.id, "si_v2", now)
        inp_market = prices_service.pair_market_data(db, pair.company_a, pair.company_b, now)
        prices = inp_market.prices
        if prices.empty:
            out["skipped"].append(f"{pair.slug}: no prices")
            continue
        base_date = pd.Timestamp(prices.index[-1]).date()
        if (pdate - base_date).days > 5:
            out["skipped"].append(f"{pair.slug}: last price {base_date} is stale relative to {pdate}")
            continue
        m2 = s2.components if s2 else []
        minfo = {"provider": inp_market.provider, "last_price_date": str(base_date), "n_prices": len(prices)}
        for c in m2:
            if c.component == "M":
                minfo.update({k: c.inputs.get(k) for k in ("rho_long", "rho_short", "z_last", "hedge_beta", "coint_pvalue")})
        baselines = (s1.confidence_breakdown or {}).get("baselines", {}) if s1 else {}
        v1mv = db.get(ModelVersion, s1.model_version_id) if s1 else None
        v2mv = db.get(ModelVersion, s2.model_version_id) if s2 else None
        docs = sorted({d for ids in news_docs_for(db, s2.supporting_event_ids) for d in ids}) if s2 else []
        for h in cfg["horizons"]:
            tv, tdef = settings_service.target_definition(int(h), cfg)
            vals = dict(
                pair_id=pair.id, prediction_date=pdate, data_cutoff=now, horizon_days=int(h), target_version=tv, target_definition=tdef,
                si_v1=s1.value if s1 else None, si_v2=s2.value if s2 else None, si_v2_status=s2.status if s2 else "unavailable",
                si_v1_score_id=s1.id if s1 else None, si_v2_score_id=s2.id if s2 else None,
                si_v1_model_version=v1mv.version if v1mv else None, si_v2_model_version=v2mv.version if v2mv else None,
                weights_version=v2mv.content_hash[:10] if v2mv else None,
                components={"si_v1": _components(db, s1), "si_v2": _components(db, s2)},
                baselines={"correlation_only": baselines.get("correlation_only"), "previous_return_gap": baselines.get("previous_return_gap")},
                confidence=s2.confidence if s2 else None, direction=s2.direction if s2 else "uncertain",
                supporting_event_ids=s2.supporting_event_ids if s2 else [], supporting_document_ids=docs,
                market_info=minfo, base_date=base_date, base_price_a=float(prices["a"].iloc[-1]), base_price_b=float(prices["b"].iloc[-1]),
                price_provider=inp_market.provider, is_total_return=inp_market.is_total_return, is_demo=inp_market.provider == "demo",
            )
            stmt = insert(PredictionSnapshot).values(**vals).on_conflict_do_nothing(constraint="uq_prediction_identity").returning(PredictionSnapshot.id)
            if db.execute(stmt).scalar() is None:
                out["existing"] += 1
            else:
                out["created"] += 1
        db.commit()
    return out


def news_docs_for(db: Session, event_ids: list[int]) -> list[list[int]]:
    if not event_ids:
        return []
    arts = db.scalars(select(ExtractedEvent.article_id).where(ExtractedEvent.id.in_(event_ids))).all()
    return [list(db.scalars(select(SourceDocument.id).where(SourceDocument.article_id == a)).all()) for a in arts]


def realized_divergence(pa0: float, pa1: float, pb0: float, pb1: float) -> tuple[float, float, float]:
    la, lb = math.log(pa1 / pa0), math.log(pb1 / pb0)
    return abs(la - lb), la, lb


def paper_trades(snap: PredictionSnapshot, la: float, lb: float) -> dict:
    """Hypothetical, before costs, never executed."""
    ra, rb = math.exp(la) - 1, math.exp(lb) - 1
    out = {"note": "hypothetical dollar-neutral paper trades; no costs, borrow fees or execution", "return_a": ra, "return_b": rb}
    z = (snap.market_info or {}).get("z_last")
    if z is not None and abs(z) >= 2:
        # classic convergence bet: short the rich leg (A if z>0), long the cheap leg
        out["convergence"] = {"entered": True, "z_at_entry": z, "return": round(-(ra - rb) if z > 0 else (ra - rb), 6)}
    else:
        out["convergence"] = {"entered": False, "reason": "|z| < 2 or z unavailable"}
    if snap.direction == "a_advantaged":
        out["bias"] = {"entered": True, "position": "long A / short B", "return": round(ra - rb, 6)}
    elif snap.direction == "b_advantaged":
        out["bias"] = {"entered": True, "position": "long B / short A", "return": round(rb - ra, 6)}
    else:
        out["bias"] = {"entered": False, "reason": f"bias {snap.direction}"}
    return out


def evaluate_outcomes(db: Session, now: datetime | None = None) -> dict:
    now = now or datetime.now(UTC)
    last_session = last_completed_session(now)
    pending = db.scalars(select(PredictionSnapshot).where(~select(RealizedOutcome.id).where(RealizedOutcome.snapshot_id == PredictionSnapshot.id).exists())).all()
    out = {"evaluated": 0, "not_matured": 0, "missing_data": []}
    cache: dict = {}
    for snap in pending:
        end = add_trading_days(snap.base_date, snap.horizon_days)
        if end > last_session:
            out["not_matured"] += 1
            continue
        pair = db.get(Pair, snap.pair_id)
        key = (pair.id, snap.price_provider)
        if key not in cache:
            sa, _, tra = prices_service.load_series(db, pair.company_a_id, snap.price_provider)
            sb, _, trb = prices_service.load_series(db, pair.company_b_id, snap.price_provider)
            cache[key] = (sa, sb, tra and trb)
        sa, sb, tr = cache[key]
        try:
            # Both endpoints from the *current* adjusted series so dividend re-basing cancels out.
            pa0, pb0 = float(sa.loc[pd.Timestamp(snap.base_date)]), float(sb.loc[pd.Timestamp(snap.base_date)])
            pa1, pb1 = float(sa.loc[pd.Timestamp(end)]), float(sb.loc[pd.Timestamp(end)])
        except KeyError:
            out["missing_data"].append(f"{pair.slug} H={snap.horizon_days}: no bar on {snap.base_date} or {end}")
            continue
        d, la, lb = realized_divergence(pa0, pa1, pb0, pb1)
        thr = float(snap.target_definition["threshold"])
        db.add(RealizedOutcome(snapshot_id=snap.id, end_date=end, end_price_a=pa1, end_price_b=pb1, log_return_a=la, log_return_b=lb,
                               divergence=d, threshold=thr, diverged=d >= thr, paper_trades=paper_trades(snap, la, lb),
                               price_provider=snap.price_provider, is_total_return=tr, is_demo=snap.is_demo))
        out["evaluated"] += 1
    db.commit()
    return out


def maturity_date(base: date, horizon: int) -> date:
    return add_trading_days(base, horizon)


__all__ = ["snapshot_predictions", "evaluate_outcomes", "realized_divergence", "maturity_date", "is_trading_day"]
