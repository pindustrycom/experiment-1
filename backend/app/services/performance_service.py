"""Live forward-test performance (kept strictly separate from backtests)."""
from __future__ import annotations

from collections import defaultdict

import numpy as np
from sqlalchemy import select
from sqlalchemy.orm import Session

from app.core.config import get_settings
from app.models import BacktestRun, Pair, PredictionSnapshot, RealizedOutcome
from app.research.evaluation import evaluate, roc_auc
from app.services import settings_service

INDICES = {
    "si_v1": lambda s: s.si_v1,
    "si_v2": lambda s: s.si_v2,
    "correlation_only": lambda s: (s.baselines or {}).get("correlation_only"),
    "previous_return_gap": lambda s: (s.baselines or {}).get("previous_return_gap"),
}


def live_rows(db: Session):
    q = select(PredictionSnapshot, RealizedOutcome, Pair).join(Pair, Pair.id == PredictionSnapshot.pair_id).outerjoin(
        RealizedOutcome, RealizedOutcome.snapshot_id == PredictionSnapshot.id)
    if not get_settings().is_demo:
        q = q.where(PredictionSnapshot.is_demo.is_(False))
    return db.execute(q).all()


def live_performance(db: Session, n_boot: int = 500) -> dict:
    cfg = settings_service.get_section(db, "predictions")
    rows = live_rows(db)
    by_h: dict[int, list] = defaultdict(list)
    for snap, out, pair in rows:
        by_h[snap.horizon_days].append((snap, out, pair))
    result: dict = {"horizons": {}, "contains_simulated": any(s.is_demo for s, _, _ in rows),
                    "note": "Live forward predictions only. Unresolved predictions are never counted as correct or incorrect."}
    for h in sorted(set(by_h) | {int(x) for x in cfg["horizons"]}):
        items = by_h.get(h, [])
        resolved = [(s, o, p) for s, o, p in items if o is not None]
        hres: dict = {"pending": len(items) - len(resolved), "resolved": len(resolved),
                      "target": settings_service.target_definition(h, cfg)[1], "indices": {}, "by_sector": {}, "sensitivity": []}
        for name, fn in INDICES.items():
            vals = [(fn(s), o, p) for s, o, p in resolved if fn(s) is not None]
            hres["indices"][name] = evaluate([v for v, _, _ in vals], [int(o.diverged) for _, o, _ in vals],
                                             [p.industry or "unclassified" for _, _, p in vals], [o.divergence for _, o, _ in vals], n_boot=n_boot)
        sectors = defaultdict(list)
        for s, o, p in resolved:
            sectors[p.industry or "unclassified"].append((s, o))
        for sec, lst in sorted(sectors.items()):
            entry = {"n": len(lst), "divergence_rate": float(np.mean([o.diverged for _, o in lst]))}
            for name in ("si_v1", "si_v2"):
                v = [(INDICES[name](s), int(o.diverged)) for s, o in lst if INDICES[name](s) is not None]
                entry[f"{name}_auc"] = roc_auc([a for a, _ in v], [b for _, b in v]) if len(v) >= 6 else None
            hres["by_sector"][sec] = entry
        for mult in cfg["sensitivity_multipliers"]:
            row = {"multiplier": mult}
            for name in ("si_v1", "si_v2", "correlation_only"):
                v = [(INDICES[name](s), int(o.divergence >= o.threshold * mult)) for s, o, _ in resolved if INDICES[name](s) is not None]
                row[f"{name}_auc"] = roc_auc([a for a, _ in v], [b for _, b in v]) if len(v) >= 20 else None
                row["base_rate"] = float(np.mean([b for _, b in v])) if v else None
            hres["sensitivity"].append(row)
        trades = {"convergence": [], "bias": []}
        for _s, o, _ in resolved:
            for k in trades:
                t = (o.paper_trades or {}).get(k, {})
                if t.get("entered"):
                    trades[k].append(t["return"])
        hres["paper_trades"] = {k: {"n": len(v), "mean_return": float(np.mean(v)) if v else None,
                                    "hit_rate": float(np.mean([x > 0 for x in v])) if v else None} for k, v in trades.items()}
        result["horizons"][str(h)] = hres
    return result


def performance_report(db: Session, n_boot: int = 500) -> dict:
    runs = db.scalars(select(BacktestRun).order_by(BacktestRun.created_at.desc())).all()
    if not get_settings().is_demo:
        runs = [r for r in runs if not r.is_demo]
    return {
        "live": live_performance(db, n_boot),
        "backtests": [{"id": r.id, "kind": r.kind, "label": r.label, "created_at": r.created_at, "params": r.params, "metrics": r.metrics,
                       "limitations": r.limitations, "n_observations": len(r.observations or []), "is_demo": r.is_demo} for r in runs],
    }
