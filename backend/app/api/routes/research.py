from __future__ import annotations

import csv
import io

from fastapi import APIRouter, Depends, HTTPException, Query
from fastapi.responses import Response, StreamingResponse
from sqlalchemy import select
from sqlalchemy.orm import Session

from app.api.deps import current_user, require_admin
from app.core.config import get_settings
from app.core.db import get_db
from app.models import BacktestRun, IndexScore, Pair, PredictionSnapshot, RealizedOutcome, User
from app.services import jobs, performance_service, reports

router = APIRouter(tags=["research"])


@router.get("/performance")
def performance(db: Session = Depends(get_db), _: User = Depends(current_user), n_boot: int = Query(500, ge=100, le=5000)):
    return performance_service.performance_report(db, n_boot)


@router.get("/research/backtests/{run_id}")
def backtest(run_id: int, db: Session = Depends(get_db), _: User = Depends(current_user)):
    r = db.get(BacktestRun, run_id)
    if r is None:
        raise HTTPException(404, "backtest not found")
    return {"id": r.id, "kind": r.kind, "label": r.label, "created_at": r.created_at, "params": r.params, "metrics": r.metrics,
            "limitations": r.limitations, "observations": r.observations, "is_demo": r.is_demo}


@router.post("/research/backtests", status_code=202)
def start_backtest(params: dict, db: Session = Depends(get_db), user: User = Depends(require_admin)):
    allowed = {"horizon", "threshold", "frequency", "n_boot"}
    if set(params) - allowed:
        raise HTTPException(422, f"allowed params: {sorted(allowed)}")
    if params.get("frequency", "annual") not in ("annual", "quarterly", "monthly"):
        raise HTTPException(422, "frequency must be annual, quarterly or monthly")
    try:
        if not (5 <= int(params.get("horizon", 504)) <= 756):
            raise HTTPException(422, "horizon must be between 5 and 756 trading days")
        if not (0 < float(params.get("threshold", 0.2)) <= 2):
            raise HTTPException(422, "threshold must be in (0, 2]")
        if not (100 <= int(params.get("n_boot", 500)) <= 5000):
            raise HTTPException(422, "n_boot must be between 100 and 5000")
    except (TypeError, ValueError) as e:
        raise HTTPException(422, "horizon, threshold and n_boot must be numbers") from e
    try:
        run = jobs.enqueue(db, "run_backtest", params, requested_by=user.id)
        db.commit()
    except jobs.AlreadyRunning as e:
        raise HTTPException(409, str(e)) from e
    return {"job_run_id": run.id}


def _csv(rows: list[dict], name: str) -> StreamingResponse:
    buf = io.StringIO()
    if rows:
        w = csv.DictWriter(buf, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    return StreamingResponse(iter([buf.getvalue()]), media_type="text/csv", headers={"Content-Disposition": f'attachment; filename="{name}.csv"'})


@router.get("/research/export/{kind}.csv")
def export_csv(kind: str, db: Session = Depends(get_db), _: User = Depends(current_user), backtest_id: int | None = None):
    demo_ok = get_settings().is_demo
    if kind == "predictions":
        q = select(PredictionSnapshot, RealizedOutcome, Pair.slug).outerjoin(RealizedOutcome, RealizedOutcome.snapshot_id == PredictionSnapshot.id).join(Pair, Pair.id == PredictionSnapshot.pair_id)
        if not demo_ok:
            q = q.where(PredictionSnapshot.is_demo.is_(False))
        rows = [{"snapshot_id": s.id, "pair": slug, "prediction_date": s.prediction_date, "data_cutoff": s.data_cutoff.isoformat(), "horizon": s.horizon_days,
                 "target_version": s.target_version, "threshold": s.target_definition.get("threshold"), "si_v1": s.si_v1, "si_v2": s.si_v2,
                 "si_v2_status": s.si_v2_status, "si_v2_model_version": s.si_v2_model_version, "confidence": s.confidence, "direction": s.direction,
                 "corr_only": (s.baselines or {}).get("correlation_only"), "prev_gap": (s.baselines or {}).get("previous_return_gap"),
                 "price_provider": s.price_provider, "total_return": s.is_total_return, "simulated": s.is_demo,
                 "resolved": o is not None, "end_date": o.end_date if o else None, "divergence": o.divergence if o else None,
                 "diverged": o.diverged if o else None} for s, o, slug in db.execute(q.order_by(PredictionSnapshot.prediction_date)).all()]
        return _csv(rows, "predictions")
    if kind == "scores":
        q = select(IndexScore, Pair.slug).join(Pair, Pair.id == IndexScore.pair_id)
        if not demo_ok:
            q = q.where(IndexScore.is_demo.is_(False))
        rows = [{"score_id": s.id, "pair": slug, "index": s.index_name, "as_of": s.as_of.isoformat(), "value": s.value, "status": s.status,
                 "coverage": s.coverage, "direction": s.direction, "confidence": s.confidence, "trigger": s.trigger, "simulated": s.is_demo,
                 **{f"comp_{c.component}": c.value for c in s.components}} for s, slug in db.execute(q.order_by(IndexScore.as_of)).all()]
        return _csv(rows, "scores")
    if kind == "backtest":
        r = db.get(BacktestRun, backtest_id) if backtest_id else db.scalar(select(BacktestRun).where(BacktestRun.kind == "walk_forward").order_by(BacktestRun.id.desc()))
        if r is None:
            raise HTTPException(404, "no backtest run")
        return _csv(r.observations or [], f"backtest-{r.id}")
    raise HTTPException(404, "kind must be predictions|scores|backtest")


@router.get("/research/report.pdf")
def report_pdf(db: Session = Depends(get_db), _: User = Depends(current_user)):
    pdf = reports.build_pdf(db)
    return Response(pdf, media_type="application/pdf", headers={"Content-Disposition": 'attachment; filename="splittability-report.pdf"'})
