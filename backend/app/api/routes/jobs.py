from __future__ import annotations

from datetime import UTC, datetime, timedelta

from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy import func, select
from sqlalchemy.orm import Session

from app.api.deps import current_user, require_admin
from app.core.config import get_settings
from app.core.db import get_db
from app.models import JobRun, ScheduledJob, User
from app.schemas.api import JobRunOut, Page, RunJobRequest, ScheduledJobOut, ScheduleUpdate
from app.services import jobs, settings_service
from worker.scheduler import cron_trigger, should_run_today

router = APIRouter(prefix="/jobs", tags=["jobs"])


def next_fire(r: ScheduledJob, now: datetime) -> datetime | None:
    """Next fire time, skipping weekends/NYSE holidays for trading-day jobs."""
    if not (r.enabled and r.cron):
        return None
    trig = cron_trigger(r.cron, r.timezone)
    prev = None
    for _ in range(400):
        t = trig.get_next_fire_time(prev, now if prev is None else prev + timedelta(seconds=1))
        if t is None:
            return None
        if not r.trading_days_only or should_run_today(True, t, r.timezone):
            return t
        prev = t
    return None


def schedule_out(db: Session, r: ScheduledJob, now: datetime) -> ScheduledJobOut:
    last = db.scalar(select(JobRun).where(JobRun.job_name == r.name).order_by(JobRun.created_at.desc()).limit(1))
    o = ScheduledJobOut.model_validate(r)
    o.next_run, o.last_run = next_fire(r, now), JobRunOut.model_validate(last) if last else None
    return o


@router.get("", response_model=list[ScheduledJobOut])
def list_jobs(db: Session = Depends(get_db), _: User = Depends(current_user)):
    demo = get_settings().is_demo
    jobs.ensure_scheduled_jobs(db, demo)
    db.commit()
    now = datetime.now(UTC)
    return [schedule_out(db, r, now) for r in db.scalars(select(ScheduledJob).order_by(ScheduledJob.name)).all() if r.name in jobs.registry(demo)]


@router.get("/runs", response_model=Page[JobRunOut])
def list_runs(db: Session = Depends(get_db), _: User = Depends(current_user), job: str | None = None, status: str | None = None,
              page: int = Query(1, ge=1), page_size: int = Query(50, ge=1, le=200)):
    q = select(JobRun)
    if job:
        q = q.where(JobRun.job_name == job)
    if status:
        q = q.where(JobRun.status == status)
    total = db.scalar(select(func.count()).select_from(q.subquery()))
    rows = db.scalars(q.order_by(JobRun.created_at.desc()).offset((page - 1) * page_size).limit(page_size)).all()
    return Page(items=[JobRunOut.model_validate(r) for r in rows], total=total, page=page, page_size=page_size)


@router.post("/run", response_model=JobRunOut, status_code=202)
def run_job(body: RunJobRequest, db: Session = Depends(get_db), user: User = Depends(require_admin)):
    """Queue a job for the worker (it is picked up within WORKER_POLL_SECONDS)."""
    if body.job not in jobs.registry(get_settings().is_demo):
        raise HTTPException(404, f"unknown job '{body.job}'")
    allowed = {"run_backtest": {"horizon", "threshold", "frequency", "n_boot"}, "compute_scores": {"pair_ids"}}
    extra = set(body.params) - allowed.get(body.job, set())
    if extra:
        raise HTTPException(422, f"unsupported params for {body.job}: {sorted(extra)}")
    try:
        run = jobs.enqueue(db, body.job, body.params, requested_by=user.id)
        db.commit()
    except jobs.AlreadyRunning as e:
        raise HTTPException(409, str(e)) from e
    return JobRunOut.model_validate(run)


@router.post("/pause")
def pause(db: Session = Depends(get_db), user: User = Depends(require_admin)):
    settings_service.put_section(db, "bot", {"paused": True}, user.id)
    db.commit()
    return {"paused": True}


@router.post("/resume")
def resume(db: Session = Depends(get_db), user: User = Depends(require_admin)):
    settings_service.put_section(db, "bot", {"paused": False}, user.id)
    db.commit()
    return {"paused": False}


@router.patch("/schedules/{name}", response_model=ScheduledJobOut)
def update_schedule(name: str, body: ScheduleUpdate, db: Session = Depends(get_db), _: User = Depends(require_admin)):
    jobs.ensure_scheduled_jobs(db, get_settings().is_demo)
    r = db.scalar(select(ScheduledJob).where(ScheduledJob.name == name))
    if r is None:
        raise HTTPException(404, "schedule not found")
    if body.cron is not None:
        try:
            cron_trigger(body.cron, r.timezone)
        except ValueError as e:
            raise HTTPException(422, f"invalid cron expression: {e}") from e
        r.cron = body.cron
    if body.enabled is not None:
        if body.enabled and not r.cron:
            raise HTTPException(422, "set a cron expression before enabling")
        r.enabled = body.enabled
    db.commit()
    return schedule_out(db, r, datetime.now(UTC))
