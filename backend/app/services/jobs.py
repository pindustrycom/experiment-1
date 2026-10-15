"""Job registry and a crash-safe runner.

Duplicate protection: partial unique indexes allow at most one 'running' and one 'queued' row per
job name, across any number of worker processes. Retries use exponential backoff. Every run,
skip, failure, duration and error is recorded in job_runs.
"""
from __future__ import annotations

import logging
import socket
import threading
import time
import traceback
from collections.abc import Callable
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta

from sqlalchemy import select, update
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session

from app.core.db import session_scope
from app.data_sources.base import ProviderError, ProviderNotConfigured
from app.data_sources.http import current_job_run_id
from app.models import JobRun, ScheduledJob
from app.services import settings_service

log = logging.getLogger(__name__)
WORKER_ID = f"{socket.gethostname()}:{id(object())}"


@dataclass
class JobDef:
    name: str
    func: Callable[[Session, dict], dict]
    cron: str | None
    description: str
    trading_days_only: bool = False
    max_attempts: int = 3
    timeout_seconds: int = 1800


def _ingest_news(db: Session, params: dict) -> dict:
    from app.services import news_service, scoring_service
    ing = news_service.ingest_news(db)
    ana = news_service.analyze_pending(db)
    rescored = scoring_service.score_all(db, trigger="event", pair_ids=ana["material_pairs"]) if ana["material_pairs"] else None
    return {"ingest": ing, "analysis": ana, "rescored": rescored}


def _analyze_news(db: Session, params: dict) -> dict:
    from app.services import news_service, scoring_service
    ana = news_service.analyze_pending(db)
    if ana["material_pairs"]:
        ana["rescored"] = scoring_service.score_all(db, trigger="event", pair_ids=ana["material_pairs"])
    return ana


def _ingest_prices(db: Session, params: dict) -> dict:
    from datetime import date

    from app.services import prices_service
    cfg = settings_service.get_section(db, "prices")
    return prices_service.ingest_prices(db, cfg["providers"], date.fromisoformat(cfg["history_start"]))


def _ingest_fundamentals(db: Session, params: dict) -> dict:
    from app.services import fundamentals_service
    return fundamentals_service.ingest_fundamentals(db)


def _sync_reference(db: Session, params: dict) -> dict:
    from app.services import fundamentals_service
    return fundamentals_service.sync_reference(db)


def _compute_scores(db: Session, params: dict) -> dict:
    from app.services import scoring_service
    return scoring_service.score_all(db, trigger=params.get("trigger", "scheduled"), pair_ids=params.get("pair_ids"))


def _snapshot(db: Session, params: dict) -> dict:
    from app.services import predictions_service
    return predictions_service.snapshot_predictions(db)


def _evaluate(db: Session, params: dict) -> dict:
    from app.services import predictions_service
    return predictions_service.evaluate_outcomes(db)


def _backup(db: Session, params: dict) -> dict:
    from app.services import backup_service
    return backup_service.run_backup()


def _backtest(db: Session, params: dict) -> dict:
    from app.services import backtest_service
    run = backtest_service.run_walk_forward(
        db, horizon=int(params.get("horizon", 504)), threshold=float(params.get("threshold", 0.20)),
        frequency=params.get("frequency", "annual"), n_boot=int(params.get("n_boot", 500)))
    return {"backtest_run_id": run.id, "n_observations": run.metrics.get("n_observations")}


def _demo_refresh(db: Session, params: dict) -> dict:
    from app.services import demo_service
    return demo_service.refresh(db)


JOBS: dict[str, JobDef] = {j.name: j for j in [
    JobDef("ingest_news", _ingest_news, "0 * * * *", "Fetch news/filings, deduplicate, analyse with Claude, re-score pairs with new material evidence", max_attempts=2),
    JobDef("analyze_news", _analyze_news, None, "Analyse pending stories (e.g. after raising the AI budget)", max_attempts=1),
    JobDef("ingest_prices", _ingest_prices, "30 17 * * 1-5", "Daily bars after the US close", trading_days_only=True),
    JobDef("ingest_fundamentals", _ingest_fundamentals, "0 6 * * *", "SEC XBRL company facts (point-in-time)"),
    JobDef("sync_reference", _sync_reference, "0 5 * * 1", "Verify CIKs and SIC codes with SEC"),
    JobDef("compute_scores", _compute_scores, "0 18 * * *", "Daily SI-v1/SI-v2 recalculation for all active pairs"),
    JobDef("snapshot_predictions", _snapshot, "30 18 * * 1-5", "Immutable daily prediction snapshots", trading_days_only=True),
    JobDef("evaluate_outcomes", _evaluate, "0 19 * * 1-5", "Resolve matured predictions against realised prices", trading_days_only=True),
    JobDef("backup_database", _backup, "0 2 * * *", "pg_dump backup with retention", max_attempts=2),
    JobDef("run_backtest", _backtest, None, "Historical walk-forward research run (manual)", max_attempts=1, timeout_seconds=7200),
]}
DEMO_JOBS = {"demo_refresh": JobDef("demo_refresh", _demo_refresh, "15 * * * *", "DEMO MODE: extend simulated data so the bot visibly runs")}


# Jobs that call external providers. They are not registered in demo mode, so simulated and real
# data never mix in one database.
EXTERNAL_JOBS = {"ingest_news", "analyze_news", "ingest_prices", "ingest_fundamentals", "sync_reference"}


def registry(demo: bool) -> dict[str, JobDef]:
    if demo:
        return {**{k: v for k, v in JOBS.items() if k not in EXTERNAL_JOBS}, **DEMO_JOBS}
    return dict(JOBS)


def ensure_scheduled_jobs(db: Session, demo: bool) -> None:
    for j in registry(demo).values():
        row = db.scalar(select(ScheduledJob).where(ScheduledJob.name == j.name))
        if row is None:
            db.add(ScheduledJob(name=j.name, description=j.description, cron=j.cron or "", enabled=j.cron is not None,
                                trading_days_only=j.trading_days_only, max_attempts=j.max_attempts, timeout_seconds=j.timeout_seconds))
    db.flush()


class AlreadyRunning(Exception):
    pass


def enqueue(db: Session, name: str, params: dict | None = None, requested_by: int | None = None, trigger: str = "manual") -> JobRun:
    run = JobRun(job_name=name, trigger=trigger, status="queued", params=params or {}, requested_by=requested_by)
    db.add(run)
    try:
        db.flush()
    except IntegrityError as e:
        db.rollback()
        raise AlreadyRunning(f"{name} is already queued") from e
    return run


def _claim_new(name: str, trigger: str, params: dict, requested_by: int | None) -> int | None:
    with session_scope() as db:
        run = JobRun(job_name=name, trigger=trigger, status="running", params=params, requested_by=requested_by,
                     started_at=datetime.now(UTC), heartbeat_at=datetime.now(UTC), worker_id=WORKER_ID)
        db.add(run)
        try:
            db.flush()
            return run.id
        except IntegrityError:
            db.rollback()
    with session_scope() as db:
        db.add(JobRun(job_name=name, trigger=trigger, status="skipped", params=params, error="another instance is already running",
                      started_at=datetime.now(UTC), finished_at=datetime.now(UTC), worker_id=WORKER_ID))
    return None


def _heartbeat(run_id: int, stop: threading.Event) -> None:
    while not stop.wait(20):
        try:
            with session_scope() as db:
                db.execute(update(JobRun).where(JobRun.id == run_id).values(heartbeat_at=datetime.now(UTC)))
        except Exception:
            log.exception("heartbeat failed")


def execute(run_id: int, job: JobDef, params: dict, sleep: Callable[[float], None] = time.sleep, base_delay: float = 5.0) -> JobRun:
    stop = threading.Event()
    hb = threading.Thread(target=_heartbeat, args=(run_id, stop), daemon=True)
    hb.start()
    t0 = time.monotonic()
    token = current_job_run_id.set(run_id)
    attempts, status, error, result = 0, "failed", None, {}
    try:
        while attempts < job.max_attempts:
            attempts += 1
            try:
                with session_scope() as db:
                    result = job.func(db, params) or {}
                status, error = "success", None
                break
            except ProviderNotConfigured as e:
                error = str(e)
                break  # retrying cannot help
            except ProviderError as e:
                error = str(e)
                if not e.retryable:
                    break
            except Exception as e:
                error = f"{type(e).__name__}: {e}\n{traceback.format_exc(limit=5)}"
            log.warning("job attempt failed", extra={"job": job.name, "attempt": attempts, "error": error[:500] if error else None})
            if attempts < job.max_attempts:
                sleep(base_delay * 2 ** (attempts - 1))
    finally:
        current_job_run_id.reset(token)
        stop.set()
    with session_scope() as db:
        run = db.get(JobRun, run_id)
        run.status, run.error, run.attempts = status, error, attempts
        run.result = _json_safe(result)
        run.finished_at = datetime.now(UTC)
        run.duration_ms = int((time.monotonic() - t0) * 1000)
        db.flush()
        db.expunge(run)
    log.info("job finished", extra={"job": job.name, "status": status, "attempts": attempts})
    return run


def _json_safe(x):
    import json
    return json.loads(json.dumps(x, default=str))


def run_now(name: str, trigger: str = "schedule", params: dict | None = None, requested_by: int | None = None, demo: bool = False,
            sleep: Callable[[float], None] = time.sleep) -> JobRun | None:
    job = registry(demo).get(name)
    if job is None:
        raise KeyError(name)
    if trigger == "schedule":
        with session_scope() as db:
            if settings_service.get_section(db, "bot")["paused"]:
                db.add(JobRun(job_name=name, trigger=trigger, status="skipped", error="bot paused", params=params or {},
                              started_at=datetime.now(UTC), finished_at=datetime.now(UTC), worker_id=WORKER_ID))
                return None
    run_id = _claim_new(name, trigger, params or {}, requested_by)
    if run_id is None:
        return None
    return execute(run_id, job, params or {}, sleep=sleep)


def claim_queued(demo: bool) -> tuple[int, JobDef, dict] | None:
    """Atomically turn one queued run into a running one (SKIP LOCKED: safe with several workers)."""
    with session_scope() as db:
        running = select(JobRun.job_name).where(JobRun.status == "running")
        run = db.scalar(select(JobRun).where(JobRun.status == "queued", JobRun.job_name.not_in(running))
                        .order_by(JobRun.created_at).with_for_update(skip_locked=True).limit(1))
        if run is None:
            return None
        job = registry(demo).get(run.job_name)
        if job is None:
            run.status, run.error, run.finished_at = "failed", "unknown job", datetime.now(UTC)
            return None
        run.status, run.started_at, run.heartbeat_at, run.worker_id = "running", datetime.now(UTC), datetime.now(UTC), WORKER_ID
        try:
            db.flush()
        except IntegrityError:
            db.rollback()  # same job already running; leave it queued for the next poll
            return None
        return run.id, job, dict(run.params or {})


def recover_interrupted(stale_after: timedelta = timedelta(minutes=2)) -> int:
    """After a crash/restart, runs whose heartbeat stopped are marked interrupted."""
    with session_scope() as db:
        cutoff = datetime.now(UTC) - stale_after
        res = db.execute(update(JobRun).where(JobRun.status == "running", (JobRun.heartbeat_at.is_(None)) | (JobRun.heartbeat_at < cutoff))
                         .values(status="interrupted", finished_at=datetime.now(UTC), error="worker stopped while the job was running"))
        return res.rowcount or 0
