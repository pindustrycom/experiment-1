"""Background worker: runs independently of the web app.

* Schedules are read from the scheduled_jobs table (persisted, editable in Settings) and evaluated in
  America/New_York with DST handled by zoneinfo; trading-day-only jobs skip NYSE holidays.
* On start it marks runs interrupted by a crash and catches up any schedule missed while down.
* Manual runs requested from the dashboard are claimed from the job_runs queue.
"""
from __future__ import annotations

import logging
import signal
import threading
import time
from datetime import UTC, datetime

from apscheduler.schedulers.background import BackgroundScheduler
from sqlalchemy import func, select

from app.core.config import get_settings
from app.core.db import session_scope
from app.core.logging import configure_logging
from app.models import JobRun, ScheduledJob
from app.services import jobs, usage
from worker.scheduler import cron_trigger, missed_run, should_run_today

log = logging.getLogger("worker")


class Worker:
    def __init__(self) -> None:
        self.s = get_settings()
        self.demo = self.s.is_demo
        self.scheduler = BackgroundScheduler(timezone=self.s.scheduler_timezone, job_defaults={"coalesce": True, "max_instances": 1, "misfire_grace_time": 3600})
        self.stop = threading.Event()
        self._schedule_sig: str | None = None
        self.pool: list[threading.Thread] = []

    def _fire(self, name: str, trading_only: bool, tz: str) -> None:
        if not should_run_today(trading_only, datetime.now(UTC), tz):
            log.info("skipping non-trading day", extra={"job": name})
            return
        try:
            jobs.run_now(name, trigger="schedule", demo=self.demo)
        except Exception:
            log.exception("scheduled job crashed", extra={"job": name})

    def load_schedules(self) -> None:
        with session_scope() as db:
            rows = db.scalars(select(ScheduledJob)).all()
            sig = "|".join(f"{r.name}:{r.cron}:{r.timezone}:{r.enabled}:{r.trading_days_only}:{r.updated_at}" for r in sorted(rows, key=lambda r: r.name))
            if sig == self._schedule_sig:
                return
            self._schedule_sig = sig
            for j in self.scheduler.get_jobs():
                j.remove()
            for r in rows:
                if r.enabled and r.cron and r.name in jobs.registry(self.demo):
                    self.scheduler.add_job(self._fire, cron_trigger(r.cron, r.timezone), args=[r.name, r.trading_days_only, r.timezone], id=r.name, replace_existing=True)
            log.info("schedules loaded", extra={"jobs": [j.id for j in self.scheduler.get_jobs()]})

    def catch_up(self) -> None:
        now = datetime.now(UTC)
        with session_scope() as db:
            for r in db.scalars(select(ScheduledJob).where(ScheduledJob.enabled.is_(True))).all():
                if not r.cron or r.name not in jobs.registry(self.demo):
                    continue
                last = db.scalar(select(func.max(JobRun.started_at)).where(JobRun.job_name == r.name, JobRun.status.in_(("success", "running", "failed"))))
                missed = missed_run(r.cron, r.timezone, r.trading_days_only, last, now)
                if missed:
                    try:
                        jobs.enqueue(db, r.name, {"missed_schedule": missed.isoformat()}, trigger="recovery")
                        db.commit()
                        log.info("queued recovery run", extra={"job": r.name, "missed": missed.isoformat()})
                    except jobs.AlreadyRunning:
                        pass

    def poll_queue(self) -> None:
        self.pool = [t for t in self.pool if t.is_alive()]
        if len(self.pool) >= 2:
            return
        claimed = jobs.claim_queued(self.demo)
        if claimed:
            run_id, job, params = claimed
            t = threading.Thread(target=jobs.execute, args=(run_id, job, params), daemon=True, name=f"job-{job.name}")
            t.start()
            self.pool.append(t)

    def heartbeat_file(self) -> None:
        with open("/tmp/worker-heartbeat", "w") as f:
            f.write(datetime.now(UTC).isoformat())

    def run(self) -> None:
        usage.install()
        with session_scope() as db:
            if not self.demo:
                from app.services.demo_service import demo_data_present
                if demo_data_present(db):
                    log.error("APP_MODE=production but the database contains SIMULATED demo data. "
                              "Run `python -m app.cli wipe-demo --yes` (or use a fresh database) before starting production.")
                    raise SystemExit(2)
            jobs.ensure_scheduled_jobs(db, self.demo)
        n = jobs.recover_interrupted()
        if n:
            log.warning("marked interrupted runs", extra={"count": n})
        self.load_schedules()
        self.scheduler.start()
        self.catch_up()
        log.info("worker started", extra={"mode": self.s.app_mode, "worker_id": jobs.WORKER_ID})
        last_reload = 0.0
        while not self.stop.is_set():
            try:
                self.poll_queue()
                if time.monotonic() - last_reload > 60:
                    self.load_schedules()
                    # A run whose heartbeat stopped (e.g. another worker was killed mid-job) would
                    # otherwise block that job forever; live runs heartbeat every 20 s.
                    if (n := jobs.recover_interrupted()):
                        log.warning("marked interrupted runs", extra={"count": n})
                    last_reload = time.monotonic()
                self.heartbeat_file()
            except Exception:
                log.exception("worker loop error")
            self.stop.wait(self.s.worker_poll_seconds)
        self.scheduler.shutdown(wait=False)
        for t in self.pool:
            t.join(timeout=30)
        log.info("worker stopped")


def main() -> None:
    configure_logging(get_settings().log_level)
    w = Worker()
    signal.signal(signal.SIGTERM, lambda *_: w.stop.set())
    signal.signal(signal.SIGINT, lambda *_: w.stop.set())
    w.run()


if __name__ == "__main__":
    main()
