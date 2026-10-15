"""Scheduler, market calendar, point-in-time cutoffs across DST, and the crash-safe job runner."""
from __future__ import annotations

from datetime import UTC, date, datetime, timedelta
from zoneinfo import ZoneInfo

import pandas as pd
import pytest

from app.research.market_calendar import add_trading_days, is_trading_day, last_completed_session, nyse_holidays
from app.scoring.pit import truncate_market
from tests.conftest import requires_db
from worker.scheduler import cron_trigger, missed_run, previous_fire_time, should_run_today

NY = ZoneInfo("America/New_York")


# ------------------------------------------------------------------ calendar
def test_nyse_holidays_2026():
    h = nyse_holidays(2026)
    for d in [date(2026, 1, 1), date(2026, 1, 19), date(2026, 2, 16), date(2026, 4, 3), date(2026, 5, 25), date(2026, 6, 19),
              date(2026, 7, 3), date(2026, 9, 7), date(2026, 11, 26), date(2026, 12, 25)]:
        assert d in h, d
    assert len(h) == 10


def test_holiday_observance_rules():
    assert date(2027, 12, 24) in nyse_holidays(2027)  # Christmas on Saturday -> Friday
    assert date(2022, 12, 31) not in nyse_holidays(2022)  # New Year on Saturday is not observed on Dec 31
    assert date(2023, 1, 2) in nyse_holidays(2023)  # New Year on Sunday -> Monday
    assert date(2021, 6, 18) not in nyse_holidays(2021)  # Juneteenth only from 2022
    assert not is_trading_day(date(2025, 1, 9))  # national day of mourning


def test_last_completed_session_around_close_and_holidays():
    assert last_completed_session(datetime(2026, 10, 14, 15, 59, tzinfo=NY)) == date(2026, 10, 13)
    assert last_completed_session(datetime(2026, 10, 14, 16, 0, tzinfo=NY)) == date(2026, 10, 14)
    assert last_completed_session(datetime(2026, 11, 27, 9, 0, tzinfo=NY)) == date(2026, 11, 25)  # Thanksgiving
    assert add_trading_days(date(2026, 12, 23), 2) == date(2026, 12, 28)


# ------------------------------------------------------------------ point-in-time across DST
@pytest.mark.parametrize("cutoff_utc, expected_last", [
    (datetime(2026, 3, 9, 19, 59, tzinfo=UTC), "2026-03-06"),  # EDT: 15:59 ET on Mar 9 -> Monday close not yet known
    (datetime(2026, 3, 9, 20, 0, tzinfo=UTC), "2026-03-09"),   # 16:00 EDT
    (datetime(2026, 11, 2, 20, 30, tzinfo=UTC), "2026-10-30"),  # EST: 15:30 ET
    (datetime(2026, 11, 2, 21, 0, tzinfo=UTC), "2026-11-02"),   # 16:00 EST
])
def test_truncate_market_respects_dst(cutoff_utc, expected_last):
    idx = pd.to_datetime(["2026-03-06", "2026-03-09", "2026-10-30", "2026-11-02"])
    s = pd.Series([1.0, 2.0, 3.0, 4.0], index=idx)
    s = s[s.index <= pd.Timestamp(expected_last) + pd.Timedelta(days=10)]
    out = truncate_market(s, cutoff_utc)
    assert str(out.index.max().date()) == expected_last


# ------------------------------------------------------------------ scheduler
def test_cron_uses_new_york_time_across_dst():
    trig = cron_trigger("30 17 * * 1-5", "America/New_York")
    summer = trig.get_next_fire_time(None, datetime(2026, 10, 30, 12, tzinfo=UTC))
    winter = trig.get_next_fire_time(None, datetime(2026, 11, 2, 12, tzinfo=UTC))
    assert summer.astimezone(UTC).hour == 21  # 17:30 EDT
    assert winter.astimezone(UTC).hour == 22  # 17:30 EST
    with pytest.raises(ValueError):
        cron_trigger("every day", "America/New_York")


def test_trading_day_jobs_skip_holidays():
    good_friday = datetime(2026, 4, 3, 17, 30, tzinfo=NY)
    assert not should_run_today(True, good_friday)
    assert should_run_today(False, good_friday)


def test_missed_run_detection():
    now = datetime(2026, 10, 15, 9, 0, tzinfo=NY)
    prev = previous_fire_time("30 17 * * 1-5", "America/New_York", now)
    assert prev == datetime(2026, 10, 14, 17, 30, tzinfo=NY)
    assert missed_run("30 17 * * 1-5", "America/New_York", True, None, now) == prev
    assert missed_run("30 17 * * 1-5", "America/New_York", True, prev + timedelta(minutes=1), now) is None
    # the fire on Thanksgiving is not a missed run for a trading-day job
    tg = datetime(2026, 11, 27, 8, 0, tzinfo=NY)
    assert missed_run("30 17 * * 1-5", "America/New_York", True, None, tg) is None


@requires_db
def test_next_run_skips_holidays_for_trading_day_jobs(db):
    from sqlalchemy import select

    from app.api.routes.jobs import next_fire
    from app.models import ScheduledJob
    r = db.scalar(select(ScheduledJob).where(ScheduledJob.name == "ingest_prices"))
    nxt = next_fire(r, datetime(2026, 11, 26, 6, 0, tzinfo=NY))  # Thanksgiving morning
    assert nxt.astimezone(NY).date() == date(2026, 11, 27)


# ------------------------------------------------------------------ job runner
@pytest.fixture()
def fake_jobs(monkeypatch):
    from app.services import jobs
    calls = {"ok": 0, "flaky": 0}

    def ok(db, params):
        calls["ok"] += 1
        return {"rows": 3, "params": params}

    def flaky(db, params):
        calls["flaky"] += 1
        if calls["flaky"] < 3:
            raise RuntimeError("temporary failure")
        return {"done": True}

    def broken(db, params):
        raise RuntimeError("always broken")

    reg = {
        "t_ok": jobs.JobDef("t_ok", ok, "0 * * * *", "test"),
        "t_flaky": jobs.JobDef("t_flaky", flaky, None, "test", max_attempts=3),
        "t_broken": jobs.JobDef("t_broken", broken, None, "test", max_attempts=2),
    }
    monkeypatch.setattr(jobs, "registry", lambda demo: reg)
    return calls


@requires_db
def test_run_now_records_success(db, fake_jobs):
    from app.services import jobs
    run = jobs.run_now("t_ok", trigger="manual", params={"x": 1})
    assert run.status == "success" and run.attempts == 1 and run.result["rows"] == 3 and run.duration_ms is not None


@requires_db
def test_retries_with_exponential_backoff(db, fake_jobs):
    from app.services import jobs
    sleeps: list[float] = []
    run = jobs.run_now("t_flaky", trigger="manual", sleep=sleeps.append)
    assert run.status == "success" and run.attempts == 3
    assert sleeps == [5.0, 10.0]
    run = jobs.run_now("t_broken", trigger="manual", sleep=sleeps.append)
    assert run.status == "failed" and run.attempts == 2 and "always broken" in run.error


@requires_db
def test_no_duplicate_concurrent_runs(db, fake_jobs):
    from sqlalchemy import select

    from app.models import JobRun
    from app.services import jobs
    db.add(JobRun(job_name="t_ok", trigger="schedule", status="running", params={}, started_at=datetime.now(UTC), heartbeat_at=datetime.now(UTC)))
    db.commit()
    assert jobs.run_now("t_ok", trigger="schedule") is None
    assert fake_jobs["ok"] == 0
    skipped = db.scalars(select(JobRun).where(JobRun.status == "skipped")).all()
    assert len(skipped) == 1 and "already running" in skipped[0].error
    # and at most one queued row per job
    jobs.enqueue(db, "t_ok")
    db.commit()
    with pytest.raises(jobs.AlreadyRunning):
        jobs.enqueue(db, "t_ok")


@requires_db
def test_paused_scheduler_skips_scheduled_but_not_manual(db, fake_jobs):
    from app.services import jobs, settings_service
    settings_service.put_section(db, "bot", {"paused": True}, None)
    db.commit()
    assert jobs.run_now("t_ok", trigger="schedule") is None
    assert jobs.run_now("t_ok", trigger="manual").status == "success"


@requires_db
def test_claim_queued_and_restart_recovery(db, fake_jobs):
    from sqlalchemy import select

    from app.models import JobRun
    from app.services import jobs
    jobs.enqueue(db, "t_ok", {"a": 1})
    db.commit()
    claimed = jobs.claim_queued(False)
    assert claimed is not None and claimed[2] == {"a": 1}
    assert jobs.claim_queued(False) is None  # nothing left
    # simulate a crash: heartbeat went stale
    db.execute(JobRun.__table__.update().where(JobRun.id == claimed[0]).values(heartbeat_at=datetime.now(UTC) - timedelta(minutes=10)))
    db.commit()
    assert jobs.recover_interrupted() == 1
    db.expire_all()
    assert db.scalar(select(JobRun).where(JobRun.id == claimed[0])).status == "interrupted"
    # unknown job names in the queue fail cleanly instead of crashing the worker
    db.add(JobRun(job_name="mystery", trigger="manual", status="queued", params={}))
    db.commit()
    assert jobs.claim_queued(False) is None
    db.expire_all()
    assert db.scalar(select(JobRun).where(JobRun.job_name == "mystery")).status == "failed"


@requires_db
def test_queue_is_not_blocked_by_a_running_job(db, fake_jobs):
    from app.models import JobRun
    from app.services import jobs
    db.add(JobRun(job_name="t_ok", trigger="schedule", status="running", params={}, started_at=datetime.now(UTC), heartbeat_at=datetime.now(UTC)))
    db.commit()
    jobs.enqueue(db, "t_ok")
    jobs.enqueue(db, "t_flaky")
    db.commit()
    claimed = jobs.claim_queued(False)
    assert claimed is not None and claimed[1].name == "t_flaky"


@requires_db
def test_migrations_downgrade_and_upgrade(migrated):
    """The schema can be rolled back and re-applied (run last: it rebuilds the schema)."""
    from alembic import command
    from sqlalchemy import inspect

    from app.core.db import get_engine
    from tests.conftest import _alembic_config
    cfg = _alembic_config()
    get_engine().dispose()
    command.downgrade(cfg, "base")
    assert "pairs" not in inspect(get_engine()).get_table_names()
    command.upgrade(cfg, "head")
    assert {"pairs", "prediction_snapshots", "job_runs"} <= set(inspect(get_engine()).get_table_names())


@requires_db
def test_price_ingest_fails_loudly_when_every_request_fails(db, monkeypatch):
    from datetime import date

    from app.data_sources.base import ProviderError
    from app.services import prices_service

    class Down:
        name, is_total_return = "stooq", False

        def configured(self):
            return True

        def fetch_daily(self, ticker, start, end):
            raise ProviderError("stooq", "HTTP 404", status=404, retryable=False)
    monkeypatch.setitem(prices_service.PROVIDERS, "stooq", Down)
    with pytest.raises(ProviderError, match="every price request failed; first error: .*HTTP 404"):
        prices_service.ingest_prices(db, ["stooq"], date(2009, 1, 1))
