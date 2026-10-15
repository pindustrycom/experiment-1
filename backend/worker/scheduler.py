"""Schedule evaluation helpers (pure, unit-tested)."""
from __future__ import annotations

from datetime import datetime, timedelta
from zoneinfo import ZoneInfo

from apscheduler.triggers.cron import CronTrigger

from app.research.market_calendar import is_trading_day


def cron_trigger(expr: str, tz: str) -> CronTrigger:
    return CronTrigger.from_crontab(expr, timezone=ZoneInfo(tz))


def previous_fire_time(expr: str, tz: str, now: datetime, lookback: timedelta = timedelta(days=8)) -> datetime | None:
    trig = cron_trigger(expr, tz)
    t, prev = trig.get_next_fire_time(None, now - lookback), None
    while t is not None and t <= now:
        prev = t
        t = trig.get_next_fire_time(t, t + timedelta(seconds=1))
    return prev


def should_run_today(trading_days_only: bool, fire_time: datetime, tz: str = "America/New_York") -> bool:
    return (not trading_days_only) or is_trading_day(fire_time.astimezone(ZoneInfo(tz)).date())


def missed_run(expr: str, tz: str, trading_days_only: bool, last_started: datetime | None, now: datetime) -> datetime | None:
    """Return the scheduled time that was missed while the worker was down, if any."""
    prev = previous_fire_time(expr, tz, now)
    if prev is None or not should_run_today(trading_days_only, prev, tz):
        return None
    if last_started is None or last_started < prev:
        return prev
    return None
