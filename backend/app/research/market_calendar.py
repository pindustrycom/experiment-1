"""NYSE trading calendar (full-day closures) with US/Eastern timezone handling.

Implements the NYSE holiday rules in force since 1998 plus Juneteenth (observed from 2022):
New Year's Day, Martin Luther King Jr. Day, Washington's Birthday, Good Friday, Memorial Day,
Juneteenth, Independence Day, Labor Day, Thanksgiving Day, Christmas Day.
Observance: Saturday holidays move to Friday (except New Year's Day, which is not observed on the
preceding Dec 31 under NYSE Rule 7.2), Sunday holidays move to Monday.
Unscheduled closures (e.g. national days of mourning) are listed explicitly in SPECIAL_CLOSURES.
Early-close half days are treated as trading days.
"""
from __future__ import annotations

from datetime import date, datetime, time, timedelta
from functools import lru_cache
from zoneinfo import ZoneInfo

from dateutil.easter import easter

NY = ZoneInfo("America/New_York")
MARKET_CLOSE = time(16, 0)

SPECIAL_CLOSURES = {
    date(2001, 9, 11), date(2001, 9, 12), date(2001, 9, 13), date(2001, 9, 14),
    date(2004, 6, 11),  # Reagan
    date(2007, 1, 2),  # Ford
    date(2012, 10, 29), date(2012, 10, 30),  # Hurricane Sandy
    date(2018, 12, 5),  # G.H.W. Bush
    date(2025, 1, 9),  # Carter
}


def _nth_weekday(year: int, month: int, weekday: int, n: int) -> date:
    d = date(year, month, 1)
    d += timedelta(days=(weekday - d.weekday()) % 7)
    return d + timedelta(weeks=n - 1)


def _last_weekday(year: int, month: int, weekday: int) -> date:
    d = date(year, month + 1, 1) - timedelta(days=1) if month < 12 else date(year, 12, 31)
    return d - timedelta(days=(d.weekday() - weekday) % 7)


def _observed(d: date, allow_saturday_shift: bool = True) -> date | None:
    if d.weekday() == 5:
        return d - timedelta(days=1) if allow_saturday_shift else None
    if d.weekday() == 6:
        return d + timedelta(days=1)
    return d


@lru_cache(maxsize=256)
def nyse_holidays(year: int) -> frozenset[date]:
    hols: set[date | None] = {
        _observed(date(year, 1, 1), allow_saturday_shift=False),
        _nth_weekday(year, 1, 0, 3),  # MLK
        _nth_weekday(year, 2, 0, 3),  # Washington's Birthday
        easter(year) - timedelta(days=2),  # Good Friday
        _last_weekday(year, 5, 0),  # Memorial Day
        _observed(date(year, 7, 4)),
        _nth_weekday(year, 9, 0, 1),  # Labor Day
        _nth_weekday(year, 11, 3, 4),  # Thanksgiving
        _observed(date(year, 12, 25)),
    }
    if year >= 2022:
        hols.add(_observed(date(year, 6, 19)))
    hols |= {d for d in SPECIAL_CLOSURES if d.year == year}
    return frozenset(h for h in hols if h is not None)


def is_trading_day(d: date) -> bool:
    return d.weekday() < 5 and d not in nyse_holidays(d.year)


def previous_trading_day(d: date) -> date:
    d -= timedelta(days=1)
    while not is_trading_day(d):
        d -= timedelta(days=1)
    return d


def last_completed_session(now: datetime) -> date:
    """Most recent session whose 16:00 ET close has passed at `now` (tz-aware)."""
    local = now.astimezone(NY)
    d = local.date()
    if is_trading_day(d) and local.time() >= MARKET_CLOSE:
        return d
    return previous_trading_day(d)


def trading_days_between(start: date, end: date) -> list[date]:
    out, d = [], start
    while d <= end:
        if is_trading_day(d):
            out.append(d)
        d += timedelta(days=1)
    return out


def add_trading_days(d: date, n: int) -> date:
    """Calendar date that is `n` NYSE sessions after `d` (expected maturity of an n-day horizon)."""
    cur, k = d, 0
    while k < n:
        cur += timedelta(days=1)
        if is_trading_day(cur):
            k += 1
    return cur
