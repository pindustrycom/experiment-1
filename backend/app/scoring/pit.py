"""Point-in-time helpers: everything handed to a scorer passes through these filters."""
from __future__ import annotations

from datetime import UTC, datetime, time
from zoneinfo import ZoneInfo

import pandas as pd

NY = ZoneInfo("America/New_York")


def close_timestamp(d) -> datetime:
    """A daily close for date d is knowable from 16:00 America/New_York on d."""
    return datetime.combine(pd.Timestamp(d).date(), time(16, 0), tzinfo=NY).astimezone(UTC)


def truncate_market(prices: pd.DataFrame | pd.Series, cutoff: datetime):
    """Drop every bar whose close was not yet known at `cutoff`."""
    if len(prices) == 0:
        return prices
    local = cutoff.astimezone(NY)
    last_known = local.date() if local.time() >= time(16, 0) else local.date() - pd.Timedelta(days=1)
    return prices.loc[prices.index <= pd.Timestamp(last_known)]
