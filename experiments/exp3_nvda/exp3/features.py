"""The daily state measurements for NVDA, built only from what was public at each session's close.

Three groups, as in the design:
- price: NVDA's returns, volatility and trend;
- market cap index: NVDA's market cap (close x shares outstanding as filed, carried through splits)
  relative to its own year, and the cap-weighted Nasdaq-100 (QQQ) and S&P 500 (SPY);
- sentiment by category, each as last week's average and a long memory of everything said
  (exponentially weighted, half-life about a quarter), plus how much was said last week:
    news         headlines' sentiment for the stock
    saturation   what headlines and posts say about NVIDIA's markets saturating or expanding
    speculation  Reddit posts' sentiment (public speculation)
    brokers      analysts' upgrades, downgrades and price-target changes

A category with nothing said in a window is *unobserved* (NaN), not neutral; the EKF skips it.
Timing: Reddit posts and analyst actions count for the first close at or after their timestamp.
News search results often carry only a date, so a headline counts from the close *after* its
publication date. The target is the 5-session log return after the close.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from . import config

NY = "America/New_York"
PRICE_COLS = ["r1", "r5", "r20", "rv20", "ma200"]
MARKET_COLS = ["mcap_dev", "mcap_chg20", "qqq_r1", "qqq_r5", "spy_r5"]
SENT_COLS = [f"{k}_{c}" for c in config.CATEGORIES for k in ("week", "long", "count")]
TARGET = "y"

BUY = {"buy", "strong buy", "outperform", "overweight", "positive", "sector outperform", "market outperform", "accumulate", "add",
       "conviction buy", "top pick", "long-term buy", "speculative buy", "moderate buy"}
HOLD = {"hold", "neutral", "equal-weight", "equal weight", "market perform", "sector perform", "peer perform", "in-line", "perform",
        "sector weight", "mixed", "market weight", "moderate hold"}
SELL = {"sell", "underperform", "underweight", "reduce", "negative", "strong sell", "sector underperform", "market underperform", "moderate sell"}


def grade(g: str) -> float | None:
    g = (g or "").strip().lower()
    return 1.0 if g in BUY else 0.0 if g in HOLD else -1.0 if g in SELL else None


def _s(x) -> str:
    return x if isinstance(x, str) else ""


def broker_score(row) -> float:
    """An analyst action as bullish (+) or bearish (-), in -1..1."""
    g = grade(_s(row["to_grade"]))
    s = 0.5 * g if g is not None else 0.0
    a = _s(row["action"]).lower()
    s += 1.0 if a == "up" else -1.0 if a == "down" else 0.0
    pt, prior = row.get("pt"), row.get("pt_prior")
    if pd.notna(pt) and pd.notna(prior) and prior and pt:
        s += 0.5 * np.sign(pt - prior)
    else:
        p = _s(row.get("pt_action")).lower()
        s += 0.5 if p == "raises" else -0.5 if p == "lowers" else 0.0
    return float(np.clip(s / 1.5, -1, 1))


def closes(sessions: pd.DatetimeIndex) -> pd.DatetimeIndex:
    return (pd.DatetimeIndex(sessions).tz_localize(NY) + pd.Timedelta(hours=16)).tz_convert("UTC")


def assign_sessions(ts: pd.Series, sessions: pd.DatetimeIndex) -> pd.Series:
    """First session whose 16:00 New York close is at or after each timestamp (NaT if none)."""
    c = closes(sessions)
    t = pd.to_datetime(ts, utc=True)
    idx = np.searchsorted(c.values, t.values, side="left")
    out = pd.Series(pd.NaT, index=ts.index, dtype="datetime64[ns]")
    ok = idx < len(sessions)
    out[ok] = sessions.values[idx[ok]]
    return out


def news_available(ts: pd.Series) -> pd.Series:
    """End of the New York calendar day of publication: the headline counts from the next close."""
    t = pd.to_datetime(ts, utc=True).dt.tz_convert(NY)
    return (t.dt.normalize() + pd.Timedelta(hours=23, minutes=59, seconds=59)).dt.tz_convert("UTC")


def category_items(news: pd.DataFrame, reddit: pd.DataFrame, analysts: pd.DataFrame, scores: dict, sessions: pd.DatetimeIndex,
                   relevance_min: float) -> pd.DataFrame:
    """One row per (item, category) with its session, value in -1..1 and weight."""
    rows = []

    def scored(df, avail):
        if df is None or df.empty:
            return pd.DataFrame()
        d = df.copy()
        d["relevance"] = d["id"].map(lambda i: scores.get(i, {}).get("relevance"))
        d["sentiment"] = d["id"].map(lambda i: scores.get(i, {}).get("sentiment"))
        d["saturation"] = d["id"].map(lambda i: scores.get(i, {}).get("saturation"))
        d = d.dropna(subset=["relevance"])
        d = d[d["relevance"] >= relevance_min]
        d["session"] = assign_sessions(avail(d["ts"]), sessions)
        return d.dropna(subset=["session"])

    n = scored(news, news_available)
    r = scored(reddit, lambda s: pd.to_datetime(s, utc=True))
    if len(n):
        rows.append(pd.DataFrame({"session": n["session"], "cat": "news", "v": n["sentiment"], "w": n["relevance"]}))
    if len(r):
        rows.append(pd.DataFrame({"session": r["session"], "cat": "speculation", "v": r["sentiment"], "w": r["relevance"]}))
    both = pd.concat([x for x in (n, r) if len(x)]) if len(n) or len(r) else pd.DataFrame()
    if len(both):
        sat = both[both["saturation"].abs() > 0]
        rows.append(pd.DataFrame({"session": sat["session"], "cat": "saturation", "v": sat["saturation"], "w": sat["relevance"]}))
    if analysts is not None and len(analysts):
        a = analysts.copy()
        a["session"] = assign_sessions(a["ts"], sessions)
        a = a.dropna(subset=["session"])
        rows.append(pd.DataFrame({"session": a["session"], "cat": "brokers", "v": a.apply(broker_score, axis=1), "w": 1.0}))
    if not rows:
        return pd.DataFrame(columns=["session", "cat", "v", "w"])
    return pd.concat(rows, ignore_index=True)


def sentiment_features(items: pd.DataFrame, sessions: pd.DatetimeIndex, week: int, halflife: int) -> pd.DataFrame:
    out = pd.DataFrame(index=sessions)
    for c in config.CATEGORIES:
        it = items[items["cat"] == c]
        num = (it["v"] * it["w"]).groupby(it["session"]).sum().reindex(sessions, fill_value=0.0)
        den = it["w"].groupby(it["session"]).sum().reindex(sessions, fill_value=0.0)
        cnt = it.groupby("session").size().reindex(sessions, fill_value=0).astype(float)
        wn, wd = num.rolling(week, min_periods=1).sum(), den.rolling(week, min_periods=1).sum()
        out[f"week_{c}"] = (wn / wd).where(wd > 0)
        ln = num.ewm(halflife=halflife, adjust=False).mean()
        ld = den.ewm(halflife=halflife, adjust=False).mean()
        seen = den.cumsum() > 0
        out[f"long_{c}"] = (ln / ld).where(seen & (ld > 1e-9))
        out[f"count_{c}"] = np.log1p(cnt.rolling(week, min_periods=1).sum())
    return out


def market_cap(px: pd.DataFrame, shares: pd.DataFrame) -> pd.Series:
    """Close x shares outstanding known at the time (latest filing before the session), with the
    share count multiplied by any split since that filing."""
    dates = px.index
    sh = shares.copy()
    sh["filed"] = pd.to_datetime(sh["filed"])
    sh = sh.sort_values("filed")
    pos = np.searchsorted(sh["filed"].values, dates.values, side="left") - 1    # filed strictly before the session
    known = pd.Series(np.where(pos >= 0, sh["shares"].to_numpy(float)[np.clip(pos, 0, None)], np.nan), index=dates)
    filed_at = pd.Series(np.where(pos >= 0, sh["filed"].values[np.clip(pos, 0, None)], np.datetime64("NaT", "ns")), index=dates)
    cum = px["split_factor"].astype(float).cumprod().to_numpy()                       # cumulative splits up to each session
    at = np.searchsorted(dates.values, filed_at.values, side="right") - 1               # last session on or before the filing
    cum_at_filing = np.where(at >= 0, cum[np.clip(at, 0, None)], 1.0)
    cum = pd.Series(cum, index=dates)
    cum_at_filing = pd.Series(cum_at_filing, index=dates)
    return px["close"] * known * (cum / cum_at_filing)


def build(prices: dict[str, pd.DataFrame], shares: pd.DataFrame, items: pd.DataFrame, cfg: config.Config) -> pd.DataFrame:
    px = prices[cfg.ticker]
    sessions = px.index
    lp = np.log(px["adj_close"].astype(float))
    df = pd.DataFrame(index=sessions)
    r = 100 * lp.diff()
    df["r1"] = r
    df["r5"] = r.rolling(5).sum()
    df["r20"] = r.rolling(20).sum()
    df["rv20"] = r.rolling(20).std()
    df["ma200"] = 100 * (lp - np.log(px["adj_close"].rolling(200).mean()))
    lm = np.log(market_cap(px, shares))
    df["log_mcap"] = lm
    df["mcap_dev"] = 100 * (lm - lm.rolling(250, min_periods=120).mean())
    df["mcap_chg20"] = 100 * (lm - lm.shift(20))
    for t, name in (("QQQ", "qqq"), ("SPY", "spy")):
        m = 100 * np.log(prices[t]["adj_close"].astype(float).reindex(sessions)).diff()
        df[f"{name}_r1"] = m
        df[f"{name}_r5"] = m.rolling(5).sum()
    df = df.join(sentiment_features(items, sessions, cfg.week, cfg.long_halflife))
    df[TARGET] = 100 * (lp.shift(-cfg.horizon) - lp)
    df["close"] = px["adj_close"].astype(float)
    return df


class Scaler:
    """Robust standardisation fitted on the training rows only; values are clipped at 5 sd. NaN stays NaN."""

    def fit(self, x: np.ndarray) -> Scaler:
        self.med = np.nanmedian(x, axis=0)
        iqr = np.nanpercentile(x, 75, axis=0) - np.nanpercentile(x, 25, axis=0)
        self.scale = np.where(iqr > 1e-9, iqr / 1.349, np.where(np.nanstd(x, axis=0) > 1e-9, np.nanstd(x, axis=0), 1.0))
        return self

    def transform(self, x: np.ndarray) -> np.ndarray:
        return np.clip((x - self.med) / self.scale, -5, 5)
