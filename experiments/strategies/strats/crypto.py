"""BTC breakout pot: Jesse's Donchian example strategy on 4-hour Bitcoin bars, the one open-source bot rule that beat
holding in research/github_bots (2019-2026, BTC +52%/yr vs +49.5%, worst drop -50% vs -77%).

Rule, checked at every 4-hour close: buy when the close is above the previous 20 bars' highest high and above its
200-bar average; sell when a close is below the previous 20 bars' lowest low. Fills at the next bar's open, the whole
pot in BTC or in cash, COST per side.

Bars: Kraken's public OHLC (XBT/USD, 4 hours, no key), with Coinbase's hourly candles as the fallback. They are cached
in data/crypto/btc_4h.csv and the cache only grows, so the rule always runs over the same history. The pot's return
between two US closes is the rule's equity at the last 4-hour bar closed by the second close over the same at the first:
the rule uses only finished bars, so this is what trading it every 4 hours would have earned.
"""
from __future__ import annotations

import json
import logging
import time
from pathlib import Path

import numpy as np
import pandas as pd
import requests

log = logging.getLogger(__name__)
KEY = "btc_donchian"
CHANNEL, TREND = 20, 200
COST = 0.0025          # 25 bp a side: a Canadian retail exchange's taker fee and spread (research used 10 bp; still beat holding at 25)
BAR = pd.Timedelta(hours=4)
KRAKEN = "https://api.kraken.com/0/public/OHLC"
COINBASE = "https://api.exchange.coinbase.com/products/BTC-USD/candles"
KRAKEN_TICKER = "https://api.kraken.com/0/public/Ticker"
COINBASE_TICKER = "https://api.exchange.coinbase.com/products/BTC-USD/ticker"
UA = {"User-Agent": "experiment-1 research (https://github.com/pindustrycom/experiment-1)"}


def _kraken() -> pd.DataFrame:
    r = requests.get(KRAKEN, params={"pair": "XBTUSD", "interval": 240}, headers=UA, timeout=30)
    r.raise_for_status()
    j = r.json()
    if j.get("error"):
        raise RuntimeError(str(j["error"]))
    rows = next(v for k, v in j["result"].items() if k != "last")
    d = pd.DataFrame([x[:5] for x in rows], columns=["t", "open", "high", "low", "close"])
    d["t"] = pd.to_datetime(d["t"].astype(int), unit="s")
    return d.set_index("t").astype(float)


def _coinbase(hours: int = 1500) -> pd.DataFrame:
    end = pd.Timestamp.now(tz="UTC").floor("h").tz_localize(None)
    frames = []
    while hours > 0:
        start = end - pd.Timedelta(hours=min(hours, 300))
        r = requests.get(COINBASE, params={"granularity": 3600, "start": start.isoformat(), "end": end.isoformat()},
                         headers=UA, timeout=30)
        r.raise_for_status()
        frames.append(pd.DataFrame(r.json(), columns=["t", "low", "high", "open", "close", "volume"]))
        hours -= 300
        end = start
        time.sleep(0.4)
    h = pd.concat(frames)
    h["t"] = pd.to_datetime(h["t"].astype(int), unit="s")
    h = h.drop_duplicates("t").set_index("t").sort_index().astype(float)
    g = h.groupby(h.index.floor("4h"))
    d = pd.DataFrame({"open": g.open.first(), "high": g.high.max(), "low": g.low.min(), "close": g.close.last()})
    return d[g.size() == 4]          # only whole 4-hour bars


def finished(d: pd.DataFrame, now: pd.Timestamp | None = None) -> pd.DataFrame:
    """Drop the bar that is still forming (index = bar open time, UTC)."""
    now = now if now is not None else pd.Timestamp.now(tz="UTC").tz_localize(None)
    return d[d.index + BAR <= now]


def update(data: Path) -> dict:
    """Fetch the latest bars and merge them into the cache; a failed source keeps the cache."""
    f = data / "crypto" / "btc_4h.csv"
    f.parent.mkdir(parents=True, exist_ok=True)
    old = load(data)
    for name, fn in (("kraken", _kraken), ("coinbase", _coinbase)):
        try:
            new = finished(fn())
            break
        except Exception as ex:  # noqa: BLE001 - try the next source
            log.warning("btc bars from %s failed: %s", name, ex)
    else:
        return {"source": "none", "bars": len(old), "last": str(old.index[-1]) if len(old) else None}
    d = pd.concat([old[~old.index.isin(new.index)], new]).sort_index() if len(old) else new
    d.to_csv(f, index_label="t")
    return {"source": name, "bars": len(d), "last": str(d.index[-1])}


def load(data: Path) -> pd.DataFrame:
    f = data / "crypto" / "btc_4h.csv"
    if not f.exists():
        return pd.DataFrame(columns=["open", "high", "low", "close"], dtype=float)
    return pd.read_csv(f, index_col="t", parse_dates=["t"]).sort_index()


def us_close_bars(idx: pd.DatetimeIndex) -> np.ndarray:
    """True for each bar that is the last one closed by 16:00 New York on its day (where the daily stop resets)."""
    c = (idx + BAR).tz_localize("UTC").tz_convert("America/New_York")
    by_close = (c.hour * 60 + c.minute) <= 16 * 60
    nxt = np.r_[c[1:], [c[-1] + pd.Timedelta(days=1)]] if len(c) else c
    nxt = pd.DatetimeIndex(nxt)
    return np.asarray(by_close & ((nxt.date != c.date) | ((nxt.hour * 60 + nxt.minute) > 16 * 60)))


def run_rule(d: pd.DataFrame, cost: float = COST, stop: float | None = None) -> pd.DataFrame:
    """Per bar: the pot's equity at the bar's close (1 at the start), whether BTC was held through the bar, and whether
    it is held after the decision at the bar's close.
    stop: the daily stop loss (stops.py). Once a bar's low is stop below BTC's price at the last US close (or the entry
    price, if bought since), the pot sells at that level (or the bar's open, if it opened below) and stays in cash until
    the next US close; then it follows the rule again, buying back at the next bar's open if the rule is still long."""
    o, h, lo, c = (d[k].to_numpy(float) for k in ("open", "high", "low", "close"))
    hi = pd.Series(h).rolling(CHANNEL).max().shift().to_numpy()
    low = pd.Series(lo).rolling(CHANNEL).min().shift().to_numpy()
    sma = pd.Series(c).rolling(TREND).mean().to_numpy()
    mark = us_close_bars(d.index) if stop else np.zeros(len(c), bool)
    n = len(c)
    eq, held, after = np.ones(n), np.zeros(n, bool), np.zeros(n, bool)
    e, pos, want, out, ref = 1.0, False, False, False, 0.0
    for j in range(n):
        go = want and not out                 # the position decided at the last bar's close, filled at this open
        if go and not pos:
            e *= (1 - cost) * c[j] / o[j]
            pos, ref = True, o[j]
        elif pos and not go:
            e *= (1 - cost) * o[j] / c[j - 1]
            pos = False
        elif pos:
            e *= c[j] / c[j - 1]
        held[j] = pos
        if pos and stop and lo[j] <= ref * (1 - stop):
            e *= (1 - cost) * min(o[j], ref * (1 - stop)) / c[j]
            pos, out = False, True
        eq[j] = e
        if not want and c[j] > hi[j] and c[j] > sma[j]:
            want = True
        elif want and c[j] < low[j]:
            want = False
        if mark[j]:
            out, ref = False, c[j]
        after[j] = want and not out
    return pd.DataFrame({"equity": eq, "held": held, "after": after}, index=d.index)


def mark_bar(eq: pd.DataFrame, us_close: pd.Timestamp) -> pd.Timestamp | None:
    """The last 4-hour bar that had closed by 16:00 New York on the session day us_close."""
    t = (us_close.normalize() + pd.Timedelta(hours=16)).tz_localize("America/New_York").tz_convert("UTC").tz_localize(None)
    done = eq.index[eq.index + BAR <= t]
    return done[-1] if len(done) else None


def price_now() -> tuple[float, str] | None:
    """BTC's last traded price in USD: Kraken, then Coinbase."""
    for name, fn in (("kraken", lambda j: next(iter(j["result"].values()))["c"][0]), ("coinbase", lambda j: j["price"])):
        try:
            url, params = (KRAKEN_TICKER, {"pair": "XBTUSD"}) if name == "kraken" else (COINBASE_TICKER, None)
            r = requests.get(url, params=params, headers=UA, timeout=30)
            r.raise_for_status()
            return float(fn(r.json())), name
        except Exception as ex:  # noqa: BLE001 - try the next source
            log.warning("btc price from %s failed: %s", name, ex)
    return None


def tick(data: Path, journal: Path, stop: float | None = None, now: pd.Timestamp | None = None,
         price: tuple[float, str] | None = None, fetch: bool = True) -> dict:
    """One 4-hour run (btc-4h.yml, a few minutes after each 4-hour bar closes): bring the bars up to date, ask the rule
    for its position after the last finished bar, and if that differs from the journal's position, write the trade
    down at BTC's price right now. The pot's P/L still comes from the rule (step_btc); the journal shows the trades
    were placed on time and how far the live price was from the rule's fill at the next bar's open."""
    now = now if now is not None else pd.Timestamp.now(tz="UTC").tz_localize(None)
    rep = update(data) if fetch else {}
    j = json.loads(journal.read_text()) if journal.exists() else {"position": None, "trades": [], "runs": []}
    d = finished(load(data), now)
    if len(d) <= TREND:
        rep["tick"] = "not enough bars for the rule yet"
        return rep
    eq = run_rule(d, stop=stop)
    bar, want = eq.index[-1], bool(eq["after"].iloc[-1])
    at = now.strftime("%Y-%m-%d %H:%M")
    if j["position"] is None:                       # first run: take the rule's position as it stands, no trade
        j["position"] = want
        j["start"] = {"at": at, "bar": str(bar), "held": want}
        rep["tick"] = f"journal started {'holding BTC' if want else 'in cash'}"
    elif want != j["position"]:
        p = price if price is not None else (price_now() if fetch else None)
        px, src = p if p else (float(d["close"].iloc[-1]), "last bar close")
        j["trades"].append({"at": at, "bar": str(bar), "side": "buy" if want else "sell", "price": round(px, 2), "source": src})
        j["position"] = want
        rep["tick"] = f"{'bought' if want else 'sold'} BTC at {px:,.2f} ({src})"
    else:
        rep["tick"] = f"no trade: {'holding BTC' if want else 'in cash'}"
    j["runs"] = (j["runs"] + [{"at": at, "bar": str(bar), "held": want}])[-60:]
    journal.parent.mkdir(parents=True, exist_ok=True)
    journal.write_text(json.dumps(j, indent=1))
    rep |= {"bar": str(bar), "held": want, "trades": len(j["trades"])}
    return rep


def journal_rows(journal: dict | None, bars: pd.DataFrame | None, n: int = 10) -> list[dict]:
    """The latest live trades, each with the rule's fill (the open of the bar after the signal bar) when known."""
    if not journal:
        return []
    out = []
    for t in journal.get("trades", [])[-n:][::-1]:
        fill = None
        if bars is not None and len(bars):
            nxt = pd.Timestamp(t["bar"]) + BAR
            if nxt in bars.index:
                fill = float(bars.at[nxt, "open"])
        out.append(t | {"rule_price": fill, "gap": (t["price"] / fill - 1) if fill else None})
    return out


def position_text(row: dict | None) -> str:
    if not row:
        return "No position yet: the pot starts at the next close."
    if row.get("w"):
        return f"Holding BTC (rule checked at the 4-hour bar opening {row.get('bar', '?')} UTC)."
    return f"In cash: no breakout above the 20-bar high in an uptrend (as of the 4-hour bar opening {row.get('bar', '?')} UTC)."
