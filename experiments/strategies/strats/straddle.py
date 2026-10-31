"""Straddle pot: each evening, buy at-the-money straddles on the S&P 500 stocks most likely to move 8% the next day.

The time-scale ladder study (research/timescale_ladder, big8.py) found that the size of tomorrow's move is predictable
(the stock flagged most each day moved 8%+ either way 14.6% of the time, 15x the base rate) but its direction is not.
A straddle (a call and a put at the same strike) pays on a big move either way, so it is the trade that fits. Whether
it pays after the option prices is the open question: no free history of stock option prices exists, so this pot
tests it going forward.

Each close: score every member with the flag model (FLAG_MODEL, a logistic regression on the ladder's volatility
layers, fitted on the point-in-time S&P 500 2000-2026; train_flag.py in the study), take the TOP_N best that have a usable
chain on Cboe's delayed quotes, and buy one at-the-money straddle on each in the nearest expiry at least MIN_DTE days
out, at the ask. At the next close, sell at the bid (a missing or zero bid is valued at what the options are worth if
exercised). ALLOC of the pot goes into the straddles each day, split equally; the rest stays in cash.

The news-tilted twin (NEWS_KEY) buys the same names on the same terms, but where today's headlines lean one way
(news.py, |lean| >= TILT_AT) it holds two of that side's option and one of the other: 2 calls + 1 put on a lean up,
1 call + 2 puts on a lean down. The news study (research/timescale_ladder/news) found headlines tell an 8% rise from
an 8% fall with AUC ~0.6; this pot tests whether that pays after option prices.
"""
from __future__ import annotations

import json
import logging
import time
from pathlib import Path

import numpy as np
import pandas as pd
import requests

from . import surface

log = logging.getLogger(__name__)
KEY = "straddle8"
NEWS_KEY = "straddle8_news"
KEYS = (KEY, NEWS_KEY)
TILT_AT = 0.2            # |news lean| from which the news pot doubles one side
TOP_N = 5                # straddles bought each close
SCAN = 15                # flagged names tried, best first, until TOP_N have a usable chain
ALLOC = 0.20             # share of the pot in straddles each day
MIN_DTE = 3              # calendar days to expiry at purchase: never an option that expires the next morning
MAX_SPREAD = 0.25        # skip a straddle whose bid-ask spread is over 25% of its mid
MIN_PRICE = 5.0
FLAG_MODEL = Path(__file__).with_name("flag_model.json")
FEATURES = ["lrv_d", "lrv_w", "lrv_m", "lrv_q", "range_vs_cc", "abs_gap", "abs_r_q", "vrel_q", "vrel_d", "vrel_w",
            "spy_lrv_w", "logp"]


def features(close: pd.DataFrame, opn: pd.DataFrame, high: pd.DataFrame, low: pd.DataFrame, volume: pd.DataFrame,
             spy: pd.Series) -> dict[str, pd.DataFrame]:
    """The ladder's layers for every stock and day (adjusted prices): the stock's own variance over a day, a week, a
    month and a quarter (intraday range plus overnight gap), the finest layer (today's range against its close-to-close
    move, the gap), the earnings cycle (biggest move and volume spike 60-66 sessions back), volume, and the market's."""
    r = np.log(close / close.shift(1))
    gap = np.log(opn / close.shift(1))
    pk = np.log(high / low) ** 2 / (4 * np.log(2))
    rv = (pk + gap.fillna(0) ** 2).clip(lower=1e-8)
    vrel = volume / volume.rolling(20, min_periods=10).mean().shift(1)
    sr = np.log(spy).diff()
    spy_lrv = np.log(sr.pow(2).rolling(5).mean().clip(lower=1e-8))
    f = {"lrv_d": np.log(rv), "lrv_w": np.log(rv.rolling(5).mean()), "lrv_m": np.log(rv.rolling(22).mean()),
         "lrv_q": np.log(rv.rolling(63).mean()),
         "range_vs_cc": np.log(pk.clip(lower=1e-8)) - np.log((r ** 2).clip(lower=1e-8)), "abs_gap": gap.abs(),
         "abs_r_q": r.abs().shift(60).rolling(7).max(), "vrel_q": vrel.shift(60).rolling(7).max(),
         "vrel_d": np.log(vrel.clip(lower=1e-3)), "vrel_w": np.log(vrel.rolling(5).mean().clip(lower=1e-3)),
         "logp": np.log(close)}
    f["spy_lrv_w"] = pd.DataFrame(np.repeat(spy_lrv.reindex(close.index).to_numpy()[:, None], close.shape[1], 1),
                                  index=close.index, columns=close.columns)
    return {k: v.replace([np.inf, -np.inf], np.nan) for k, v in f.items()}


def design(X: pd.DataFrame) -> np.ndarray:
    """Logistic inputs: the features and their squares (the chance of an 8% day rises faster than linearly with vol)."""
    Z = X[FEATURES].to_numpy(float)
    return np.column_stack([Z, Z ** 2])


def score(model: dict, X: pd.DataFrame) -> pd.Series:
    Z = np.clip((design(X) - np.array(model["mean"])) / np.array(model["std"]), -10, 10)  # an outlier day cannot overflow
    return pd.Series(1 / (1 + np.exp(-(Z @ np.array(model["coef"]) + model["intercept"]))), index=X.index)


def flags(bars: dict[str, pd.DataFrame], volume: pd.DataFrame, px: pd.DataFrame, d: pd.Timestamp, members: list[str],
          model: dict | None = None) -> pd.Series:
    """Members ranked by the model's chance of an 8%+ move (either way) over the next session, best first."""
    model = model or json.loads(FLAG_MODEL.read_text())
    cols = [m for m in members if m in px.columns and m in bars["open"].columns and m in volume.columns]
    win = px.index[px.index <= d][-90:]
    sub = {k: bars[k].reindex(index=win, columns=cols) for k in ("open", "high", "low")}
    F = features(px.reindex(index=win, columns=cols), sub["open"], sub["high"], sub["low"], volume.reindex(index=win, columns=cols),
                 px["SPY"].reindex(win))
    X = pd.DataFrame({k: v.iloc[-1] for k, v in F.items()})
    X = X[(px.loc[d, cols] > MIN_PRICE).reindex(X.index).fillna(False)].dropna()
    return score(model, X).sort_values(ascending=False) if len(X) else pd.Series(dtype=float)


def fetch_chain(ticker: str) -> tuple[pd.DataFrame, float, str] | None:
    try:
        r = requests.get(surface.URL.format(ticker.replace("-", ".")), headers=surface.UA, timeout=60)
        if r.status_code == 404:
            r = requests.get(surface.URL.format("_" + ticker.replace("-", ".")), headers=surface.UA, timeout=60)
        r.raise_for_status()
        ch, spot, day = surface.parse_cboe(r.json())
        return (ch, spot, day) if not ch.empty else None
    except Exception as ex:  # noqa: BLE001 - one missing chain only skips that name
        log.warning("chain %s: %s", ticker, ex)
        return None


def pick_straddle(ch: pd.DataFrame, spot: float, d: pd.Timestamp) -> dict | None:
    """The at-the-money straddle in the nearest expiry at least MIN_DTE days out, with both legs quoted, at the ask."""
    ch = ch[(ch["expiration"] - d).dt.days >= MIN_DTE]
    for exp in sorted(ch["expiration"].unique()):
        g = ch[ch["expiration"] == exp]
        both = g.pivot_table(index="strike", columns="option_type", values=["bid", "ask"], aggfunc="first").dropna()
        if both.empty:
            continue
        both = both[(both[("ask", "call")] > 0) & (both[("ask", "put")] > 0)]
        if both.empty:
            continue
        k = float(both.index[np.argmin(np.abs(both.index.to_numpy() - spot))])
        ask = float(both.at[k, ("ask", "call")] + both.at[k, ("ask", "put")])
        bid = float(both.at[k, ("bid", "call")] + both.at[k, ("bid", "put")])
        mid = (ask + bid) / 2
        if mid <= 0 or (ask - bid) / mid > MAX_SPREAD:
            return None
        return {"expiry": str(pd.Timestamp(exp).date()), "strike": k, "spot": round(spot, 4), "ask": round(ask, 4),
                "bid": round(bid, 4), "implied_move": round(ask / spot, 4),
                "call_ask": round(float(both.at[k, ("ask", "call")]), 4), "put_ask": round(float(both.at[k, ("ask", "put")]), 4)}
    return None


def value_at_bid(pos: dict, ch: pd.DataFrame | None, spot: float | None) -> tuple[float, str]:
    """What the held straddle sells for now: both legs at the bid, or what they are worth if exercised if a bid is missing.
    A tilted position holds pos["calls"] calls and pos["puts"] puts (1 and 1 when absent)."""
    nc, npu = pos.get("calls", 1), pos.get("puts", 1)
    intrinsic = (nc * max(spot - pos["strike"], 0) + npu * max(pos["strike"] - spot, 0)) if spot is not None else None
    if ch is not None:
        g = ch[(ch["expiration"] == pd.Timestamp(pos["expiry"])) & (np.isclose(ch["strike"], pos["strike"]))]
        bids = {t: g.loc[g["option_type"] == t, "bid"] for t in ("call", "put")}
        if all(len(b) and pd.notna(b.iloc[0]) for b in bids.values()):
            v = float(nc * bids["call"].iloc[0] + npu * bids["put"].iloc[0])
            if v > 0 or intrinsic is None:
                return max(v, intrinsic or 0.0), "bid"
    if intrinsic is not None:
        return intrinsic, "exercise value"
    return pos["ask"], "no quote: carried at cost"


def tilt(s: dict, lean: dict | None) -> dict:
    """The news pot's version of straddle s: two of the side the news leans to, one of the other; cost at the asks."""
    x = (lean or {}).get("lean")
    nc, npu = (2, 1) if x is not None and x >= TILT_AT else (1, 2) if x is not None and x <= -TILT_AT else (1, 1)
    ask = nc * s["call_ask"] + npu * s["put_ask"] if "call_ask" in s else s["ask"]
    return {**s, "calls": nc, "puts": npu, "ask": round(ask, 4), "implied_move": round(ask / s["spot"], 4),
            "lean": x, "headlines": (lean or {}).get("n", 0)}


def book_names(ranked: pd.Series | None, chains: dict[str, tuple | None], d: pd.Timestamp) -> list[str]:
    """The TOP_N names bought this close (best flagged first, with a usable straddle)."""
    out = []
    for t in (ranked.index[:SCAN] if ranked is not None else []):
        c = chains.get(t)
        if c and pick_straddle(c[0], c[1], d):
            out.append(t)
        if len(out) >= TOP_N:
            break
    return out


def step(rows: list[dict], last: dict | None, d: pd.Timestamp, ds: str, ranked: pd.Series | None,
         chains: dict[str, tuple | None], leans: dict[str, dict] | None = None) -> str:
    """Sell yesterday's straddles at the bid, then buy today's. chains: ticker -> fetch_chain() result for every name
    held or scanned this close. leans (news pot only): ticker -> news.leans() entry; the book is tilted by it."""
    trades, ret = [], 0.0
    for t, pos in (last or {}).get("straddles", {}).items():
        c = chains.get(t)
        v, how = value_at_bid(pos, c[0] if c else None, c[1] if c else None)
        rr = v / pos["ask"] - 1
        trades.append({"ticker": t, **pos, "exit": round(v, 4), "exit_how": how, "ret": round(rr, 4),
                       "stock_move": round(c[1] / pos["spot"] - 1, 4) if c else None})
        ret += pos["alloc"] * rr
    book = {}
    for t in (ranked.index[:SCAN] if ranked is not None else []):
        if len(book) >= TOP_N:
            break
        c = chains.get(t)
        s = pick_straddle(c[0], c[1], d) if c else None
        if s:
            book[t] = {**s, "p8": round(float(ranked[t]), 4)}
            if leans is not None:
                book[t] = tilt(book[t], leans.get(t))
    for s in book.values():
        s["alloc"] = round(ALLOC / len(book), 6)
    idx = (last["index"] if last else 1.0) * (1 + ret)
    rows.append({"date": ds, "ret": ret, "index": idx, "w": {t: s["alloc"] for t, s in book.items()}, "reb": True,
                 "straddles": book, "closed": trades})
    sold = ", ".join(f"{x['ticker']} {x['ret']:+.0%}" for x in trades) or "nothing"
    return f"{ds}: {ret:+.4%}; sold {sold}; bought {', '.join(book) or 'none (no usable chains)'}"


def tickers_needed(last: dict | None, ranked: pd.Series | None) -> list[str]:
    held = list((last or {}).get("straddles", {}))
    return held + [t for t in (ranked.index[:SCAN] if ranked is not None else []) if t not in held]


def fetch_chains(names: list[str], pause: float = 0.4) -> dict[str, tuple | None]:
    out = {}
    for t in names:
        out[t] = fetch_chain(t)
        time.sleep(pause)
    return out


def position_text(row: dict | None) -> str:
    if not row:
        return "No straddles yet: the first ones are bought at the next close."
    b = row.get("straddles") or {}
    if not b:
        return f"In cash after the {row['date']} close: no flagged stock had a usable option chain."
    legs = ", ".join(f"{t} {s['strike']:g} ({s['expiry']}, {_legs(s)}costs {s['implied_move']:.1%} of the stock, "
                     f"P(8% move) {s['p8']:.0%})" for t, s in b.items())
    return f"From the {row['date']} close: {sum(s['alloc'] for s in b.values()):.0%} of the pot in straddles on " + legs


def _legs(s: dict) -> str:
    if "calls" not in s:
        return ""
    lean = "no news lean" if s.get("lean") is None else f"news lean {s['lean']:+.1f}"
    return f"{s['calls']} call{'s' * (s['calls'] > 1)} + {s['puts']} put{'s' * (s['puts'] > 1)}, {lean}, "
