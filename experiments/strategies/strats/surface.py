"""Implied volatility surfaces from the day's option chains (SPY, QQQ, IWM), worked back from the option prices.

The chains are Cboe's free delayed quotes, the same source as the daily options summary (openbb-ideas.yml, which
keeps only a few numbers per ticker). Here the whole chain is kept and, for every expiry:
  1. the forward price and discount factor come from put-call parity, C - P = D (F - K), fitted on the strikes near
     the money, so no interest rate or dividend guess is needed;
  2. each out-of-the-money option's mid price (puts below the forward, calls above) is turned back into the volatility
     that makes Black's formula give that price (bisection);
  3. the vols are put on a grid of moneyness (strike / forward) by expiry, which is the surface drawn in 3D.
Quotes with no bid, a spread wider than the mid, or a price under intrinsic value are dropped. Cboe's own implied vol
column is kept only to check ours against it.
"""
from __future__ import annotations

import gzip
import io
import json
import logging
import re
import time
from pathlib import Path

import numpy as np
import pandas as pd
import requests

log = logging.getLogger(__name__)
TICKERS = ("SPY", "QQQ", "IWM")
URL = "https://cdn.cboe.com/api/global/delayed_quotes/options/{}.json"
UA = {"User-Agent": "experiment-1 research (https://github.com/pindustrycom/experiment-1)"}
KEEP_DAYS = 30                      # chains kept in the state, newest first
MIN_DTE, MAX_DTE = 7, 400
GRID = np.round(np.arange(0.80, 1.101, 0.02), 2)    # strike / forward
OCC = re.compile(r"^(?P<root>[A-Z.]+?)(?P<ymd>\d{6})(?P<cp>[CP])(?P<k>\d{8})$")
COLS = ["expiration", "option_type", "strike", "bid", "ask", "iv", "open_interest", "volume"]



def _ncdf(x: np.ndarray) -> np.ndarray:
    """Standard normal CDF (Abramowitz and Stegun 7.1.26, error under 1.5e-7: far below a quote's spread)."""
    z = np.abs(x) / np.sqrt(2)
    t = 1 / (1 + 0.3275911 * z)
    y = 1 - (((((1.061405429 * t - 1.453152027) * t) + 1.421413741) * t - 0.284496736) * t + 0.254829592) * t * np.exp(-z * z)
    return 0.5 * (1 + np.sign(x) * y)


def parse_cboe(payload: dict) -> tuple[pd.DataFrame, float, str]:
    """(chain rows, underlying price, quote date) from Cboe's delayed quotes JSON."""
    d = payload["data"]
    rows = []
    for o in d.get("options") or []:
        m = OCC.match(str(o.get("option", "")))
        if not m:
            continue
        rows.append({"expiration": pd.Timestamp("20" + m["ymd"]), "option_type": "call" if m["cp"] == "C" else "put",
                     "strike": int(m["k"]) / 1000, **{k: o.get(k) for k in ("bid", "ask", "iv", "open_interest", "volume")}})
    spot = float(d.get("current_price") or d.get("close"))
    ts = payload.get("timestamp")     # New York time, e.g. "2026-10-23 16:15:00"
    day = pd.Timestamp(ts).date() if ts else pd.Timestamp.now(tz="America/New_York").date()
    return pd.DataFrame(rows, columns=COLS), spot, str(day)


def fetch(data: Path, tickers: tuple[str, ...] = TICKERS) -> dict:
    """Today's chains into data/chains/<date>/<ticker>.csv.gz (with the spot in chains/<date>/spot.json); keeps KEEP_DAYS dates."""
    out, spots = {}, {}
    for t in tickers:
        try:
            r = requests.get(URL.format(t), headers=UA, timeout=60)
            r.raise_for_status()
            ch, spot, day = parse_cboe(r.json())
            if ch.empty:
                out[t] = "empty chain"
                continue
            d = data / "chains" / day
            d.mkdir(parents=True, exist_ok=True)
            ch.to_csv(d / f"{t}.csv.gz", index=False)
            spots[t] = spot
            sf = d / "spot.json"
            sf.write_text(json.dumps({**(json.loads(sf.read_text()) if sf.exists() else {}), t: spot}))
            out[t] = f"{day}: {len(ch)} contracts, spot {spot}"
        except Exception as ex:  # noqa: BLE001 - a day without a chain only leaves the last surface up
            out[t] = f"failed: {type(ex).__name__}: {str(ex)[:120]}"
        time.sleep(0.5)
    days = sorted(p.name for p in (data / "chains").glob("????-??-??")) if (data / "chains").exists() else []
    for old in days[:-KEEP_DAYS]:
        for f in (data / "chains" / old).iterdir():
            f.unlink()
        (data / "chains" / old).rmdir()
    return out


def latest(data: Path, ticker: str) -> tuple[pd.DataFrame, float, str] | None:
    days = sorted(p.name for p in (data / "chains").glob("????-??-??") if (p / f"{ticker}.csv.gz").exists()) if (data / "chains").exists() else []
    if not days:
        return None
    d = data / "chains" / days[-1]
    with gzip.open(d / f"{ticker}.csv.gz", "rt") as fh:
        ch = pd.read_csv(io.StringIO(fh.read()), parse_dates=["expiration"])
    return ch, float(json.loads((d / "spot.json").read_text())[ticker]), days[-1]


def black(F: np.ndarray, K: np.ndarray, T: np.ndarray, D: np.ndarray, vol: np.ndarray, call: np.ndarray) -> np.ndarray:
    sd = vol * np.sqrt(T)
    d1 = (np.log(F / K) + 0.5 * sd ** 2) / sd
    d2 = d1 - sd
    c = D * (F * _ncdf(d1) - K * _ncdf(d2))
    return np.where(call, c, c - D * (F - K))


def implied_vol(price, F, K, T, D, call, lo: float = 1e-3, hi: float = 4.0, iters: int = 60) -> np.ndarray:
    """The vol that makes Black's formula give price (bisection; NaN where no vol in [lo, hi] fits)."""
    price, F, K, T, D, call = (np.asarray(x, dtype=float if i < 5 else bool) for i, x in enumerate((price, F, K, T, D, call)))
    a, b = np.full(price.shape, lo), np.full(price.shape, hi)
    ok = (black(F, K, T, D, a, call) <= price) & (black(F, K, T, D, b, call) >= price)
    for _ in range(iters):
        m = 0.5 * (a + b)
        up = black(F, K, T, D, m, call) < price
        a, b = np.where(up, m, a), np.where(up, b, m)
    return np.where(ok, 0.5 * (a + b), np.nan)


def forward(g: pd.DataFrame, spot: float) -> tuple[float, float] | None:
    """(F, D) for one expiry from put-call parity on the strikes within 5% of the spot (or the 8 nearest) that have both
    quotes. Mids move in cents, so on SPY's one-dollar strikes a fit over a few strikes alone can put D above 1: a fit a
    little above 1 is taken as 1 (rates near zero for the expiry), anything far off drops the expiry."""
    q = g[(g["bid"] > 0) & (g["ask"] > 0)].assign(mid=lambda x: (x["bid"] + x["ask"]) / 2)
    c = q[q["option_type"] == "call"].set_index("strike")["mid"]
    p = q[q["option_type"] == "put"].set_index("strike")["mid"]
    both = c.index.intersection(p.index)
    if len(both) < 3:
        return None
    near = sorted(both, key=lambda x: abs(x - spot))
    k = np.array([x for x in near if abs(x / spot - 1) <= 0.05] if sum(abs(x / spot - 1) <= 0.05 for x in near) >= 8 else near[:8], dtype=float)
    y = (c.loc[k] - p.loc[k]).to_numpy()
    slope, _ = np.polyfit(k, y, 1)
    D = min(-slope, 1.0)
    if not 0.8 < -slope < 1.02:
        return None
    return float(np.mean(y / D + k)), float(D)


def surface(ch: pd.DataFrame, spot: float, day: str) -> dict:
    """Implied vols per expiry from mid prices, the grid for the 3D view, smiles, the ATM term structure and the check
    against Cboe's own vols."""
    today = pd.Timestamp(day)
    ch = ch.assign(dte=(pd.to_datetime(ch["expiration"]) - today).dt.days)
    ch = ch[(ch["dte"] >= MIN_DTE) & (ch["dte"] <= MAX_DTE)]
    pts, smiles, term, dropped = [], {}, [], {"no forward": 0, "few quotes": 0}
    for exp, g in ch.groupby("expiration"):
        fd = forward(g, spot)
        if fd is None:
            dropped["no forward"] += 1
            continue
        F, D = fd
        dte = int(g["dte"].iloc[0])
        T = dte / 365
        q = g[(g["bid"] > 0) & (g["ask"] > g["bid"])].copy()
        q["mid"] = (q["bid"] + q["ask"]) / 2
        q = q[(q["ask"] - q["bid"]) <= q["mid"]]
        q = q[((q["option_type"] == "put") & (q["strike"] < F)) | ((q["option_type"] == "call") & (q["strike"] >= F))]
        q = q[(q["strike"] / F > 0.5) & (q["strike"] / F < 1.5)]
        if len(q) < 5:
            dropped["few quotes"] += 1
            continue
        call = (q["option_type"] == "call").to_numpy()
        q["our_iv"] = implied_vol(q["mid"], np.full(len(q), F), q["strike"], np.full(len(q), T), np.full(len(q), D), call)
        q = q.dropna(subset=["our_iv"])
        if len(q) < 5:
            dropped["few quotes"] += 1
            continue
        q = q.sort_values("strike")
        m = (q["strike"] / F).to_numpy()
        iv = q["our_iv"].to_numpy()
        row = [float(np.interp(x, m, iv)) if m[0] <= x <= m[-1] else None for x in GRID]
        pts.append({"expiration": str(pd.Timestamp(exp).date()), "dte": dte, "forward": round(F, 2), "discount": round(D, 5),
                    "iv": [None if v is None else round(v, 4) for v in row], "n": int(len(q))})
        term.append({"dte": dte, "atm": round(float(np.interp(1.0, m, iv)), 4)})
        smiles[dte] = [[round(float(a), 4), round(float(b), 4)] for a, b in zip(m, iv) if 0.7 <= a <= 1.15]
        if "iv" in q and q["iv"].notna().any():
            pts[-1]["cboe_gap"] = round(float((q["our_iv"] - q["iv"]).abs()[q["iv"] > 0].median()), 4)
    if not pts:
        return {}
    gaps = [p["cboe_gap"] for p in pts if p.get("cboe_gap") is not None]
    near = min(pts, key=lambda p: abs(p["dte"] - 30))
    pick = sorted({min(pts, key=lambda p: abs(p["dte"] - x))["dte"] for x in (10, 30, 90, 180, 365)})
    i90, i110 = int(np.argmin(np.abs(GRID - 0.90))), int(np.argmin(np.abs(GRID - 1.10)))
    return {"date": day, "spot": spot, "grid": [float(x) for x in GRID], "expiries": pts, "term": term,
            "smiles": {str(k): smiles[k] for k in pick},
            "skew_30d": None if near["iv"][i90] is None or near["iv"][i110] is None else round(near["iv"][i90] - near["iv"][i110], 4),
            "atm_30d": next((t["atm"] for t in term if t["dte"] == near["dte"]), None),
            "contracts": int(sum(p["n"] for p in pts)), "dropped_expiries": dropped, "cboe_median_gap": round(float(np.median(gaps)), 4) if gaps else None}


def options_history(f: Path | None, tickers: tuple[str, ...] = TICKERS) -> dict:
    """The daily summary that openbb-ideas.yml collects (options_daily.csv): ATM vol, skew and put/call by day."""
    if f is None or not f.exists():
        return {}
    d = pd.read_csv(f)
    out = {}
    for t in tickers:
        g = d[d["ticker"] == t].sort_values("date")
        out[t] = [{k: (None if pd.isna(r[k]) else (r[k] if k == "date" else round(float(r[k]), 4)))
                   for k in ("date", "spot", "iv30_atm", "iv90_atm", "skew30", "pc_oi")} for _, r in g.iterrows()]
    out["names"] = int(d[d["date"] == d["date"].max()]["ticker"].nunique()) if len(d) else 0
    return out
