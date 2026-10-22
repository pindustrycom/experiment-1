"""Risk-premia study: do the returns that research says you are paid for (taking risks others avoid) still pay, and
does a diversified mix of them beat holding SPY?

python premia.py CACHE OUT

Part A, long history (1990 on): the market, Fama-French value, profitability, investment and size, momentum, and
Cboe's put-write and buy-write indexes, by decade and before / after the paper that made each one famous.
Part B, tradeable since 2012 and since 2019: sleeves (SPY, put-write, French factors with a cost haircut, FX carry,
VIX carry, trend, long bonds, gold) mixed four ways, against SPY and 60/40.

Every sleeve is a daily total return (cash included). Portfolio weights use only the trailing year of data, set at
each month-end close; 2 bp a side to rebalance between sleeves. Paper only, nothing is traded."""
from __future__ import annotations

import json
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "kakushadze151"))
import run as k151
import strategies as S
from engine import backtest, stats

from premia_data import cboe, french, yahoo

# What implementing each French factor costs a year, roughly (turnover x spread, plus borrow on the short side).
HAIRCUT = {"SMB": 0.01, "HML": 0.01, "RMW": 0.01, "CMA": 0.01, "UMD": 0.02}
PUT_HAIRCUT = 0.005          # option spreads on the monthly roll (WisdomTree's PUTW charged 0.44%)
PUBLISHED = {"SMB": 1981, "HML": 1992, "UMD": 1993, "RMW": 2015, "CMA": 2015, "PUT": 2004, "BXM": 2002}
DECADES = [("1990-99", "1990", "1999"), ("2000-09", "2000", "2009"), ("2010-18", "2010", "2018"), ("2019-26", "2019", "2026")]
ETFS = ["SPY", "IEF", "TLT", "GLD", "BIL", "VIXY", "VIXM", "FXE", "FXB", "FXA", "FXY", "FXC", "FXF",
        "EFA", "EEM", "SLV", "USO", "UNG", "DBA", "CPER"]


def sharpe(x: pd.Series, per_year=252) -> float:
    return float(x.mean() / x.std() * np.sqrt(per_year)) if x.std() > 0 else float("nan")


def load(cache: str) -> dict:
    ff = french(cache)
    idx = {s: cboe(s, cache) for s in ("PUT", "BXM")}
    raw = {t: yahoo(t, cache, start="2005-01-01") for t in ETFS + list(k151.FX)}
    return {"french": ff, "cboe": idx, "raw": {k: v for k, v in raw.items() if v is not None}}


def part_a(ff: pd.DataFrame, cb: dict) -> dict:
    """Excess returns of each long-history premium, by decade and before / after publication."""
    ex = ff[["Mkt-RF", "SMB", "HML", "RMW", "CMA", "UMD"]].copy()
    for f, h in HAIRCUT.items():
        ex[f] = ex[f] - h / 252
    for s, ser in cb.items():
        if ser is not None:
            r = ser.pct_change().reindex(ff.index)
            ex[s] = r - ff["RF"] - PUT_HAIRCUT / 252
    full = ex
    ex = ex[ex.index >= "1990-01-01"]
    out = {}
    for c in ex.columns:
        x = ex[c].dropna()
        xf = full[c].dropna()           # publication split uses the whole history (French data starts 1963)
        row = {"years": f"{x.index[0].year}-{x.index[-1].year}", "excess_yr": float(x.mean() * 252), "sharpe": sharpe(x),
               "corr_mkt": float(x.corr(ex["Mkt-RF"])) if c != "Mkt-RF" else 1.0}
        for name, a, b in DECADES:
            xd = x[a:b]
            row[name] = sharpe(xd) if len(xd) > 200 else float("nan")
        if c in PUBLISHED:
            y = PUBLISHED[c]
            row["sharpe_before_pub"] = sharpe(xf[: str(y)]) if len(xf[: str(y)]) > 500 else float("nan")
            row["sharpe_after_pub"] = sharpe(xf[str(y + 1):])
        out[c] = row
    return out


def sleeves(data: dict) -> tuple[pd.DataFrame, pd.Series]:
    """Daily total returns of every tradeable sleeve on the NYSE calendar, plus the T-bill return."""
    raw = data["raw"]
    cal = raw["SPY"].index
    ohlc = {}
    for t, d in raw.items():
        if t in k151.FX:
            name, inv = k151.FX[t]
            d = k151.to_calendar(d, cal)
            if inv:
                d = pd.DataFrame({"open": 1 / d.open, "high": 1 / d.low, "low": 1 / d.high, "close": 1 / d.close})
            ohlc[name] = d
        else:
            ohlc[t] = d.reindex(cal)
    close, ret, rf, rd = k151.build({"ohlc": ohlc, "rates": pd.DataFrame()})
    out = {}
    for t in ("SPY", "IEF", "TLT", "GLD"):
        out[t] = ret[t]

    def run(W):
        W = W.reindex(ret.index)
        W = W.loc[:, [c for c in W.columns if c in ret.columns]]
        r = backtest(W, ret, rf, cost_bp=k151.cost_bp, borrow_yr=k151.borrow).ret
        live = W.fillna(0).abs().sum(axis=1).gt(0)
        return r.where(r.index > live.idxmax()) if live.any() else r * np.nan

    fxc = [c for c in ("EUR", "GBP", "AUD", "JPY", "CAD", "CHF") if c in close]
    if not rd.empty:
        out["FX carry"] = run(S.fx_carry(rd[fxc + ["USD"]], close[fxc]))
    out["VIX carry"] = run(S.vol_carry_etns(close))
    fut = close[[t for t in k151.FUT if t in close]]
    out["Trend"] = run(S.trend_multi(fut))
    ff = data["french"]
    if ff is not None:
        f = ff.reindex(cal)
        for c, h in HAIRCUT.items():
            out[c] = f[c] + f["RF"] - h / 252
    for s, ser in data["cboe"].items():
        if ser is not None:
            out[s] = ser.reindex(cal).pct_change(fill_method=None) - PUT_HAIRCUT / 252
    return pd.DataFrame(out), rf


def mix(sl: pd.DataFrame, method: str, target_vol=0.10, cap=2.0, base: dict | None = None) -> pd.DataFrame:
    """Month-end target weights over the sleeves that have a full trailing year."""
    W = pd.DataFrame(np.nan, index=sl.index, columns=sl.columns)
    for d in S.month_ends(sl.index):
        i = sl.index.get_loc(d)
        if i < 252:
            continue
        win = sl.iloc[i - 251:i + 1]
        ok = win.columns[win.notna().all()]
        if base:
            ok = [c for c in ok if c not in base]
        if len(ok) == 0:
            continue
        vol = win[ok].std() * np.sqrt(252)
        w = pd.Series(0.0, index=sl.columns)
        if method == "equal":
            w[ok] = 1 / len(ok)
        else:
            iv = 1 / vol
            w[ok] = iv / iv.sum()
            if method == "riskparity":
                pv = float(np.sqrt(w[ok] @ win[ok].cov().to_numpy() @ w[ok] * 252))
                w = w * min(target_vol / pv, cap)
        if base:
            w = w * (1 - sum(base.values()))
            for k, v in base.items():
                w[k] = v
        W.loc[d] = w
    return W


def part_b(sl: pd.DataFrame, rf: pd.Series, start: str) -> dict:
    spy = sl["SPY"].fillna(0)
    s = sl[sl.index >= pd.Timestamp(start) - pd.Timedelta(days=400)]
    end = min(s[c].last_valid_index() for c in s if s[c].last_valid_index() is not None)   # French data lags ~1 month
    s = s[s.index <= end]
    spy_vol = float(spy[start:].std() * np.sqrt(252))
    books = {
        "Hold SPY": pd.DataFrame({"SPY": 1.0}, index=s.index),
        "60/40 SPY/IEF": mix(s, "equal", base={"SPY": 0.6, "IEF": 0.4}),
        "Equal weight, all sleeves": mix(s, "equal"),
        "Inverse vol, all sleeves": mix(s, "inverse"),
        "Risk parity at 10% vol (max 2x)": mix(s, "riskparity", 0.10),
        f"Risk parity at SPY's vol {spy_vol:.0%} (max 3x)": mix(s, "riskparity", spy_vol, 3.0),
        "70% SPY + 30% inverse-vol premia": mix(s, "inverse", base={"SPY": 0.7}),
    }
    ret = s.fillna(0.0)
    res = {}
    bench = spy[spy.index >= start]
    for name, W in books.items():
        r = backtest(W, ret, rf, cost_bp=2.0, borrow_yr=0.0).ret
        r = r[r.index >= start]
        st = stats(r, bench)
        res[name] = {k: st[k] for k in ("cagr", "vol", "sharpe", "max_dd", "beta", "alpha_ann", "alpha_t")}
        last = W.dropna(how="all")
        if len(last):
            res[name]["last_weights"] = {k: round(float(v), 3) for k, v in last.iloc[-1].items() if abs(v) > 1e-4}
    # each sleeve on its own over the same window
    solo = {}
    for c in s.columns:
        x = s[c][s.index >= start].dropna()
        if len(x) > 250:
            st = stats(x, bench.reindex(x.index).fillna(0))
            solo[c] = {k: st[k] for k in ("cagr", "vol", "sharpe", "max_dd", "beta", "alpha_t")}
            solo[c]["corr_spy"] = float(x.corr(bench.reindex(x.index)))
    m = (1 + s[s.index >= start]).resample("ME").prod() - 1
    corr = m.corr().round(2)
    return {"window": f"{start} to {end.date()}", "portfolios": res, "sleeves": solo, "corr_monthly": corr.to_dict()}


def report(a: dict, bs: dict, out: str) -> str:
    L = ["# Risk premia: what still pays, and does a mix beat SPY?", "",
         "## A. Long history: Sharpe of each premium's excess return (after the cost haircut)", "",
         "| premium | data | excess/yr | Sharpe | corr to market | 1990-99 | 2000-09 | 2010-18 | 2019-26 | before paper | after paper |",
         "|" + "---|" * 11]
    for c, r in a.items():
        L.append(f"| {c} | {r['years']} | {r['excess_yr']:+.1%} | {r['sharpe']:.2f} | {r['corr_mkt']:.2f} | "
                 + " | ".join(f"{r[n]:.2f}" for n, _, _ in DECADES)
                 + f" | {r.get('sharpe_before_pub', float('nan')):.2f} | {r.get('sharpe_after_pub', float('nan')):.2f} |")
    for b in bs:
        L += ["", f"## B. Portfolios, {b['window']}", "",
              "| portfolio | CAGR | vol | Sharpe | worst drop | beta | alpha/yr | alpha t |", "|" + "---|" * 8]
        for n, r in sorted(b["portfolios"].items(), key=lambda kv: -kv[1]["sharpe"]):
            L.append(f"| {n} | {r['cagr']:+.1%} | {r['vol']:.1%} | {r['sharpe']:.2f} | {r['max_dd']:.0%} | {r['beta']:.2f} | "
                     f"{r['alpha_ann']:+.1%} | {r['alpha_t']:.1f} |")
        L += ["", "Sleeves alone:", "", "| sleeve | CAGR | vol | Sharpe | worst drop | corr to SPY | alpha t |", "|" + "---|" * 7]
        for n, r in sorted(b["sleeves"].items(), key=lambda kv: -kv[1]["sharpe"]):
            L.append(f"| {n} | {r['cagr']:+.1%} | {r['vol']:.1%} | {r['sharpe']:.2f} | {r['max_dd']:.0%} | {r['corr_spy']:.2f} | "
                     f"{r['alpha_t']:.1f} |")
    md = "\n".join(L)
    os.makedirs(out, exist_ok=True)
    with open(f"{out}/table.md", "w") as f:
        f.write(md + "\n")
    with open(f"{out}/results.json", "w") as f:
        json.dump({"long_history": a, "portfolios": bs}, f, indent=1, default=float)
    return md


def main(cache: str, out: str):
    np.seterr(all="ignore")
    data = load(cache)
    ff = data["french"]
    print(f"french: {None if ff is None else (ff.index[0].date(), ff.index[-1].date())}")
    for s, ser in data["cboe"].items():
        print(f"cboe {s}: {None if ser is None else (ser.index[0].date(), ser.index[-1].date())}")
        if ser is not None:
            yr = ser.resample("YE").last().pct_change().dropna()
            print("  yearly:", {d.year: round(float(v), 3) for d, v in yr.items()})
    a = part_a(ff, data["cboe"]) if ff is not None else {}
    sl, rf = sleeves(data)
    print("sleeves start:", {c: str(sl[c].first_valid_index().date()) for c in sl if sl[c].first_valid_index() is not None})
    bs = [part_b(sl, rf, "2012-01-03"), part_b(sl, rf, "2019-01-02")]
    print(report(a, bs, out))


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
