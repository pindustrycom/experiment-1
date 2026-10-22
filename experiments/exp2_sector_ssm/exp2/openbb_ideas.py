"""Three uses of the free data that OpenBB opens up (Joaquin, 2026-10-19). Paper research only: nothing is traded.

  1. vol products   Experiment 2's best sleeve is a short SPY variance swap, which a retail account cannot trade. The same
                    timing rule (exp2/collapse.py vol_carry: VIX^2 above the HAR variance forecast and the Markov filter
                    calm) is applied to things that can be traded: SVXY (inverse VIX futures ETF), a short VIXY position,
                    and a short second-month VIX future held one month (Cboe's VX curve through OpenBB).
  2. yield curve    the 10-year minus 3-month Treasury spread (Federal Reserve H.15 through OpenBB) as a regime signal:
                    does it improve the vol forecast, and does switching the vol sleeve and the SPY regime filter off
                    while the curve is inverted help?
  3. options        free option chains are only a snapshot of today, so there is no history to backtest. options_snapshot()
                    records each day's implied vol, skew and put/call ratios per stock (run daily by openbb-ideas.yml);
                    they can be tested once a few months have accumulated.

Everything that is chosen is chosen on 2005-2018 (or from when a product exists); 2019 to today is the test, as in
collapse.py. Sleeve returns are monthly, in % of capital, indexed by the month they are earned.
"""
from __future__ import annotations

import json
import logging
import math
import time
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import pandas as pd

from . import collapse, papers, volpred
from .papers import TRAIN_END, sstats

log = logging.getLogger("exp2.openbb_ideas")
ETPS = ["SVXY", "VIXY", "^VIX3M"]
VEGA = 0.5             # % of capital per vol point, as the variance swap in collapse.vol_carry
VX_COST = 0.1          # vol points to trade a VX future in and out
ETP_COST_BP = 10.0     # per unit traded, each side
BORROW_PCT_YEAR = 3.0  # to borrow VIXY for a short
SVXY_CHANGE = pd.Period("2018-03", "M")    # SVXY went from -1x to -0.5x daily VIX futures on 2018-02-28


# ---------------------------------------------------------------- data

def fetch_extra(data: Path, start: str = "2004-01-01") -> dict:
    """Yahoo: SVXY, VIXY, VIX3M. OpenBB: the VX futures curve at each month end (Cboe) and the Treasury curve (Fed)."""
    rep = {"yahoo": papers.update(data, ETPS, max_age_days=0.5)}
    from openbb import obb
    try:
        spy = papers.load(data, ["SPY"])["SPY"]
        ends = spy.groupby(spy.index.to_period("M")).tail(1).index
        ends = [d.strftime("%Y-%m-%d") for d in ends if d >= pd.Timestamp("2006-01-01")]
        vx = vx_history(ends)
        vx.to_csv(data / "vx_curve_monthly.csv", index=False)
        rep["vx_curve"] = {"rows": len(vx), "first": str(vx["date"].min()), "last": str(vx["date"].max())}
    except Exception as e:  # noqa: BLE001 - report and carry on with what there is
        rep["vx_curve"] = f"failed: {type(e).__name__}: {str(e)[:300]}"
    try:
        tr = obb.fixedincome.government.treasury_rates(provider="federal_reserve", start_date=start).to_df().reset_index()
        tr.to_csv(data / "treasury_curve.csv", index=False)
        rep["treasury_curve"] = {"rows": len(tr), "first": str(tr["date"].min()), "last": str(tr["date"].max())}
    except Exception as e:  # noqa: BLE001
        rep["treasury_curve"] = f"failed: {type(e).__name__}: {str(e)[:300]}"
    return rep


def vx_history(dates: list[str]) -> pd.DataFrame:
    """The VX futures curve on each date, one row per contract: date, expiration, symbol, price. OpenBB's
    derivatives.futures.curve pivots several dates into one column per date and drops the expirations, so this calls
    the Cboe helper underneath it (openbb_cboe.utils.vix, pinned in requirements.txt) to keep them."""
    import asyncio

    from openbb_cboe.utils.vix import get_vx_by_date
    df = asyncio.run(get_vx_by_date(date=dates, vx_type="eod", use_cache=False))
    return df.reset_index() if "date" not in df.columns else df


def timing(data: Path, train_end: str = TRAIN_END) -> dict:
    """The inputs of collapse.vol_carry: SPY, VIX, T-bill, the Markov turbulence probability and the HAR forecast."""
    px = papers.load(data, ["SPY", "^VIX", "^IRX"])
    spy, vix = px["SPY"]["adj_close"], px["^VIX"]["close"]
    r = np.log(spy).diff()
    fit = (r.index <= pd.Timestamp(train_end)) & r.notna().to_numpy()
    ph, _ = collapse.hmm2(r.fillna(0).to_numpy(), fit)
    p_hi = pd.Series(ph, index=spy.index)
    _, sdf = volpred.spy_timing(data, train_end)
    rf = (px["^IRX"]["close"].reindex(spy.index).ffill() / 100 / 252).fillna(0.0)
    K = vix.groupby(vix.index.to_period("M")).last()
    fvar = np.exp(2 * sdf["fvol"]) * 252 * 1e4
    ph_m = p_hi.groupby(p_hi.index.to_period("M")).last()
    rich = (K ** 2 > fvar.reindex(K.index))
    calm = ph_m.reindex(K.index) < 0.5
    return {"spy": spy, "vix": vix, "rf": rf, "p_hi": p_hi, "sdf": sdf, "rich": rich, "calm": calm, "on": rich & calm}


def _next(s: pd.Series) -> pd.Series:
    """A position decided at the end of month m earns month m + 1."""
    return collapse._shift(s)


def etp_month(px: pd.DataFrame) -> pd.Series:
    return collapse.month_ret(px["adj_close"].pct_change().dropna())


# ---------------------------------------------------------------- 1. vol products

def vx_roll(curve: pd.DataFrame) -> pd.Series:
    """Short the second VX contract at each month end and buy it back at the next month end (by then the front month):
    P&L in vol points, indexed by the month it is earned. Matching by expiration month keeps it the same contract."""
    if "price" not in curve.columns:
        raise ValueError(f"VX curve has no price column: {list(curve.columns)[:12]}")
    c = curve.dropna(subset=["price"]).copy()
    c["m"] = pd.to_datetime(c["date"]).dt.to_period("M")
    c["exp"] = c["expiration"].astype(str).str[:7]
    out = {}
    months = sorted(c["m"].unique())
    for m, m1 in zip(months[:-1], months[1:]):
        if m1 != m + 1:
            continue
        now = c[c["m"] == m].sort_values("exp")
        if len(now) < 2:
            continue
        exp2nd, f0 = now["exp"].iloc[1], float(now["price"].iloc[1])
        later = c[(c["m"] == m1) & (c["exp"] == exp2nd)]
        if len(later):
            out[m1] = f0 - float(later["price"].iloc[0])
    return pd.Series(out, dtype=float).sort_index()


def vol_products(data: Path, T: dict) -> tuple[dict, dict]:
    """Sleeves in % of capital per month. ETPs: excess return over T-bills while held, flat otherwise; held at weight 1
    (the portfolio step rescales). VX: VEGA % of capital per vol point, like the variance swap."""
    on = T["on"].astype(float)
    rf_m = collapse.month_ret(T["rf"])
    px = papers.load(data, ["SVXY", "VIXY"])
    sl, notes = {}, {}
    for name, sign, extra in (("svxy", 1.0, 0.0), ("shortvixy", -1.0, BORROW_PCT_YEAR / 12)):
        t = "SVXY" if name == "svxy" else "VIXY"
        if t not in px:
            notes[name] = f"{t} prices missing"
            continue
        r = etp_month(px[t])
        # long SVXY: return over T-bills; short VIXY: minus VIXY's return, less the borrow fee (short rebate ignored)
        ex = r - rf_m.reindex(r.index).fillna(0) if sign > 0 else -r - extra
        held = _next(on).reindex(ex.index).fillna(0.0)
        trade = held.diff().abs().fillna(held)
        sl[f"{name}_always"] = ex                                           # bought once and held
        sl[f"{name}_timed"] = held * ex - trade * ETP_COST_BP / 100
    f = data / "vx_curve_monthly.csv"
    if f.exists():
        try:
            roll = vx_roll(pd.read_csv(f))
            held = _next(on).reindex(roll.index).fillna(0.0)
            pnl = VEGA * (roll - VX_COST)
            sl["vxfut_always"], sl["vxfut_timed"] = pnl, pnl.where(held > 0, 0.0)
            notes["vx_months"] = int(len(roll))
        except Exception as e:  # noqa: BLE001 - keep the other products
            notes["vxfut"] = f"{type(e).__name__}: {str(e)[:300]}"
    else:
        notes["vxfut"] = "no VX curve file"
    # the synthetic variance swap, rebuilt here so every sleeve uses the same timing and dates
    vc = collapse.vol_carry(T["spy"], T["vix"], T["p_hi"], T["sdf"])
    sl["varswap_always"], sl["varswap_timed"] = _next(vc["varswap_always"]), _next(vc["varswap_timed"])
    return sl, notes


def period_stats(s: pd.Series, train_end: str = TRAIN_END) -> dict:
    s = s.dropna()
    cut = pd.Period(pd.Timestamp(train_end), "M")
    out = {"from": str(s.index.min()) if len(s) else None}
    for k, part in (("before_test", s[(s.index <= cut) & (s.index.year >= collapse.FIRST_YEAR)]), ("test", s[s.index > cut])):
        st = sstats(part)
        if len(part) >= 12:
            cum = np.log1p(part.clip(lower=-99) / 100).cumsum()
            st["max_drawdown_pct"] = float(100 * (1 - np.exp((cum - cum.cummax()).min())))
            st["worst_month_pct"] = float(part.min())
        out[k] = st
    for m in ("2018-02", "2020-03", "2024-08", "2025-04"):
        p = pd.Period(m, "M")
        out[f"month_{m}"] = None if p not in s.index else round(float(s[p]), 2)
    return out


def portfolio_with(ref: pd.DataFrame, swap: dict[str, pd.Series] | None, extra: dict[str, pd.Series] | None = None,
                   train_end: str = TRAIN_END) -> dict:
    """collapse.combine on the reference sleeves, with the variance swap replaced by another vol product and/or extra
    variants added; returns the min-variance portfolio's test numbers and the sleeves it chose."""
    S = {k: ref[k] for k in ref.columns}
    if swap is not None:
        S["varswap_always"], S["varswap_timed"] = swap["always"], swap["timed"]
    S.update(extra or {})
    res, _ = collapse.combine(S, train_end)
    p = res["portfolios"].get("min_variance", {})
    keep = ("leverage", "sharpe", "mean_pct_month", "vol_pct_year", "max_drawdown_pct", "pre_test_sharpe", "t", "n")
    return {"chosen": res["chosen"], "weights": res["weights"].get("min_variance"), **{k: p.get(k) for k in keep},
            "spy_test": res["spy_test"]}


# ---------------------------------------------------------------- 2. yield curve

def curve_spread(data: Path) -> pd.Series | None:
    f = data / "treasury_curve.csv"
    if not f.exists():
        return None
    tr = pd.read_csv(f, parse_dates=["date"]).set_index("date").sort_index()
    s = (tr["year_10"] - tr["month_3"]) * 100              # OpenBB gives decimals; percentage points
    return s.dropna()


def vol_forecast_with_curve(sdf: pd.DataFrame, spread_m: pd.Series, train_end: str = TRAIN_END) -> dict:
    """volpred.spy_timing's HAR regression (expanding window, refit monthly on the past), with and without the spread
    and its 3-month change; mean squared error of next month's log vol before and in the test."""
    df = sdf[["lv5", "lv22", "lv63", "lvix", "lvol_next"]].copy()
    df["spread"] = spread_m.reindex(df.index)
    df["dspread"] = df["spread"] - df["spread"].shift(3)
    y = df["lvol_next"].to_numpy()
    out = {}
    for name, cols in (("har", ["lv5", "lv22", "lv63", "lvix"]), ("har_curve", ["lv5", "lv22", "lv63", "lvix", "spread", "dspread"])):
        X = df[cols].to_numpy()
        f = np.full(len(df), np.nan)
        for i in range(60, len(df)):
            ok = ~np.isnan(X[:i - 1]).any(1) & ~np.isnan(y[:i - 1])
            if ok.sum() > 48 and not np.isnan(X[i]).any():
                b = np.linalg.lstsq(np.c_[np.ones(ok.sum()), X[:i - 1][ok]], y[:i - 1][ok], rcond=None)[0]
                f[i] = np.r_[1.0, X[i]] @ b
        df[f"f_{name}"] = f
    cut = pd.Period(pd.Timestamp(train_end), "M")
    both = df[["f_har", "f_har_curve", "lvol_next"]].dropna()
    for per, sel in (("before_test", (both.index <= cut) & (both.index.year >= collapse.FIRST_YEAR)), ("test", both.index > cut)):
        b = both[sel]
        e1, e2 = (b["lvol_next"] - b["f_har"]) ** 2, (b["lvol_next"] - b["f_har_curve"]) ** 2
        d = e1 - e2
        out[per] = {"n": len(b), "mse_har": float(e1.mean()), "mse_har_curve": float(e2.mean()),
                    "gain_pct": float(100 * (1 - e2.mean() / e1.mean())), "dm_t": float(d.mean() / d.std() * math.sqrt(len(d))) if d.std() else None}
    return out


def curve_sleeves(T: dict, spread: pd.Series) -> dict[str, pd.Series]:
    """Variants of two existing sleeves that stand aside while the curve is inverted (10y below 3m at the month end for
    the swap; at yesterday's close for the daily SPY filter)."""
    sp_m = spread.groupby(spread.index.to_period("M")).last()
    vc = collapse.vol_carry(T["spy"], T["vix"], T["p_hi"], T["sdf"])
    idx = vc["varswap_always"].index
    held = T["on"].reindex(idx, fill_value=False).astype(bool) & (sp_m.reindex(idx) > 0)
    out = {"varswap_timedcurve": _next(vc["varswap_always"].where(held, 0.0))}
    spy, rf = T["spy"], T["rf"]
    r = spy.pct_change().dropna()
    ex = r - rf.reindex(r.index).fillna(0)
    calm = (T["p_hi"].reindex(r.index).shift(1).fillna(0) < 0.5)
    up = (spread.reindex(r.index).ffill().shift(1) > 0)
    for k, held in (("regime_spycurve", (calm & up).astype(float)), ("regime_spycurveonly", up.astype(float))):
        trade = held.diff().abs().fillna(held.iloc[0])
        out[k] = collapse.month_ret(held * ex - trade * papers.COST_BP / 1e4)
    return out


def inversion_episodes(spread: pd.Series, spy: pd.Series) -> list[dict]:
    """Each spell of inversion (month-end spread below zero), with SPY's return over the following 12 months."""
    sp_m = spread.groupby(spread.index.to_period("M")).last()
    px_m = spy.groupby(spy.index.to_period("M")).last()
    inv = sp_m < 0
    eps, start = [], None
    for m, v in inv.items():
        if v and start is None:
            start = m
        if not v and start is not None:
            eps.append((start, m - 1))
            start = None
    if start is not None:
        eps.append((start, inv.index[-1]))
    out = []
    for a, b in eps:
        n = (b - a).n + 1
        if n < 2:
            continue
        def ret(m0, k):
            m1 = m0 + k
            return None if m0 not in px_m.index or m1 not in px_m.index else round(float(100 * (px_m[m1] / px_m[m0] - 1)), 1)
        out.append({"from": str(a), "to": str(b), "months": n, "spy_next_12m_from_start_pct": ret(a, 12), "spy_next_12m_from_end_pct": ret(b, 12)})
    return out


# ---------------------------------------------------------------- 3. options snapshot

def chain_summary(ch: pd.DataFrame) -> dict:
    """One row per underlying from today's chain: 30- and 90-day at-the-money implied vol, 25-delta put and call vol at
    30 days (skew = put - call), put/call open interest and volume."""
    ch = ch.dropna(subset=["implied_volatility", "dte", "strike"])
    ch = ch[(ch["implied_volatility"] > 0) & (ch["implied_volatility"] < 5) & (ch["dte"] >= 5)]
    if ch.empty:
        return {}
    spot = float(ch["underlying_price"].dropna().iloc[0])
    row = {"spot": spot}

    def atm(dte_target):
        iv = {}
        for d, g in ch.groupby("dte"):
            g = g.assign(dist=(g["strike"] - spot).abs())
            near = g[g["dist"] == g["dist"].min()]
            iv[d] = float(near["implied_volatility"].mean())
        s = pd.Series(iv).sort_index()
        if s.empty or s.index.max() < dte_target * 0.5:
            return None
        return float(np.interp(dte_target, s.index.to_numpy(), s.to_numpy()))

    row["iv30_atm"], row["iv90_atm"] = atm(30), atm(90)
    dtes = np.array(sorted(ch["dte"].unique()))
    d30 = int(dtes[np.argmin(np.abs(dtes - 30))])
    g = ch[ch["dte"] == d30].dropna(subset=["delta"])
    for side, target, key in (("put", -0.25, "iv30_put25"), ("call", 0.25, "iv30_call25")):
        h = g[g["option_type"].str.lower() == side]
        row[key] = None if h.empty else float(h.loc[(h["delta"] - target).abs().idxmin(), "implied_volatility"])
    row["skew30"] = None if row["iv30_put25"] is None or row["iv30_call25"] is None else row["iv30_put25"] - row["iv30_call25"]
    typ = ch["option_type"].str.lower()
    for k, col in (("pc_oi", "open_interest"), ("pc_volume", "volume")):
        c, p = ch.loc[typ == "call", col].sum(), ch.loc[typ == "put", col].sum()
        row[k] = float(p / c) if c else None
    row["dte_30_used"] = d30
    return row


def options_snapshot(tickers: list[str], out: Path, limit_s: float = 2400.0, pause: float = 0.2) -> dict:
    """Append today's summary for each ticker to out (CSV). Returns counts."""
    from openbb import obb
    t0, rows, failed = time.time(), [], {}
    day = None
    for t in tickers:
        if time.time() - t0 > limit_s:
            failed[t] = "time limit"
            continue
        try:
            ch = obb.derivatives.options.chains(papers.ysym(t), provider="cboe").to_df()
            s = chain_summary(ch)
            if s:
                if day is None and "eod_date" in ch and ch["eod_date"].notna().any():
                    day = str(pd.to_datetime(ch["eod_date"].dropna().iloc[0]).date())
                rows.append({"ticker": t, **s})
        except Exception as e:  # noqa: BLE001 - names without listed options are expected
            failed[t] = f"{type(e).__name__}: {str(e)[:80]}"
        time.sleep(pause)
    day = day or datetime.now(UTC).strftime("%Y-%m-%d")
    new = pd.DataFrame(rows)
    if not new.empty:
        new.insert(0, "date", day)
        old = pd.read_csv(out) if out.exists() else pd.DataFrame()
        allr = pd.concat([old, new], ignore_index=True).drop_duplicates(["date", "ticker"], keep="last")
        allr.to_csv(out, index=False, float_format="%.5g")
    return {"date": day, "rows": len(rows), "failed": len(failed), "failed_sample": dict(list(failed.items())[:10]),
            "seconds": round(time.time() - t0, 1)}


# ---------------------------------------------------------------- run

def run(data: Path, out: Path, ref_sleeves: Path, fetch: bool = True, train_end: str = TRAIN_END) -> dict:
    t0 = time.time()
    out.mkdir(parents=True, exist_ok=True)
    rep = {"built": datetime.now(UTC).strftime("%Y-%m-%d %H:%M UTC"), "train_end": train_end, "warnings": []}
    if fetch:
        rep["fetch"] = fetch_extra(data)
    T = timing(data, train_end)
    ref = pd.read_csv(ref_sleeves, index_col=0)
    ref.index = pd.PeriodIndex(ref.index, freq="M")
    rep["reference_portfolio"] = portfolio_with(ref, None)

    # 1. vol products
    sl, rep["vol_products_notes"] = vol_products(data, T)
    rep["vol_products"] = {k: period_stats(v) for k, v in sl.items()}
    D = pd.DataFrame(sl)
    rep["vol_products_corr_with_varswap_timed"] = D.corr()["varswap_timed"].round(2).to_dict()
    rep["timing_on_share"] = float(T["on"].mean())
    rep["portfolio_with"] = {}
    for name in ("svxy", "shortvixy", "vxfut"):
        if f"{name}_timed" in sl:
            rep["portfolio_with"][name] = portfolio_with(ref, {"always": sl[f"{name}_always"], "timed": sl[f"{name}_timed"]})
    if "svxy_timed" in sl:
        post = {k: v[v.index >= SVXY_CHANGE] for k, v in sl.items() if k.startswith(("svxy", "varswap"))}
        rep["svxy_since_half_leverage"] = {k: sstats(v) for k, v in post.items()}

    # 2. yield curve
    spread = curve_spread(data)
    if spread is None:
        rep["warnings"].append("No Treasury curve: the yield-curve tests were skipped.")
    else:
        sp_m = spread.groupby(spread.index.to_period("M")).last()
        rep["curve"] = {"last": {"date": str(spread.index[-1].date()), "spread_pts": round(float(spread.iloc[-1]), 2)},
                        "inverted_share_since_2005": float((sp_m[sp_m.index.year >= 2005] < 0).mean()),
                        "episodes": inversion_episodes(spread, T["spy"]),
                        "vol_forecast": vol_forecast_with_curve(T["sdf"], sp_m, train_end)}
        cs = curve_sleeves(T, spread)
        rep["curve"]["sleeves"] = {k: period_stats(v) for k, v in cs.items()}
        for k in ("varswap_timed", "regime_spy"):
            if k in ref:
                rep["curve"]["sleeves"][k + " (reference)"] = period_stats(ref[k])
        rep["curve"]["portfolio_with_curve_variants"] = portfolio_with(ref, None, cs)
        pd.DataFrame(cs).to_csv(out / "curve_sleeves_monthly.csv", float_format="%.4f")
    D.to_csv(out / "vol_products_monthly.csv", float_format="%.4f")
    rep["seconds"] = round(time.time() - t0, 1)
    (out / "summary.json").write_text(json.dumps(rep, indent=1, default=str))
    return rep
