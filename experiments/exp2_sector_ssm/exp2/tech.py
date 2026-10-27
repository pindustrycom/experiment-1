"""The volatility-timed portfolio of exp2/collapse.py, rebuilt on the tech sector.

Joaquin asked whether the Exp2 portfolio (built on SPY) can be extended to other tickers, especially tech. Each
sleeve gets a tech counterpart:

  variance swap   a synthetic short one-month QQQ variance swap struck at VXN^2 (Nasdaq-100 volatility index), timed
                  like the SPY one: only when VXN^2 is above our forecast of next month's QQQ variance and a 2-state
                  regime model on QQQ says calm
  regime          QQQ (or XLK) held only while the regime model says calm, else T-bills
  stock book      long-only books of the S&P 500 technology members of each month (inverse volatility, minimum variance, HRP,
                  each also without the 10% most collapse-prone), and a wider "tech and internet" universe that adds
                  communication services and the large internet retailers
  trend           QQQ and XLK trend filters (hold when the past 12 months beat T-bills)

The combination is the same as collapse.combine: the best variant of each family is picked on 2005-2018 if its
Sharpe is above 0.3, weights and leverage are set on 2005-2018, and 2019 on is the untouched test. Costs as in
collapse.py. Paper research only.
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

from . import collapse, papers, pit
from .papers import COST_BP, TRAIN_END

log = logging.getLogger("exp2.tech")
EXTRA = ["QQQ", "XLK", "^VXN"]
INTERNET = ["AMZN", "EBAY", "BKNG", "NFLX", "GOOGL", "GOOG", "META", "TTWO", "EA"]


def vol_timing(under: pd.Series, vol_index: pd.Series) -> pd.DataFrame:
    """Monthly forecast of next month's log volatility of `under` (HAR in logs plus the volatility index), refit
    each month on the past only; the same model as volpred.spy_timing."""
    r = np.log(under).diff().dropna()
    vi = vol_index.reindex(r.index).ffill()
    g = r.index.to_period("M")
    df = pd.DataFrame({"lv5": np.log(r.rolling(5).std()).groupby(g).last(), "lv22": np.log(r.rolling(22).std()).groupby(g).last(),
                       "lv63": np.log(r.rolling(63).std()).groupby(g).last(), "lvix": np.log(vi / 100 / math.sqrt(252)).groupby(g).last()})
    y = np.log(r.groupby(g).std()).shift(-1).to_numpy()
    X = df.to_numpy()
    f = np.full(len(df), np.nan)
    for i in range(60, len(df)):
        ok = ~np.isnan(X[:i - 1]).any(1) & ~np.isnan(y[:i - 1])
        if ok.sum() > 48 and not np.isnan(X[i]).any():
            b = np.linalg.lstsq(np.c_[np.ones(ok.sum()), X[:i - 1][ok]], y[:i - 1][ok], rcond=None)[0]
            f[i] = np.r_[1.0, X[i]] @ b
    df["fvol"] = f
    return df


def regime(under: pd.Series, train_end: str) -> pd.Series:
    r = np.log(under).diff()
    fit = (r.index <= pd.Timestamp(train_end)) & r.notna().to_numpy()
    p, _ = collapse.hmm2(r.fillna(0).to_numpy(), fit)
    return pd.Series(p, index=under.index)


def held_when_calm(under: pd.Series, rf: pd.Series, p_hi: pd.Series) -> pd.Series:
    """Monthly % excess return of holding `under` only while yesterday's regime says calm."""
    r = under.pct_change().dropna()
    ex = r - rf.reindex(r.index).fillna(0)
    h = (p_hi.reindex(r.index).shift(1).fillna(0) < 0.5).astype(float)
    trade = h.diff().abs().fillna(h.iloc[0])
    return collapse.month_ret(h * ex - trade * COST_BP / 1e4)


def trend_filter(under: pd.Series, rf: pd.Series) -> pd.Series:
    r = under.pct_change().dropna()
    ex = r - rf.reindex(r.index).fillna(0)
    Mx = ex.groupby(ex.index.to_period("M")).apply(lambda d: (1 + d).prod() - 1) * 100
    on = (Mx.rolling(12).sum() > 0).astype(float)
    return (on.shift(1) * Mx - on.diff().abs().shift(1) * COST_BP / 100).iloc[13:]


def run(data: Path, out: Path, train_end: str = TRAIN_END, fetch: bool = True) -> dict:
    t0 = time.time()
    out.mkdir(parents=True, exist_ok=True)
    rep = {"built": datetime.now(UTC).strftime("%Y-%m-%d %H:%M UTC"), "train_end": train_end,
           "warnings": [] if pit.former(data) else ["Stock books use today's S&P 500 members only, which flatters long-only results."]}
    if fetch:
        rep["fetch"] = papers.update(data, EXTRA, pause=0.3)
    P, C, member, spy, vix, rf, sectors = collapse.load_all(data, out, train_end)
    ext = papers.load(data, EXTRA)
    missing = [t for t in EXTRA if t not in ext]
    if missing:
        raise RuntimeError(f"no prices for {missing}")
    qqq, xlk = ext["QQQ"]["adj_close"], ext["XLK"]["adj_close"]
    vxn = ext["^VXN"]["close"]
    rep["history_from"] = {t: str(d.index[0].date()) for t, d in ext.items()}
    # collapse probabilities need the SPY regime, as in collapse.run
    ph_spy = regime(spy, train_end)
    P["p_hi"] = P["month"].map(ph_spy.groupby(ph_spy.index.to_period("M")).last())
    P = collapse.collapse_labels(P, C, spy)
    _, P = collapse.collapse_model(P, train_end)
    rf_m = collapse.month_ret(rf)
    p_q, p_x = regime(qqq, train_end), regime(xlk, train_end)
    sleeves: dict[str, pd.Series] = {}
    vc = collapse.vol_carry(qqq, vxn, p_q, vol_timing(qqq, vxn))
    sleeves.update({k.replace("varswap", "varswap_qqq"): collapse._shift(v) for k, v in vc.items()})
    sleeves["regime_qqq"] = held_when_calm(qqq, rf, p_q)
    sleeves["regime_xlk"] = held_when_calm(xlk, rf, p_x)
    sleeves["trend_qqq_filter"] = trend_filter(qqq, rf)
    sleeves["trend_xlk_filter"] = trend_filter(xlk, rf)
    universes = {"it": {t for t, s in sectors.items() if s == "Information Technology"}}
    universes["tech_internet"] = universes["it"] | {t for t, s in sectors.items() if s == "Communication Services"} | set(INTERNET)
    rep["universe_size_today"] = {k: len(v) for k, v in universes.items()}
    for name, u in universes.items():
        books = collapse.construction(P[P["ticker"].isin(u)], C, rf_m, min_names=20)
        sleeves.update({f"port_{name}_{k[5:]}": collapse._shift(v) for k, v in books.items()})
    log.info("sleeves (%.0fs)", time.time() - t0)
    # benchmark: QQQ buy and hold, in excess of T-bills (combine reads it as "spy")
    r = qqq.pct_change().dropna()
    sleeves["spy"] = collapse.month_ret(r - rf.reindex(r.index).fillna(0))
    rep["combined"], books = collapse.combine(sleeves, train_end)
    rep["combined"]["qqq_test"] = rep["combined"].pop("spy_test")
    # the SPY portfolio of collapse.py over the same months, for comparison
    sp = pd.read_csv(out.parent / "collapse" / "portfolio_test_monthly.csv", index_col=0) if (out.parent / "collapse" / "portfolio_test_monthly.csv").exists() else None
    if sp is not None and "min_variance" in sp:
        books = books.join(sp["min_variance"].rename("spy_min_variance").set_axis(pd.PeriodIndex(sp.index, freq="M")), how="left")
        both = books[["min_variance", "spy_min_variance"]].dropna()
        rep["corr_with_spy_portfolio"] = float(both.corr().iloc[0, 1])
        half = (both.mean(axis=1))
        rep["half_and_half"] = papers.sstats(half)
    pd.DataFrame(sleeves).sort_index().to_csv(out / "sleeves_monthly.csv", float_format="%.4f")
    books.to_csv(out / "portfolio_test_monthly.csv", float_format="%.4f")
    (out / "panel_stage1.pkl").unlink(missing_ok=True)
    rep["seconds"] = round(time.time() - t0, 1)
    (out / "summary.json").write_text(json.dumps(rep, indent=1, default=str))
    return rep
