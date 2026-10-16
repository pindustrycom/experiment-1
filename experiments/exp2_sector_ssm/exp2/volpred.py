"""Volatility first, then returns: a two-stage test on the S&P 500 price cache of exp2/papers.py.

Stage 1 forecasts each stock's volatility over the next month (pooled HAR in logs and gradient boosting, refit
every January on the past only). Stage 2 uses those out-of-sample forecasts as inputs to:

  A. the GKX-style return ranking (boosted trees), with and without the volatility forecast features;
  B. the same ranking trained on risk-adjusted returns (next return / forecast volatility), and legs weighted by
     inverse forecast volatility;
  C. the low-volatility anomaly: sort stocks on forecast volatility;
  D. volatility-scaled momentum (Barroso & Santa-Clara), stock by stock and for the momentum book as a whole;
  E. SPY: next month's return and a volatility-targeted SPY using the forecast.

Every comparison uses the same months, the same costs (10 bp per unit traded) and the same refit schedule.
Paper research only: nothing is traded.
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

from . import papers
from .papers import COST_BP, FEATURES, TRAIN_END, deciles, rank_features, sstats

log = logging.getLogger("exp2.volpred")
VOL_X = ["lv5", "lv22", "lv63", "lv252", "lv_on", "lv_mkt", "lvix", "lv_sector", "mom1", "mom12_2", "beta12m", "ldvol", "lprice", "hi52", "max1m"]
HAR_X = ["lv5", "lv22", "lv63", "lv252", "lvix"]
FVOL_X = ["fvol", "fvol_chg", "vol_surprise", "fvol_rel"]


def vol_features(px: dict[str, pd.DataFrame], spy: pd.Series, vix: pd.Series) -> pd.DataFrame:
    """Per (month, ticker): log daily volatility over the last 5/22/63/252 sessions, overnight volatility, the market's,
    and the target: log volatility of daily returns over the next month."""
    mr = np.log(spy).diff()
    g_m = mr.index.to_period("M")
    lv_mkt = np.log(mr.rolling(22).std()).groupby(g_m).last()
    lvix = np.log(vix / 100 / math.sqrt(252)).groupby(vix.index.to_period("M")).last()     # VIX as a daily vol
    rows = []
    for t, d in px.items():
        c = d["adj_close"].astype(float)
        r = np.log(c).diff()
        o = d["open"].astype(float) * d["adj_close"] / d["close"]
        on = np.log(o / c.shift(1)).where(lambda s: s.abs() < 0.5)
        g = r.index.to_period("M")
        f = pd.DataFrame(index=r.groupby(g).size().index)
        for k in (5, 22, 63, 252):
            f[f"lv{k}"] = np.log(r.rolling(k, min_periods=int(0.8 * k)).std().clip(lower=1e-4)).groupby(g).last()
        f["lv_on"] = np.log(on.rolling(63, min_periods=40).std().clip(lower=1e-4)).groupby(g).last()
        sd = r.groupby(g).std()
        n = r.groupby(g).count()
        f["lvol_next"] = np.log(sd.clip(lower=1e-4)).where(n >= 15).shift(-1)
        f["lv_mkt"] = lv_mkt.reindex(f.index)
        f["lvix"] = lvix.reindex(f.index)
        f["ticker"] = t
        rows.append(f)
    return pd.concat(rows).reset_index(names="month")


def _yearly(P: pd.DataFrame, first_year: int, fit_predict) -> np.ndarray:
    """Refit every January on all earlier months whose target is known; predict that year's months."""
    yrs = P["month"].dt.year.to_numpy()
    dec = P["month"].dt.month.to_numpy() == 12
    out = np.full(len(P), np.nan)
    for yr in range(first_year, int(yrs.max()) + 1):
        tr = (yrs < yr) & ~((yrs == yr - 1) & dec)            # December's target is January's
        te = yrs == yr
        if te.any() and tr.sum() > 3000:
            out[te] = fit_predict(tr, te)
    return out


def _gbt(**kw):
    from sklearn.ensemble import HistGradientBoostingRegressor
    p = {"max_depth": 3, "learning_rate": 0.05, "max_iter": 300, "min_samples_leaf": 500, "l2_regularization": 1.0, "random_state": 0}
    p.update(kw)
    return HistGradientBoostingRegressor(**p)


# ---------------------------------------------------------------- stage 1

def stage1(P: pd.DataFrame, first_year: int = 2003, train_end: str = TRAIN_END) -> tuple[dict, pd.DataFrame]:
    P = P.copy()
    P["lv_sector"] = P.groupby(["month", "sector"])["lv22"].transform("mean")
    y = P["lvol_next"].to_numpy()
    X = P[VOL_X].to_numpy(float)
    H = P[HAR_X].to_numpy(float)

    def har(tr, te):
        ok = tr & ~np.isnan(y) & ~np.isnan(H).any(1)
        b = np.linalg.lstsq(np.c_[np.ones(ok.sum()), H[ok]], y[ok], rcond=None)[0]
        h = np.where(np.isnan(H[te]), np.nanmean(H[ok], 0), H[te])
        return np.c_[np.ones(te.sum()), h] @ b

    def gbt(tr, te):
        ok = tr & ~np.isnan(y)
        return _gbt().fit(X[ok], y[ok]).predict(X[te])

    P["f_last"] = P["lv22"]
    P["f_har"] = _yearly(P, first_year, har)
    P["f_gbt"] = _yearly(P, first_year, gbt)
    log.info("stage 1 fitted")
    res = {}
    test = P["month_end"] > pd.Timestamp(train_end)
    early = ~test & (P["month"].dt.year >= first_year)
    for per, sel in (("before_test", early), ("test", test)):
        s = sel & P["f_gbt"].notna() & P["f_har"].notna() & P["lvol_next"].notna() & P["f_last"].notna()
        row = {}
        for k in ("f_last", "f_har", "f_gbt"):
            e = P.loc[s, "lvol_next"] - P.loc[s, k]
            # variance QLIKE with each forecaster's own mean bias removed (a constant learned from that period)
            v_true, v_hat = np.exp(2 * P.loc[s, "lvol_next"]), np.exp(2 * (P.loc[s, k] + e.mean()))
            ql = (v_true / v_hat - np.log(v_true / v_hat) - 1).mean()
            ic = P[s].groupby("month").apply(lambda g, k=k: g["lvol_next"].rank().corr(g[k].rank())).mean()
            row[k] = {"mse_log": float((e ** 2).mean()), "r2_log": float(1 - (e ** 2).sum() / ((P.loc[s, "lvol_next"] - P.loc[s, "lvol_next"].mean()) ** 2).sum()),
                      "qlike": float(ql), "rank_ic": float(ic), "n": int(s.sum())}
        d = (P.loc[s, "lvol_next"] - P.loc[s, "f_last"]) ** 2 - (P.loc[s, "lvol_next"] - P.loc[s, "f_gbt"]) ** 2
        dm = d.groupby(P.loc[s, "month"]).mean()                       # monthly loss differential (cross-sectional mean)
        row["gbt_vs_last_t"] = float(dm.mean() / dm.std() * math.sqrt(len(dm)))
        row["skill_vs_last_pct"] = float(100 * (1 - row["f_gbt"]["mse_log"] / row["f_last"]["mse_log"]))
        res[per] = row
    return res, P


# ---------------------------------------------------------------- stage 2

def stage2_rank(P: pd.DataFrame, first_year: int = 2008, train_end: str = TRAIN_END) -> tuple[dict, pd.DataFrame]:
    """A and B: return ranking with and without the volatility forecast."""
    P = P.copy()
    P["fvol"] = P["f_gbt"]
    P["fvol_chg"] = P["fvol"] - P["lv22"]                                  # where vol is expected to go
    prev = P.sort_values("month").groupby("ticker")["fvol"].shift(1)
    P["vol_surprise"] = P["lv22"] - prev.reindex(P.index)                   # realised this month minus last month's forecast
    P["fvol_rel"] = P["fvol"] - P.groupby(["month", "sector"])["fvol"].transform("mean")
    P = P[P["fvol"].notna()].reset_index(drop=True)
    Z = rank_features(P, FEATURES + FVOL_X)
    yd = P.groupby("month")["y"].transform(lambda s: s - s.mean()).to_numpy()
    sig = np.exp(P["fvol"].to_numpy()) * math.sqrt(21)                      # forecast monthly vol
    yr_adj = P.groupby("month")["y"].transform(lambda s: s - s.mean()).to_numpy() / sig
    yr_adj = np.clip(yr_adj, -5, 5)
    specs = {"base": (FEATURES, yd), "plus_vol": (FEATURES + FVOL_X, yd), "risk_adj_target": (FEATURES + FVOL_X, yr_adj),
             "vol_only": (FVOL_X, yd)}
    for name, (cols, target) in specs.items():
        X = Z[cols].to_numpy(np.float32)

        def fp(tr, te, X=X, target=target):
            ok = tr & ~np.isnan(target)
            return _gbt().fit(X[ok], target[ok]).predict(X[te])

        P[f"p_{name}"] = _yearly(P, first_year, fp)
        log.info("stage 2 %s fitted", name)
    P["inv_vol"] = 1 / sig
    test = P["month_end"] > pd.Timestamp(train_end)
    early = ~test & (P["month"].dt.year >= first_year)
    res, books = {}, {}
    for per, sel in (("before_test", early), ("test", test)):
        row = {}
        for name in specs:
            s = sel & P[f"p_{name}"].notna()
            ew = deciles(P[s], f"p_{name}")
            iv = deciles(P[s], f"p_{name}", weight="inv_vol")
            row[name] = {"ls_equal": sstats(ew), "ls_inverse_vol": sstats(iv)}
            books[(per, name, "ew")], books[(per, name, "iv")] = ew, iv
        for name in ("plus_vol", "risk_adj_target"):
            d = (books[(per, name, "ew")] - books[(per, "base", "ew")]).dropna()
            row[f"{name}_minus_base"] = sstats(d)
        d = (books[(per, "base", "iv")] - books[(per, "base", "ew")]).dropna()
        row["base_inverse_vol_minus_equal"] = sstats(d)
        res[per] = row
    return res, P


def low_vol(P: pd.DataFrame, train_end: str = TRAIN_END) -> dict:
    """C: decile portfolios on forecast volatility; returns, volatility and Sharpe of each, and low minus high."""
    out = {}
    for per, sel in (("before_test", P["month_end"] <= pd.Timestamp(train_end)), ("test", P["month_end"] > pd.Timestamp(train_end))):
        g = P[sel & P["f_gbt"].notna() & P["y"].notna()].copy()
        g["dec"] = g.groupby("month")["f_gbt"].transform(lambda s: np.ceil(10 * s.rank(pct=True)).clip(1, 10))
        m = g.groupby(["month", "dec"])["y"].mean().unstack() * 100
        rows = {}
        for k in (1, 5, 10):
            s = m[k].dropna()
            rows[f"decile_{k}"] = {"mean_pct_month": float(s.mean()), "vol_pct_month": float(s.std()), "sharpe": float(s.mean() / s.std() * math.sqrt(12))}
        ls = (m[1] - m[10]).dropna()
        out[per] = {**rows, "low_minus_high": sstats(ls), "low_minus_high_net": sstats(deciles(g.assign(neg=-g["f_gbt"]), "neg"))}
    return out


def vol_momentum(P: pd.DataFrame, train_end: str = TRAIN_END) -> dict:
    """D: momentum per unit of forecast volatility (stock level), and the momentum book scaled by its own forecast risk."""
    P = P.copy()
    P["mom_riskadj"] = P["mom12_2"] / np.exp(P["f_gbt"])
    out = {}
    for per, sel in (("before_test", P["month_end"] <= pd.Timestamp(train_end)), ("test", P["month_end"] > pd.Timestamp(train_end))):
        g = P[sel & P["f_gbt"].notna()]
        mom = deciles(g, "mom12_2")
        out[per] = {"mom12_2": sstats(mom), "mom_per_unit_vol": sstats(deciles(g, "mom_riskadj"))}
    # book-level scaling (Barroso & Santa-Clara): weight = target / last 6 months' realised vol of the momentum book
    full = deciles(P[P["f_gbt"].notna()], "mom12_2")
    vol6 = full.rolling(6).std().shift(1)
    w = (full.std() / vol6).clip(upper=2.0)
    scaled = (w * full - COST_BP / 100 * w.diff().abs()).dropna()
    for per, sel in (("before_test", scaled.index.to_timestamp("M") <= pd.Timestamp(train_end)), ("test", scaled.index.to_timestamp("M") > pd.Timestamp(train_end))):
        out[per]["mom12_2_vol_scaled_book"] = sstats(scaled[sel])
        out[per]["mom12_2_same_months"] = sstats(full.reindex(scaled.index)[sel])
    return out


def spy_timing(data: Path, train_end: str = TRAIN_END) -> tuple[dict, pd.DataFrame]:
    """E: SPY. Forecast next month's log vol (HAR in logs + VIX, refit monthly on the past); does it predict next month's
    excess return, and does a volatility-targeted SPY beat buy and hold?"""
    px = papers.load(data, ["SPY", "^VIX", "^IRX"])
    r = np.log(px["SPY"]["adj_close"]).diff().dropna()
    rf = (px["^IRX"]["close"].reindex(r.index).ffill() / 100 / 252).fillna(0.0)
    vix = px["^VIX"]["close"].reindex(r.index).ffill()
    g = r.index.to_period("M")
    df = pd.DataFrame({"ex": (r - rf).groupby(g).sum() * 100,
                       "lv5": np.log(r.rolling(5).std()).groupby(g).last(), "lv22": np.log(r.rolling(22).std()).groupby(g).last(),
                       "lv63": np.log(r.rolling(63).std()).groupby(g).last(), "lvix": np.log(vix / 100 / math.sqrt(252)).groupby(g).last()})
    df["lvol_next"] = np.log(r.groupby(g).std()).shift(-1)
    df["ex_next"] = df["ex"].shift(-1)
    X = df[["lv5", "lv22", "lv63", "lvix"]].to_numpy()
    f = np.full(len(df), np.nan)
    for i in range(60, len(df)):
        ok = ~np.isnan(X[:i - 1]).any(1) & ~np.isnan(df["lvol_next"].to_numpy()[:i - 1])
        if ok.sum() > 48 and not np.isnan(X[i]).any():
            b = np.linalg.lstsq(np.c_[np.ones(ok.sum()), X[:i - 1][ok]], df["lvol_next"].to_numpy()[:i - 1][ok], rcond=None)[0]
            f[i] = np.r_[1.0, X[i]] @ b
    df["fvol"] = f
    # 1. does the forecast predict next month's excess return? expanding regression, out of sample R^2 vs the mean
    y = df["ex_next"].to_numpy()
    pred, bench = np.full(len(df), np.nan), np.full(len(df), np.nan)
    for i in range(96, len(df)):
        ok = ~np.isnan(f[:i - 1]) & ~np.isnan(y[:i - 1])
        if ok.sum() > 36 and not np.isnan(f[i]):
            b = np.linalg.lstsq(np.c_[np.ones(ok.sum()), f[:i - 1][ok]], y[:i - 1][ok], rcond=None)[0]
            pred[i], bench[i] = b[0] + b[1] * f[i], y[:i - 1][ok].mean()
    df["pred_ex"] = pred
    # 2. vol-targeted SPY: weight = target vol / forecast vol, capped at 1.5; target = long-run median of the forecast
    sig = np.exp(df["fvol"]) * math.sqrt(21) * 100
    w = (np.nanmedian(sig) / sig).clip(upper=1.5)
    df["w_target"] = w
    df["book_target"] = w * df["ex_next"] - COST_BP / 100 * w.diff().abs()
    lag_sig = np.exp(df["lv22"]) * math.sqrt(21) * 100                     # same rule with last month's vol instead of the forecast
    w2 = (np.nanmedian(lag_sig) / lag_sig).clip(upper=1.5)
    df["book_lastvol"] = w2 * df["ex_next"] - COST_BP / 100 * w2.diff().abs()
    out = {}
    for per, sel in (("before_test", (df.index.to_timestamp("M") <= pd.Timestamp(train_end)) & (df.index.year >= 2004)),
                     ("test", df.index.to_timestamp("M") > pd.Timestamp(train_end))):
        s = sel & ~np.isnan(pred) & ~np.isnan(y)
        row = {"r2_oos_return_from_vol": float(1 - np.sum((y[s] - pred[s]) ** 2) / np.sum((y[s] - bench[s]) ** 2)),
               "vol_forecast_mse_log": float(np.nanmean((df["lvol_next"].to_numpy()[s] - f[s]) ** 2)),
               "lastvol_mse_log": float(np.nanmean((df["lvol_next"].to_numpy()[s] - df["lv22"].to_numpy()[s]) ** 2)), "n": int(s.sum())}
        for k, col in (("buy_hold", "ex_next"), ("vol_target_forecast", "book_target"), ("vol_target_lastvol", "book_lastvol")):
            v = df.loc[sel & df["fvol"].notna(), col].dropna()
            mdd = (v.cumsum() - v.cumsum().cummax()).min()
            row[k] = {"sharpe": float(v.mean() / v.std() * math.sqrt(12)), "mean_pct_month": float(v.mean()), "max_drawdown_pct_log": float(mdd)}
        out[per] = row
    return out, df


def run(data: Path, out: Path, train_end: str = TRAIN_END, first_years: tuple[int, int] = (2003, 2008)) -> dict:
    t0 = time.time()
    out.mkdir(parents=True, exist_ok=True)
    rep = {"built": datetime.now(UTC).strftime("%Y-%m-%d %H:%M UTC"), "train_end": train_end, "warnings": []}
    cur = pd.read_csv(data / "sp500_current.csv")
    changes = pd.read_csv(data / "sp500_changes.csv", parse_dates=["date"])
    if changes.empty or changes["removed"].isna().all():
        rep["warnings"].append("Membership from current members only (dates added): companies that later left the index are missing.")
    member = papers.membership(cur, changes, pd.date_range(papers.START, pd.Timestamp.today(), freq="ME"))
    tickers = sorted(member.columns)
    px = papers.load(data, tickers)
    base = papers.load(data, ["SPY", "^VIX"])
    spy, vix = base["SPY"]["adj_close"], base["^VIX"]["close"]
    P = papers.panel(px, spy, dict(zip(cur["ticker"], cur["sector"])), member)
    V = vol_features(px, spy, vix)
    P = P.merge(V, on=["month", "ticker"], how="left")
    log.info("panel %d rows (%.0fs)", len(P), time.time() - t0)
    rep["stage1_vol_forecast"], P = stage1(P, first_years[0], train_end)
    rep["low_vol"] = low_vol(P, train_end)
    rep["vol_momentum"] = vol_momentum(P, train_end)
    rep["stage2_ranking"], _ = stage2_rank(P, first_years[1], train_end)
    rep["spy"], sdf = spy_timing(data, train_end)
    P[["month", "ticker", "lvol_next", "f_last", "f_har", "f_gbt"]].to_csv(out / "vol_forecasts.csv", index=False, float_format="%.4g")
    sdf.to_csv(out / "spy_monthly.csv", float_format="%.5g")
    rep["seconds"] = round(time.time() - t0, 1)
    (out / "summary.json").write_text(json.dumps(rep, indent=1, default=str))
    return rep
