"""Three stock-prediction ideas from the literature, tested with free data (paper research only).

    python -m exp2 papers --data papers_data --out papers_results

1. Variance risk premium (Bollerslev, Tauchen & Zhou 2009): VRP = VIX^2 (risk-neutral expected variance) minus the
   expected realised variance of the S&P 500. A high premium is said to predict higher returns over the next 1-6
   months. Expected variance comes from three forecasters: last month's realised variance (the paper's choice), HAR,
   and a one-state stochastic-volatility Kalman filter (the same model as the EKF volatility state in Experiment 3).
   Forecasts are made in real time: each month's regression uses only returns that were complete by then.
2. Machine-learning ranking of S&P 500 stocks (Gu, Kelly & Xiu 2020, "GKX-lite"): monthly features from prices and
   volume only (momentum, short-term reversal, industry momentum, volatility, idiosyncratic volatility, beta, max
   daily return, 52-week-high ratio, dollar volume, price), ranked within each month; next month's return predicted
   by OLS on 3 features, ridge, gradient-boosted trees and a neural net, refitted every January on all earlier data.
   Stocks enter the universe only from the month they joined the S&P 500 (membership rebuilt from Wikipedia's list of
   changes); companies that left and are no longer on Yahoo are missing, so some survivorship bias remains.
3. Overnight versus intraday momentum (Lou, Polk & Skouras 2019): past-year momentum built from close-to-open returns
   and from open-to-close returns separately, and where next month's long-short return arrives (overnight or intraday).
Scores: out-of-sample R^2, decile long-short returns with costs, Sharpe, by year; train up to 2018, test from 2019.
"""
from __future__ import annotations

import io
import json
import logging
import math
import time
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import pandas as pd
import requests

from . import openbb_src

log = logging.getLogger("exp2.papers")
TRAIN_END = "2018-12-31"
START = "1998-01-01"
WIKI = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
FALLBACK = "https://raw.githubusercontent.com/datasets/s-and-p-500-companies/main/data/constituents.csv"
COST_BP = 10.0           # per unit traded, each side (monthly rebalanced books)
FEATURES = ["mom12_2", "mom1", "ind_mom", "vol1m", "ivol3m", "beta12m", "max1m", "hi52", "ldvol", "lprice"]
OLS3 = ["mom12_2", "ldvol", "vol1m"]


# ---------------------------------------------------------------- data

def fetch_yahoo(ticker: str, start: str = START) -> pd.DataFrame:
    t0 = int(pd.Timestamp(start, tz="UTC").timestamp())
    r = requests.get(f"https://query2.finance.yahoo.com/v8/finance/chart/{ticker}",
                     params={"period1": t0, "period2": int(time.time()) + 86400, "interval": "1d", "events": "div,splits",
                             "includeAdjustedClose": "true"},
                     headers={"User-Agent": "Mozilla/5.0"}, timeout=30)
    r.raise_for_status()
    return parse_yahoo_chart(r.json())


def parse_yahoo_chart(payload: dict) -> pd.DataFrame:
    res = payload["chart"]["result"][0]
    q = res["indicators"]["quote"][0]
    adj = res["indicators"].get("adjclose", [{}])[0].get("adjclose", q["close"])
    dates = pd.to_datetime(res["timestamp"], unit="s", utc=True).tz_convert("America/New_York").strftime("%Y-%m-%d")
    df = pd.DataFrame({"date": dates, "open": q["open"], "close": q["close"], "adj_close": adj, "volume": q["volume"]})
    df[["close", "adj_close"]] = df[["close", "adj_close"]].astype(float)
    df = df.dropna(subset=["close"])
    # The daily bars can lag a session behind the quote. Once the market has closed, the quote's price is that day's
    # close: add it as a bar.
    meta = res.get("meta") or {}
    t, px = meta.get("regularMarketTime"), meta.get("regularMarketPrice")
    if t and px:
        ts = pd.Timestamp(int(t), unit="s", tz="UTC").tz_convert("America/New_York")
        day = ts.strftime("%Y-%m-%d")
        if (df.empty or day > df["date"].iloc[-1]) and ts.hour >= 16 and ts.weekday() < 5:
            df = pd.concat([df, pd.DataFrame([{"date": day, "open": None, "close": float(px), "adj_close": None,
                                               "volume": meta.get("regularMarketVolume")}])], ignore_index=True)
    # Yahoo often leaves the latest session's adjusted close empty until the next day: carry the last adjustment.
    df["adj_close"] = df["adj_close"].fillna(df["close"] * (df["adj_close"] / df["close"]).ffill())
    return df.dropna(subset=["adj_close"]).reset_index(drop=True)


def sp500_tables(html: str | None = None) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Current S&P 500 members (ticker, sector) and the dated list of additions and removals."""
    if html is None:
        r = requests.get(WIKI, headers={"User-Agent": "experiment-1 research (https://github.com/pindustrycom/experiment-1)"}, timeout=30)
        r.raise_for_status()
        html = r.text
    tabs = pd.read_html(io.StringIO(html))
    for t in tabs:
        t.columns = ["_".join(dict.fromkeys(str(x) for x in c)) if isinstance(c, tuple) else str(c) for c in t.columns]
    cur0 = next(t for t in tabs if "Symbol" in t.columns)
    cur = cur0.rename(columns={"Symbol": "ticker", next(c for c in cur0.columns if "Sector" in c): "sector"})[["ticker", "sector"]]
    ch = next((t for t in tabs if any("Added" in c for c in t.columns) and any("Removed" in c for c in t.columns)), None)
    if ch is None and "Date added" in cur0.columns:
        # no change list on the page: use each current member's date added (removed companies stay missing)
        changes = pd.DataFrame({"date": pd.to_datetime(cur0["Date added"], errors="coerce", format="mixed"), "added": cur0["Symbol"], "removed": None})
        return cur, changes.dropna(subset=["date"])
    try:
        if ch is None:
            raise StopIteration

        def pick(word):
            cols = [c for c in ch.columns if word in c]
            return next((c for c in cols if "Ticker" in c or "Symbol" in c), cols[0])

        changes = pd.DataFrame({"date": pd.to_datetime(ch[ch.columns[0]], errors="coerce", format="mixed"), "added": ch[pick("Added")],
                                "removed": ch[pick("Removed")]}).dropna(subset=["date"])
    except StopIteration:
        raise ValueError(f"no changes table; tables' columns: {[list(t.columns)[:6] for t in tabs[:4]]}") from None
    return cur, changes


def membership(cur: pd.DataFrame, changes: pd.DataFrame, months: pd.DatetimeIndex) -> pd.DataFrame:
    """Boolean month x ticker table of S&P 500 membership, walking the change list backwards from today."""
    members = set(cur["ticker"])
    ch = changes.sort_values("date", ascending=False)
    rows, k = {}, 0
    events = list(ch.itertuples(index=False))
    for m in sorted(months, reverse=True):
        while k < len(events) and events[k].date > m:
            if isinstance(events[k].added, str):
                members.discard(events[k].added)
            if isinstance(events[k].removed, str):
                members.add(events[k].removed)
            k += 1
        rows[m] = set(members)
    tickers = sorted(set().union(*rows.values()))
    return pd.DataFrame({t: [t in rows[m] for m in sorted(rows)] for t in tickers}, index=sorted(rows))


def ysym(t: str) -> str:
    return t.replace(".", "-")


def update(data: Path, tickers: list[str], pause: float = 0.3, max_age_days: float = 3.0) -> dict:
    d = data / "prices"
    d.mkdir(parents=True, exist_ok=True)
    out = {"fetched": 0, "failed": []}
    for t in tickers:
        f = d / f"{t}.csv"
        if f.exists() and time.time() - f.stat().st_mtime < max_age_days * 86400:
            continue
        try:
            df = fetch_yahoo(ysym(t))
            if len(df) > 60:
                df.to_csv(f, index=False)
                out["fetched"] += 1
            else:
                out["failed"].append(t)
        except Exception:  # noqa: BLE001 - delisted names are expected to fail
            out["failed"].append(t)
        if out["failed"][-1:] == [t] and openbb_src.supported(t):
            _openbb_fallback(t, f, out)
        time.sleep(pause)
    return out


def _openbb_fallback(t: str, f: Path, out: dict) -> None:
    """Yahoo failed for an index only it carries: fetch it through OpenBB instead, unless that series disagrees with
    the stored one (then the stored file is kept and the ticker stays failed)."""
    try:
        df = openbb_src.fetch(t, START)
        old = pd.read_csv(f, dtype={"date": str}) if f.exists() else None
        if len(df) <= 60 or not openbb_src.agrees(t, old, df):
            out.setdefault("openbb_refused", []).append(t)
            return
        df.to_csv(f, index=False)
        out["failed"].remove(t)
        out["fetched"] += 1
        out.setdefault("openbb", []).append(t)
    except Exception as e:  # noqa: BLE001 - the fallback must never stop a run
        out.setdefault("openbb_refused", []).append(t)
        log.warning("OpenBB fallback for %s failed: %s", t, e)


def load(data: Path, tickers: list[str]) -> dict[str, pd.DataFrame]:
    out = {}
    for t in tickers:
        f = data / "prices" / f"{t}.csv"
        if f.exists():
            d = pd.read_csv(f, parse_dates=["date"]).drop_duplicates("date").set_index("date").sort_index()
            if len(d) > 60:
                out[t] = d
    return out


# ---------------------------------------------------------------- 1. variance risk premium

def sv_filter(r: np.ndarray, fit: np.ndarray) -> np.ndarray:
    """One-state stochastic volatility, h_t = mu + phi (h_{t-1} - mu) + w, log(r^2 + c) = h + noise (var pi^2/2), filtered
    by a Kalman filter; (mu, phi, q) chosen by maximum likelihood on the `fit` rows by a coarse grid. Returns the
    one-day-ahead forecast of log variance at each day (uses data up to that day)."""
    y = np.log(r ** 2 + 1e-4)
    R = math.pi ** 2 / 2

    def run(mu, phi, q):
        h, P = mu, q / (1 - phi ** 2)
        pred, ll = np.empty(len(y)), 0.0
        for t in range(len(y)):
            if not np.isnan(y[t]):
                S = P + R
                e = y[t] - (h - 1.27)                     # E log chi2(1) = -1.27
                if fit[t]:
                    ll += -0.5 * (math.log(S) + e * e / S)
                K = P / S
                h, P = h + K * e, (1 - K) * P
            h, P = mu + phi * (h - mu), phi ** 2 * P + q
            pred[t] = h
        return pred, ll

    mu0 = float(np.nanmean(y[fit])) + 1.27
    best = max(((mu0, phi, q) for phi in (0.95, 0.97, 0.98, 0.99) for q in (0.005, 0.01, 0.02, 0.05)), key=lambda p: run(*p)[1])
    return run(*best)[0]


def vrp(data: Path, train_end: str = TRAIN_END) -> tuple[dict, pd.DataFrame]:
    """Monthly: VRP_t = VIX_t^2 / 12 - E_t[next month's realised variance]; predict SPY's excess return over h months."""
    px = load(data, ["SPY", "^VIX", "^IRX"])
    spy = px["SPY"]["adj_close"]
    r = 100 * np.log(spy).diff().dropna()
    vix = px["^VIX"]["close"].reindex(r.index).ffill()
    rf = (px["^IRX"]["close"].reindex(r.index).ffill() / 100 / 252 * 100).fillna(0.0)    # % per day
    fit = np.asarray(r.index <= pd.Timestamp(train_end))
    lv_sv = pd.Series(sv_filter(r.to_numpy(), fit), index=r.index)
    m_end = r.groupby(r.index.to_period("M")).apply(lambda s: s.index[-1])
    rv_m = (r ** 2).groupby(r.index.to_period("M")).sum()                                 # monthly realised variance, %^2
    # HAR on monthly realised variance, refit each month on the past (expanding)
    rv_d = (r ** 2).rolling(22).sum()
    har_x = pd.DataFrame({"d": (r ** 2) * 22, "w": (r ** 2).rolling(5).mean() * 22, "m": rv_d})
    df = pd.DataFrame(index=rv_m.index)
    df["date"] = m_end.values
    df["rv"] = rv_m.values
    df["vix_var"] = (vix.reindex(df["date"]).to_numpy() ** 2) / 12
    df["exp_rv_last"] = df["rv"]
    df["exp_rv_sv"] = np.exp(lv_sv.reindex(df["date"]).to_numpy()) * 21                  # next day's variance forecast x 21 sessions
    hx = har_x.reindex(df["date"]).to_numpy()
    ex = (r - rf).groupby(r.index.to_period("M")).sum()                                  # monthly excess log return, %
    df["ex"] = ex.values
    df["rv_next"] = df["rv"].shift(-1)
    har_pred = np.full(len(df), np.nan)
    for i in range(36, len(df)):
        ok = ~np.isnan(hx[:i]).any(1) & ~np.isnan(df["rv_next"].to_numpy()[:i])
        ok[i - 1:] = False                                                               # next month's rv must be known
        if ok.sum() > 60 and not np.isnan(hx[i]).any():
            L = np.log(np.maximum(hx[:i][ok], 1e-3))                                     # HAR in logs (stable)
            A = np.c_[np.ones(ok.sum()), L]
            b = np.linalg.lstsq(A, np.log(df["rv_next"].to_numpy()[:i][ok]), rcond=None)[0]
            har_pred[i] = np.exp(np.r_[1.0, np.log(np.maximum(hx[i], 1e-3))] @ b)
    df["exp_rv_har"] = har_pred
    out = {"months": [str(df.index[0]), str(df.index[-1]), len(df)]}
    test0 = pd.Period(pd.Timestamp(train_end), "M")
    for name in ("last", "har", "sv"):
        df[f"vrp_{name}"] = df["vix_var"] - df[f"exp_rv_{name}"]
        # how well does each forecaster predict next month's realised variance (log scale), on test months
        t = df.index > test0
        e = np.log(df.loc[t, "rv_next"]) - np.log(df.loc[t, f"exp_rv_{name}"].clip(lower=1e-3))
        out.setdefault("rv_forecast_mse_log_test", {})[name] = float(np.nanmean(e ** 2))
    rv12 = df["rv"].rolling(12).mean().to_numpy()
    for h in (1, 3, 6):
        y = df["ex"].rolling(h).sum().shift(-h)                                          # next h months' excess return
        res = {}
        for name in ("vix_var", "vrp_last", "vrp_har", "vrp_sv"):
            x = df[name].to_numpy()
            yy = y.to_numpy()
            pred, bench = np.full(len(df), np.nan), np.full(len(df), np.nan)
            for i in range(60, len(df)):
                ok = ~np.isnan(x[:i]) & ~np.isnan(yy[:i])
                ok[max(0, i - h):] = False                                               # targets must have ended
                if ok.sum() < 36 or np.isnan(x[i]):
                    continue
                A = np.c_[np.ones(ok.sum()), x[:i][ok]]
                b = np.linalg.lstsq(A, yy[:i][ok], rcond=None)[0]
                pred[i] = b[0] + b[1] * x[i]
                bench[i] = yy[:i][ok].mean()
            row = {}
            for per, sel in (("all_oos", df.index >= df.index[60]), ("test", df.index > test0)):
                s = sel & ~np.isnan(pred) & ~np.isnan(yy)
                if s.sum() < 12:
                    continue
                r2 = 1 - np.sum((yy[s] - pred[s]) ** 2) / np.sum((yy[s] - bench[s]) ** 2)
                row[per] = {"r2_oos": float(r2), "n": int(s.sum())}
                if h == 1:   # timing book: weight = clip(pred / (3 x expected variance), 0, 1.5), monthly
                    w = np.clip(pred[s] * 100 / (3 * rv12[s]), 0, 1.5)                # mean-variance weight, risk aversion 3
                    ret = w * yy[s] - COST_BP / 100 * np.abs(np.diff(w, prepend=w[0]))
                    row[per]["timing_sharpe"] = float(ret.mean() / ret.std() * math.sqrt(12))
                    row[per]["buyhold_sharpe"] = float(yy[s].mean() / yy[s].std() * math.sqrt(12))
                    row[per]["avg_weight"] = float(w.mean())
            res[name] = row
        out[f"h{h}"] = res
    out["corr_vrp_variants"] = df[["vrp_last", "vrp_har", "vrp_sv"]].corr().round(2).to_dict()
    return out, df


# ---------------------------------------------------------------- 2. GKX-lite and 3. overnight momentum

def panel(px: dict[str, pd.DataFrame], spy: pd.Series, sectors: dict[str, str], member: pd.DataFrame) -> pd.DataFrame:
    """One row per (month end, stock): features known at the month end, next month's return, and next month's
    overnight / intraday returns; only stocks in the index at that month end."""
    spy_r = np.log(spy).diff()
    rows = []
    for t, d in px.items():
        c = d["adj_close"].astype(float)
        r = np.log(c).diff()
        adj = d["adj_close"] / d["close"]
        o = d["open"].astype(float) * adj
        on = np.log(o / c.shift(1))                      # close t-1 -> open t
        intra = np.log(c / o)                            # open t -> close t
        bad = (o <= 0) | (on.abs() > 0.5)
        on, intra = on.mask(bad), intra.mask(bad)
        dv = (d["close"] * d["volume"]).astype(float)
        mkt = spy_r.reindex(r.index)
        g = r.index.to_period("M")
        mr = r.groupby(g).sum()
        m_on, m_in = on.groupby(g).sum(min_count=10), intra.groupby(g).sum(min_count=10)
        f = pd.DataFrame(index=mr.index)
        f["ret"] = mr
        f["mom12_2"] = mr.shift(1).rolling(11).sum()
        f["mom1"] = mr
        f["on12"] = m_on.rolling(12).sum()
        f["in12"] = m_in.rolling(12).sum()
        f["vol1m"] = r.groupby(g).std()
        cov = (r * mkt).rolling(252).mean() - r.rolling(252).mean() * mkt.rolling(252).mean()
        beta = cov / mkt.rolling(252).var()
        f["beta12m"] = beta.groupby(g).last()
        res = r - beta.shift(1) * mkt
        f["ivol3m"] = res.rolling(63).std().groupby(g).last()
        f["max1m"] = r.groupby(g).max()
        f["hi52"] = (c / c.rolling(252).max()).groupby(g).last()
        f["ldvol"] = np.log(dv.rolling(21).mean()).groupby(g).last()
        f["lprice"] = np.log(d["close"]).groupby(g).last()
        f["mcap_w"] = dv.rolling(63).mean().groupby(g).last()            # weight proxy (no shares outstanding): dollar volume
        f["y"] = mr.shift(-1)
        f["y_on"] = m_on.shift(-1)
        f["y_in"] = m_in.shift(-1)
        f["ticker"] = t
        f["sector"] = sectors.get(t, "Unknown")
        rows.append(f)
    P = pd.concat(rows).reset_index(names="month")
    P["month_end"] = P["month"].dt.to_timestamp("M")
    mem = member.copy()
    mem.index = pd.DatetimeIndex(mem.index).to_period("M")
    ok = [bool(mem.at[m, t]) if (m in mem.index and t in mem.columns) else False for m, t in zip(P["month"], P["ticker"])]
    P = P[np.array(ok)].reset_index(drop=True)
    P["ind_mom"] = P.groupby(["month", "sector"])["mom12_2"].transform("mean")
    return P


def rank_features(P: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    """Cross-sectional ranks scaled to [-1, 1] each month (missing -> 0, the median), as in Gu, Kelly & Xiu."""
    Z = P.groupby("month")[cols].rank(pct=True)
    return (2 * Z - 1).fillna(0.0)


def deciles(P: pd.DataFrame, score: str, ret: str = "y", weight: str | None = None) -> pd.Series:
    """Monthly long top decile minus short bottom decile, net of costs from monthly turnover of each leg."""
    out, prev = {}, (set(), set())
    for m, g in P.dropna(subset=[score, ret]).groupby("month"):
        if len(g) < 50:
            continue
        q = g[score].rank(pct=True)
        hi, lo = g[q > 0.9], g[q <= 0.1]
        if weight:
            rh = np.average(hi[ret], weights=hi[weight].clip(lower=1)) if len(hi) else 0.0
            rl = np.average(lo[ret], weights=lo[weight].clip(lower=1)) if len(lo) else 0.0
        else:
            rh, rl = hi[ret].mean(), lo[ret].mean()
        th, tl = set(hi["ticker"]), set(lo["ticker"])
        turn = (1 - len(th & prev[0]) / max(len(th), 1)) + (1 - len(tl & prev[1]) / max(len(tl), 1))
        prev = (th, tl)
        out[m] = 100 * (rh - rl) - 2 * turn * COST_BP / 100
    return pd.Series(out)


def sstats(s: pd.Series) -> dict:
    s = s.dropna()
    if len(s) < 12 or s.std() == 0:
        return {"n": len(s)}
    return {"n": len(s), "mean_pct_month": float(s.mean()), "sharpe": float(s.mean() / s.std() * math.sqrt(12)), "t": float(s.mean() / s.std() * math.sqrt(len(s)))}


def gkx(P: pd.DataFrame, train_end: str = TRAIN_END, first_test_year: int = 2005, seeds=(0, 1, 2)) -> tuple[dict, pd.DataFrame]:
    import torch
    from sklearn.ensemble import HistGradientBoostingRegressor

    Z = rank_features(P, FEATURES)
    y = P.groupby("month")["y"].transform(lambda s: s - s.mean()).to_numpy()        # next month's return net of the month's average
    yrs = P["month"].dt.year.to_numpy()
    preds = {k: np.full(len(P), np.nan) for k in ("ols3", "ridge", "gbt", "nn")}
    Xall = Z[FEATURES].to_numpy(np.float32)
    for yr in range(first_test_year, int(yrs.max()) + 1):
        tr = (yrs < yr) & ~np.isnan(y)
        tr &= ~((yrs == yr - 1) & (P["month"].dt.month.to_numpy() == 12))             # December's target is January's return
        te = yrs == yr
        if tr.sum() < 5000 or te.sum() == 0:
            continue
        X, Y = Xall[tr], y[tr]
        A = np.c_[np.ones(len(X)), Z[OLS3].to_numpy()[tr]]
        b = np.linalg.lstsq(A, Y, rcond=None)[0]
        preds["ols3"][te] = np.c_[np.ones(te.sum()), Z[OLS3].to_numpy()[te]] @ b
        w = np.linalg.solve(X.T @ X + 1000.0 * np.eye(X.shape[1]), X.T @ (Y - Y.mean()))
        preds["ridge"][te] = Xall[te] @ w + Y.mean()
        g = HistGradientBoostingRegressor(max_depth=3, learning_rate=0.05, max_iter=300, min_samples_leaf=500, l2_regularization=1.0, random_state=0)
        g.fit(X, Y)
        preds["gbt"][te] = g.predict(Xall[te])
        nn_p = []
        for s in seeds:
            torch.manual_seed(s)
            net = torch.nn.Sequential(torch.nn.Linear(X.shape[1], 32), torch.nn.ReLU(), torch.nn.Linear(32, 16), torch.nn.ReLU(), torch.nn.Linear(16, 8),
                                      torch.nn.ReLU(), torch.nn.Linear(8, 1))
            opt = torch.optim.Adam(net.parameters(), lr=1e-3, weight_decay=1e-4)
            Xt, Yt = torch.tensor(X), torch.tensor(Y, dtype=torch.float32)[:, None]
            n = len(Xt)
            va = torch.tensor(np.random.default_rng(s).random(n) < 0.15)                  # random 15% for early stopping
            best, state, bad = 1e9, None, 0
            for _ in range(40):
                perm = torch.randperm(n)
                for k in range(0, n, 2048):
                    idx = perm[k:k + 2048]
                    idx = idx[~va[idx]]
                    loss = ((net(Xt[idx]) - Yt[idx]) ** 2).mean()
                    opt.zero_grad()
                    loss.backward()
                    opt.step()
                with torch.no_grad():
                    vl = float(((net(Xt[va]) - Yt[va]) ** 2).mean())
                if vl < best - 1e-7:
                    best, state, bad = vl, {k: v.clone() for k, v in net.state_dict().items()}, 0
                else:
                    bad += 1
                    if bad >= 5:
                        break
            net.load_state_dict(state)
            with torch.no_grad():
                nn_p.append(net(torch.tensor(Xall[te])).numpy()[:, 0])
        preds["nn"][te] = np.mean(nn_p, 0)
        log.info("GKX year %d fitted", yr)
    for k, v in preds.items():
        P[f"p_{k}"] = v
    res = {}
    test = P["month_end"] > pd.Timestamp(train_end)
    early = (P["month_end"] <= pd.Timestamp(train_end)) & (P["month"].dt.year >= first_test_year)
    for k in preds:
        row = {}
        for per, sel in (("2005_2018", early), ("test_2019_on", test)):
            s = sel & P[f"p_{k}"].notna()
            yy, pp = y[s.to_numpy()], P.loc[s, f"p_{k}"].to_numpy()
            ok = ~np.isnan(yy)
            row[per] = {"r2_oos_vs_zero": float(1 - np.sum((yy[ok] - pp[ok]) ** 2) / np.sum(yy[ok] ** 2)),
                        "ls_equal": sstats(deciles(P[s], f"p_{k}")), "ls_value": sstats(deciles(P[s], f"p_{k}", weight="mcap_w"))}
        res[k] = row
    # single-feature deciles in the test years (which characteristics carry the ranking)
    res["single_features_test"] = {f: sstats(deciles(P[test], f)) for f in FEATURES}
    return res, P


def overnight(P: pd.DataFrame, train_end: str = TRAIN_END) -> dict:
    """Lou, Polk & Skouras: sort on past-12-month overnight, intraday and total momentum; where does next month's long-short return arrive?"""
    out = {}
    for per, sel in (("1999_2018", P["month_end"] <= pd.Timestamp(train_end)), ("test_2019_on", P["month_end"] > pd.Timestamp(train_end))):
        g = P[sel]
        row = {}
        for sig in ("mom12_2", "on12", "in12"):
            row[sig] = {part: sstats(deciles(g, sig, ret)) for part, ret in (("total", "y"), ("overnight", "y_on"), ("intraday", "y_in"))}
        out[per] = row
    return out


def run(data: Path, out: Path, fetch: bool = True, train_end: str = TRAIN_END) -> dict:
    t0 = time.time()
    out.mkdir(parents=True, exist_ok=True)
    rep = {"built": datetime.now(UTC).strftime("%Y-%m-%d %H:%M UTC"), "train_end": train_end, "warnings": []}
    data.mkdir(parents=True, exist_ok=True)
    cur_f, ch_f = data / "sp500_current.csv", data / "sp500_changes.csv"
    if fetch:
        try:
            cur, changes = sp500_tables()
            cur.to_csv(cur_f, index=False)
            changes.to_csv(ch_f, index=False)
        except Exception as ex:  # noqa: BLE001
            log.exception("S&P 500 list from Wikipedia")
            rep["warnings"].append(f"S&P 500 list from Wikipedia: {type(ex).__name__}: {str(ex)[:200]}")
    if not cur_f.exists() and fetch:
        # fallback: today's members only (no change history, so the universe has full survivorship bias)
        r = requests.get(FALLBACK, timeout=30)
        r.raise_for_status()
        c = pd.read_csv(io.StringIO(r.text)).rename(columns={"Symbol": "ticker", "GICS Sector": "sector", "Sector": "sector"})
        c[["ticker", "sector"]].to_csv(cur_f, index=False)
        pd.DataFrame(columns=["date", "added", "removed"]).to_csv(ch_f, index=False)
    cur = pd.read_csv(cur_f)
    changes = pd.read_csv(ch_f, parse_dates=["date"])
    if changes.empty:
        rep["warnings"].append("Used today's S&P 500 members only (no membership history): results carry survivorship bias.")
    elif changes["removed"].isna().all():
        rep["warnings"].append("Membership from current members' dates added only: stocks count from the month they joined, "
                               "but companies that later left the index are missing (partial survivorship bias).")
    rep["membership_changes"] = len(changes)
    months = pd.date_range(START, pd.Timestamp.today(), freq="ME")
    member = membership(cur, changes, months)
    tickers = sorted(member.columns)
    if fetch:
        rep["fetch"] = update(data, ["SPY", "^VIX", "^IRX", *tickers])
        rep["fetch"]["failed"] = rep["fetch"]["failed"][:400]
    rep["universe"] = {"tickers_ever_members": len(tickers)}

    # 1. VRP
    try:
        rep["vrp"], vdf = vrp(data, train_end)
        vdf.to_csv(out / "vrp_monthly.csv")
    except Exception as ex:  # noqa: BLE001
        rep["warnings"].append(f"vrp: {type(ex).__name__}: {ex}")
        log.exception("vrp")
    log.info("VRP done (%.0fs)", time.time() - t0)

    # 2 and 3
    px = load(data, tickers)
    have_mem = member.reindex(columns=[t for t in tickers if t in px])
    rep["universe"]["with_prices"] = len(px)
    rep["universe"]["avg_members_with_prices"] = float(have_mem.sum(axis=1).mean())
    rep["universe"]["members_with_prices_by_year"] = {str(y): int(v) for y, v in have_mem.sum(axis=1).groupby(have_mem.index.year).mean().items()}
    spy = load(data, ["SPY"])["SPY"]["adj_close"]
    sectors = dict(zip(cur["ticker"], cur["sector"]))
    P = panel(px, spy, sectors, member)
    log.info("panel %d rows (%.0fs)", len(P), time.time() - t0)
    rep["overnight_momentum"] = overnight(P, train_end)
    rep["gkx"], P = gkx(P, train_end)
    P[["month", "ticker", "y", "p_ols3", "p_ridge", "p_gbt", "p_nn"]].to_csv(out / "gkx_predictions.csv", index=False, float_format="%.5g")
    rep["seconds"] = round(time.time() - t0, 1)
    (out / "summary.json").write_text(json.dumps(rep, indent=1, default=str))
    return rep

