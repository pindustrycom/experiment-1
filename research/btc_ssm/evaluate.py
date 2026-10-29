"""Score the walk-forward predictions: risk (volatility / distribution) and direction separately, then trading tests
after costs against BTC buy-and-hold and the live BTC Donchian 4h rule.

    python evaluate.py DATA_DIR OUT_DIR

Writes OUT_DIR/results.json, OUT_DIR/tables.md and OUT_DIR/daily_returns.csv.
"""
from __future__ import annotations

import glob
import json
import os
import sys

import numpy as np
import pandas as pd
from scipy import stats as st

import baselines as bl
import data
import run

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "github_bots"))
from engine import Rules, simulate  # (same backtester as the live Donchian pot's study)

H = 24
NAMES = {"hist": "Random walk + 30d hist vol", "ewma": "RW + seasonal EWMA vol", "garch": "RW + seasonal GARCH(1,1)",
         "ridge": "Ridge on the same inputs", "kf_fixed": "Kalman, fixed A,B,C,Q,R", "kf_feat": "Kalman + vol/jump inputs",
         "gru": "Standalone GRU", "hyb_dyn": "Hybrid: GRU dynamics only", "hyb_unc": "Hybrid: GRU uncertainty only",
         "hyb_both": "Hybrid: both heads"}


def nw_t(x, lags=H):
    """t-stat of the mean with Newey-West (Bartlett) standard errors."""
    x = np.asarray(x, float)
    x = x[np.isfinite(x)]
    n = len(x)
    e = x - x.mean()
    s = e @ e / n
    for k in range(1, lags + 1):
        s += 2 * (1 - k / (lags + 1)) * (e[k:] @ e[:-k]) / n
    return x.mean() / np.sqrt(s / n) if s > 0 else np.nan


def nw_reg(y, x, lags=5):
    """OLS y = a + b x; Newey-West t of a. Returns a, b, t_a."""
    X = np.c_[np.ones(len(x)), x]
    b = np.linalg.lstsq(X, y, rcond=None)[0]
    e = y - X @ b
    n = len(y)
    S = (X * e[:, None]).T @ (X * e[:, None]) / n
    for k in range(1, lags + 1):
        G = (X[k:] * e[k:, None]).T @ (X[:-k] * e[:-k, None]) / n
        S += (1 - k / (lags + 1)) * (G + G.T)
    Q = np.linalg.inv(X.T @ X / n)
    V = Q @ S @ Q / n
    return b[0], b[1], b[0] / np.sqrt(V[0, 0])


def kupiec(hits, p):
    n, x = len(hits), int(hits.sum())
    if x == 0 or x == n:
        return np.nan
    lr = -2 * ((n - x) * np.log(1 - p) + x * np.log(p) - (n - x) * np.log(1 - x / n) - x * np.log(x / n))
    return 1 - st.chi2.cdf(lr, 1)


# ---------------------------------------------------------------- predictions

def baseline_preds(d, f, years, start=None, inputs="price+rv"):
    """Volatility baselines (mu = 0) and a ridge mean forecast, refit each test year like the models."""
    out = {k: [] for k in ("hist", "ewma", "garch", "ridge")}
    r = d.r.to_numpy()
    ok = (~d.gap).to_numpy()
    for y in years:
        tr0, _, te0, te1 = run.fold(d, y, start)
        vb, _ = run.vol_baselines(d, tr0, te0)
        idx = d.index[te0:te1]
        for k in ("hist", "ewma", "garch"):
            out[k].append(pd.DataFrame({"mu": 0.0, "var": vb[k][te0:te1]}, index=idx))
        z = data.standardize(f[data.SETS[inputs]], d.index[te0]).to_numpy()
        X = np.c_[np.ones(len(z)), z]
        rows = np.arange(tr0, te0 - 1)
        rows = rows[ok[rows + 1]]
        lam = 10.0 * len(rows) ** 0.5
        A = X[rows].T @ X[rows] + lam * np.eye(X.shape[1])
        A[0, 0] -= lam
        beta = np.linalg.solve(A, X[rows].T @ r[rows + 1])
        mu = np.r_[np.nan, (X @ beta)[:-1]][te0:te1]          # forecast for row t+1 from row t
        out["ridge"].append(pd.DataFrame({"mu": mu, "var": vb["ewma"][te0:te1]}, index=idx))
    return {k: pd.concat(v) for k, v in out.items()}


def load_preds(out, prefix):
    rows = {}
    for p in sorted(glob.glob(f"{out}/preds/{prefix}_*.pkl")):
        x = pd.read_pickle(p)
        a = x.attrs
        rows.setdefault((a["kind"], a["inputs"], a["seed"]), []).append(x)
    return {k: pd.concat(v).sort_index() for k, v in rows.items()}


def ensemble(seed_preds):
    mus = pd.concat([p.mu for p in seed_preds], axis=1)
    vs = pd.concat([p["var"] for p in seed_preds], axis=1)
    return pd.DataFrame({"mu": mus.mean(1), "var": vs.mean(1) + mus.var(1, ddof=0)}).astype(float)


# ---------------------------------------------------------------- scoring

def score(p, d, ref_nll=None, regime=None):
    x = p.join(d[["r", "gap"]], how="inner")
    x = x[~x.gap & np.isfinite(x.mu) & np.isfinite(x["var"])]
    r, mu, v = x.r.to_numpy(), x.mu.to_numpy(), x["var"].to_numpy()
    z = (r - mu) / np.sqrt(v)
    nl = bl.nll(r - mu, v)
    out = {"n": len(r), "nll": nl.mean()}
    if ref_nll is not None:
        diff = nl - ref_nll.reindex(x.index).to_numpy()
        out["d_nll"], out["t_d_nll"] = np.nanmean(diff), nw_t(diff)
    out["qlike"] = np.mean(r ** 2 / v + np.log(v))
    out["mz_r2_hour"] = np.corrcoef(r ** 2, v)[0, 1] ** 2
    day = x.index.floor("D")
    rv_d = pd.Series(r ** 2, index=x.index).groupby(day).sum()
    v_d = pd.Series(v, index=x.index).groupby(day).sum()
    out["mz_r2_day_log"] = np.corrcoef(np.log(rv_d + 1e-6), np.log(v_d))[0, 1] ** 2
    out["cover90"] = np.mean(np.abs(z) < 1.6449)
    out["cover99"] = np.mean(np.abs(z) < 2.5758)
    hits = r < mu - 2.3263 * np.sqrt(v)
    out["var1_rate"], out["var1_kupiec_p"] = hits.mean(), kupiec(hits, 0.01)
    if np.any(mu != 0):
        nz = mu != 0
        out["hit"] = np.mean(np.sign(mu[nz]) == np.sign(r[nz]))
        out["ic"] = np.corrcoef(mu, r)[0, 1]
        prod = (mu - mu.mean()) * (r - r.mean()) / (mu.std() * r.std())
        out["t_ic"] = nw_t(prod)
        out["r2_oos"] = 1 - np.sum((r - mu) ** 2) / np.sum(r ** 2)
    if regime is not None:
        g = regime.reindex(x.index).to_numpy()
        for k in np.unique(g[pd.notna(g)]):
            m = g == k
            out[f"nll_{k}"] = nl[m].mean()
            if ref_nll is not None:
                out[f"d_nll_{k}"] = np.nanmean(diff[m])
                out[f"t_d_nll_{k}"] = nw_t(diff[m])
    return out, pd.Series(nl, index=x.index)


def regimes(d, ewma_var):
    """Trailing 30-day realized vol terciles (cut points from 2018-2020 only) and jump hours (|r| > 4 EWMA sigma)."""
    rv = (d.r ** 2).rolling(30 * H).mean().shift()          # known before the hour starts
    cuts = rv[:"2020-12-31"].quantile([1 / 3, 2 / 3]).to_numpy()
    lab = pd.Series(np.where(rv < cuts[0], "lowvol", np.where(rv < cuts[1], "midvol", "highvol")), index=d.index)
    jump = (d.r.abs() > 4 * np.sqrt(ewma_var.reindex(d.index))).map({True: "jump", False: "nojump"})
    return lab, jump


# ---------------------------------------------------------------- trading

def strategy_returns(d, preds, cost_bp, start, end):
    """Daily strategies decided at 00:00 UTC from forecasts made up to then.
    - sign: long/flat on the sign of the average of the last 24 hourly mu forecasts (expected drift per hour)
    - ls: long/short on the same sign
    - voltgt: long BTC scaled to 2.5%/day predicted vol (weight cap 2), the risk forecast's use case"""
    px = d.close
    ret_d = np.expm1(np.log(px.resample("D").last()).diff())   # day D: close of the 23:00 bar of D-1 -> of D (00:00 to 00:00 UTC)
    out = {}
    for name, p in preds.items():
        mu_d = p.mu.rolling(H, min_periods=H).mean().resample("D").last()   # last value of day D-1 ... aligned below
        v_d = p["var"].rolling(H, min_periods=H).sum().resample("D").last()
        mu_d.index = mu_d.index + pd.Timedelta(days=1)                       # computed by end of D-1 -> trade on D
        v_d.index = v_d.index + pd.Timedelta(days=1)
        sig = {}
        if (p.mu != 0).any():
            sig["sign"] = (mu_d > 0).astype(float)
            sig["ls"] = np.sign(mu_d)
        sig["voltgt"] = (2.5 / np.sqrt(v_d)).clip(upper=2.0)
        for s, w in sig.items():
            w = w.reindex(ret_d.index).loc[start:end].fillna(0.0)
            rr = ret_d.loc[start:end].fillna(0.0)
            turn = w.diff().abs().fillna(w.abs())
            out[f"{name}|{s}"] = w * rr - turn * cost_bp / 1e4
    return pd.DataFrame(out), ret_d.loc[start:end].fillna(0.0)


def donchian_daily(d, cost_bp, start, end):
    """The live BTC pot's rule (Jesse Donchian on 4h bars: close above the previous 20-bar high and above SMA200,
    exit below the previous 20-bar low), on 4h bars built from the same Binance hours."""
    b = d[["open", "high", "low", "close"]].resample("4h").agg({"open": "first", "high": "max", "low": "min", "close": "last"})
    hi, lo = b.high.rolling(20).max().shift(), b.low.rolling(20).min().shift()
    rules = Rules((b.close > hi) & (b.close > b.close.rolling(200).mean()), b.close < lo)
    ret, trades = simulate(b, rules, 240, cost_bp)
    daily = np.expm1(np.log1p(ret).groupby(ret.index.floor("D")).sum())
    return daily.loc[start:end], len(trades)


def perf(x, bench=None):
    x = x.dropna()
    eq = (1 + x).cumprod()
    yrs = len(x) / 365
    out = {"cagr": eq.iloc[-1] ** (1 / yrs) - 1, "vol": x.std() * np.sqrt(365),
           "sharpe": x.mean() / x.std() * np.sqrt(365) if x.std() > 0 else np.nan,
           "max_dd": (eq / eq.cummax() - 1).min(), "final_x": eq.iloc[-1]}
    if bench is not None:
        b = bench.reindex(x.index).fillna(0)
        a, beta, t = nw_reg(x.to_numpy(), b.to_numpy())
        out.update(alpha_ann=a * 365, beta=beta, t_alpha=t)
    return out


def holm(ps):
    ps = np.asarray(ps, float)
    o = np.argsort(ps)
    adj = np.empty_like(ps)
    m = len(ps)
    run_max = 0
    for i, k in enumerate(o):
        run_max = max(run_max, (m - i) * ps[k])
        adj[k] = min(1.0, run_max)
    return adj


# ---------------------------------------------------------------- side-channel timing (measurement vs input)

def lead_lag(d, f):
    """Does a side channel describe current risk (measurement) or future risk (input)? Daily log RV regressed on the
    channel at the start of the day, controlling for the previous day's log RV; same for the day before."""
    rv = np.log((d.r ** 2).resample("D").sum() + 1e-4)
    res = {}
    for ch in ("fng", "dvol"):
        s = f[ch].resample("D").first()                       # value known at the start of day D
        df = pd.DataFrame({"rv": rv, "rv_prev": rv.shift(), "ch": s, "ch_ext": (s - 50).abs() if ch == "fng" else s}).dropna()
        df = df.loc["2021-04-01":]
        out = {}
        for col in ("ch", "ch_ext") if ch == "fng" else ("ch",):
            X = np.c_[np.ones(len(df)), df.rv_prev, df[col]]
            y = df.rv.to_numpy()
            b = np.linalg.lstsq(X, y, rcond=None)[0]
            e = y - X @ b
            se = np.sqrt(np.diag(np.linalg.inv(X.T @ X)) * e.var())
            out[col] = {"coef": b[2], "t": b[2] / se[2],
                        "corr_same_day": df[col].corr(df.rv), "corr_next_day": df[col].corr(df.rv.shift(-1))}
        res[ch] = out
    return res


# ---------------------------------------------------------------- main

def main(folder, out):
    d, f = run.prepare(folder)
    res = {}

    # ---- main comparison, test years 2021-2026
    base = baseline_preds(d, f, run.YEARS)
    lab, jump = regimes(d, base["ewma"]["var"])
    reg = lab.where(jump != "jump", "jump")
    _, ref_nll = score(base["ewma"], d)
    main = load_preds(out, "main")
    table = {}
    for k, p in base.items():
        table[k] = score(p, d, ref_nll, reg)[0]
    kinds = sorted({k[0] for k in main}, key=list(NAMES).index)
    ens = {}
    for kind in kinds:
        seeds = [p for (kd, _, _), p in main.items() if kd == kind]
        per_seed = pd.DataFrame([score(p, d, ref_nll)[0] for p in seeds])
        ens[kind] = ensemble(seeds)
        row = score(ens[kind], d, ref_nll, reg)[0]
        row["seeds"] = len(seeds)
        row["d_nll_seed_sd"] = per_seed.d_nll.std() if len(seeds) > 1 else 0.0
        row["d_nll_seed_mean"] = per_seed.d_nll.mean()
        table[kind] = row
    _, garch_nll = score(base["garch"], d)
    for k, p in {**base, **ens}.items():
        _, nl = score(p, d)
        diff = (nl - garch_nll.reindex(nl.index)).to_numpy()
        table[k]["d_nll_garch"], table[k]["t_d_nll_garch"] = np.nanmean(diff), nw_t(diff)
    res["main"] = table
    # per-year delta NLL vs EWMA
    yearly = {}
    for k, p in {**base, **ens}.items():
        _, nl = score(p, d)
        diff = (nl - ref_nll.reindex(nl.index))
        yearly[k] = diff.groupby(diff.index.year).mean().to_dict()
    res["d_nll_by_year"] = yearly

    # ---- side-channel ablation, test years 2023-2026, all trained from 2021-04
    abl = load_preds(out, "abl")
    ab = {}
    if abl:
        base_a = baseline_preds(d, f, run.ABL_YEARS, run.ABL_START)
        _, ref_a = score(base_a["ewma"], d)
        for k in ("ewma", "garch"):
            ab[k] = score(base_a[k], d, ref_a)[0]
        for inp in data.SETS:
            base_r = baseline_preds(d, f, run.ABL_YEARS, run.ABL_START, inp)["ridge"]
            ab[f"ridge|{inp}"] = score(base_r, d, ref_a)[0]
        for kind in ("gru", "hyb_both"):
            for inp in data.SETS:
                seeds = [p for (kd, ip, _), p in abl.items() if kd == kind and ip == inp]
                if seeds:
                    per = pd.DataFrame([score(p, d, ref_a)[0] for p in seeds])
                    row = score(ensemble(seeds), d, ref_a)[0]
                    row["d_nll_seed_sd"] = per.d_nll.std()
                    ab[f"{kind}|{inp}"] = row
    res["ablation"] = ab
    res["lead_lag"] = lead_lag(d, f)

    # ---- trading after costs, 2021-01-01 .. last full day
    start, end = "2021-01-01", str((d.index[-1] - pd.Timedelta(days=1)).date())
    trade_preds = {k: base[k] for k in ("hist", "ewma", "garch", "ridge")} | ens
    trades = {}
    daily_all = {}
    for cost in (10, 25):
        sr, hold = strategy_returns(d, trade_preds, cost, start, end)
        don, ntr = donchian_daily(d, 25 if cost == 25 else cost, start, end)
        don = don.reindex(sr.index).fillna(0)
        trades[cost] = {"btc_hold": perf(hold), "donchian_4h": dict(perf(don, hold), trades=ntr)}
        for c in sr:
            row = perf(sr[c], hold)
            dd = sr[c] - don
            row["t_vs_donchian"] = nw_t(dd.to_numpy(), 5)
            trades[cost][c] = row
        if cost == 10:
            daily_all = sr.assign(btc_hold=hold, donchian_4h=don)
    # hourly gross sign-trading (no costs): is there any gross directional edge at all?
    gross = {}
    for k, p in trade_preds.items():
        if (p.mu != 0).any():
            x = p.join(d[["r", "gap"]], how="inner")
            x = x[~x.gap]
            g = np.sign(x.mu) * x.r / 100
            gross[k] = {"ann_ret_gross": g.mean() * 24 * 365, "t": nw_t(g.to_numpy()),
                        "turnover_per_day": float(np.abs(np.diff(np.sign(x.mu))).sum() / 2 / (len(x) / 24))}
    res["hourly_gross"] = gross
    # Holm over every after-cost strategy's alpha vs BTC hold (10 bp)
    names = [k for k in trades[10] if "|" in k]
    ps = [2 * (1 - st.norm.cdf(abs(trades[10][k]["t_alpha"]))) for k in names]
    for k, a in zip(names, holm(ps)):
        trades[10][k]["holm_p_alpha"] = a
    res["trading"] = trades
    daily_all.to_csv(f"{out}/daily_returns.csv")
    with open(f"{out}/results.json", "w") as fh:
        json.dump(res, fh, indent=1, default=lambda o: float(o) if np.isscalar(o) else str(o))
    with open(f"{out}/tables.md", "w") as fh:
        fh.write(tables(res))
    print(tables(res))


def fmt(v, nd=4):
    if v is None or (isinstance(v, float) and not np.isfinite(v)):
        return ""
    return f"{v:.{nd}f}" if isinstance(v, (float, np.floating)) else str(v)


def tables(res):
    L = ["## Risk / distribution forecasts, hourly, test 2021-01 to 2026-10 (lower NLL is better)", "",
         "| model | NLL | dNLL vs EWMA | t | dNLL vs GARCH | t | seed sd | 90% cover | 99% cover | VaR1% rate | Kupiec p | daily log-RV R2 |",
         "|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for k, r in res["main"].items():
        L.append(f"| {NAMES.get(k, k)} | {fmt(r['nll'])} | {fmt(r.get('d_nll'))} | {fmt(r.get('t_d_nll'), 1)} | "
                 f"{fmt(r.get('d_nll_garch'))} | {fmt(r.get('t_d_nll_garch'), 1)} | "
                 f"{fmt(r.get('d_nll_seed_sd'))} | {fmt(r['cover90'], 3)} | {fmt(r['cover99'], 3)} | "
                 f"{fmt(r['var1_rate'], 4)} | {fmt(r['var1_kupiec_p'], 3)} | {fmt(r['mz_r2_day_log'], 3)} |")
    L += ["", "## dNLL vs EWMA by regime (negative = better than EWMA)", "",
          "| model | low vol | mid vol | high vol | jump hours |", "|---|---|---|---|---|"]
    for k, r in res["main"].items():
        L.append(f"| {NAMES.get(k, k)} | " + " | ".join(
            f"{fmt(r.get(f'd_nll_{g}'))} (t {fmt(r.get(f't_d_nll_{g}'), 1)})" for g in ("lowvol", "midvol", "highvol", "jump")) + " |")
    L += ["", "## dNLL vs EWMA by test year", "", "| model | " + " | ".join(str(y) for y in run.YEARS) + " |",
          "|---|" + "---|" * len(run.YEARS)]
    for k, r in res["d_nll_by_year"].items():
        L.append(f"| {NAMES.get(k, k)} | " + " | ".join(fmt(r.get(y)) for y in run.YEARS) + " |")
    L += ["", "## Direction, hourly (sign hit rate, information coefficient, out-of-sample R2 vs zero forecast)", "",
          "| model | hit rate | IC | t(IC) | R2 oos |", "|---|---|---|---|---|"]
    for k, r in res["main"].items():
        if "ic" in r:
            L.append(f"| {NAMES.get(k, k)} | {fmt(r['hit'], 4)} | {fmt(r['ic'], 4)} | {fmt(r['t_ic'], 1)} | {fmt(r['r2_oos'], 5)} |")
    L += ["", "Hourly sign trading, gross of costs:", "", "| model | ann. return gross | t | flips/day |", "|---|---|---|---|"]
    for k, r in res["hourly_gross"].items():
        L.append(f"| {NAMES.get(k, k)} | {fmt(r['ann_ret_gross'], 3)} | {fmt(r['t'], 1)} | {fmt(r['turnover_per_day'], 1)} |")
    if res.get("ablation"):
        L += ["", "## Side channels (test 2023-2026, all trained from 2021-04): dNLL vs EWMA, and IC", "",
              "| model, inputs | dNLL | t | seed sd | IC | t(IC) |", "|---|---|---|---|---|---|"]
        for k, r in res["ablation"].items():
            L.append(f"| {k} | {fmt(r.get('d_nll'))} | {fmt(r.get('t_d_nll'), 1)} | {fmt(r.get('d_nll_seed_sd'))} | "
                     f"{fmt(r.get('ic'))} | {fmt(r.get('t_ic'), 1)} |")
    L += ["", "## Side-channel timing: daily log RV on the channel at the start of the day, controlling for yesterday's RV", "",
          "| channel | form | coef | t | corr same day | corr next day |", "|---|---|---|---|---|---|"]
    for ch, v in res["lead_lag"].items():
        for form, x in v.items():
            L.append(f"| {ch} | {form} | {fmt(x['coef'])} | {fmt(x['t'], 1)} | {fmt(x['corr_same_day'], 3)} | {fmt(x['corr_next_day'], 3)} |")
    for cost, tr in res["trading"].items():
        L += ["", f"## Daily trading 2021-01 to 2026-10, {cost} bp per side (Donchian always 25 bp at the 25 bp row)", "",
              "| strategy | CAGR | vol | Sharpe | max DD | alpha/yr vs hold | t alpha | beta | t vs Donchian | Holm p |",
              "|---|---|---|---|---|---|---|---|---|---|"]
        for k, r in tr.items():
            L.append(f"| {k} | {fmt(r['cagr'], 3)} | {fmt(r['vol'], 3)} | {fmt(r['sharpe'], 2)} | {fmt(r['max_dd'], 3)} | "
                     f"{fmt(r.get('alpha_ann'), 3)} | {fmt(r.get('t_alpha'), 1)} | {fmt(r.get('beta'), 2)} | "
                     f"{fmt(r.get('t_vs_donchian'), 1)} | {fmt(r.get('holm_p_alpha'), 3)} |")
    return "\n".join(L) + "\n"


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
