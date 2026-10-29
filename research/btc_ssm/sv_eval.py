"""Score the stochastic-volatility models against the earlier baselines and the GRU hybrid, then the two follow-ups
that use a volatility forecast: the volatility trade (forecast vs Deribit implied vol) and smoothed position sizing.

    python sv_eval.py DATA_DIR OUT_DIR      (after run.py, run.py --tdist and sv_run.py)

Writes OUT_DIR/sv_results.json and OUT_DIR/sv_tables.md.
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
import evaluate as ev
import run

H = 24
NAMES = {"gbm": "GBM (constant vol)", "ewma": "EWMA", "garch": "GARCH(1,1)", "har": "HAR-RV", "logou": "log-vol OU (SV)",
         "heston": "Heston", "heston_j": "Heston + jumps", "rough": "Rough vol + jumps", "gru": "Standalone GRU",
         "hyb_unc": "Hybrid KF+GRU (uncertainty)", "gru_t": "Standalone GRU, Student-t", "hyb_unc_t": "Hybrid KF+GRU, Student-t"}
SV = ["gbm", "ewma", "garch", "har", "logou", "heston", "heston_j", "rough"]


def load_sv(out):
    hourly, daily, info = {}, {}, {}
    for p in sorted(glob.glob(f"{out}/sv/*.pkl")):
        x = pd.read_pickle(p)
        m, y = os.path.basename(p)[:-4].rsplit("_", 1)
        hourly.setdefault(m, []).append(x["hourly"])
        daily.setdefault(m, []).append(x["daily"])
        info.setdefault(m, {})[y] = x["info"]
    return ({m: pd.concat(v).sort_index() for m, v in hourly.items()},
            {m: pd.concat(v).sort_index() for m, v in daily.items()}, info)


def neural(d, out):
    """Hourly ll/pit/var and day/30-day forecasts (persistence of the next-hour variance) for the GRU models."""
    hourly, daily = {}, {}
    r = d.r.to_numpy()
    hours = d.index.hour.to_numpy()
    sets = {"gru": ("main", "gru"), "hyb_unc": ("main", "hyb_unc"), "gru_t": ("tdist", "gru"), "hyb_unc_t": ("tdist", "hyb_unc")}
    for name, (prefix, kind) in sets.items():
        files = sorted(glob.glob(f"{out}/preds/{prefix}_{kind}_*.pkl"))
        if not files:
            continue
        by_seed = {}
        nus = {}
        for f in files:
            p = pd.read_pickle(f)
            by_seed.setdefault(p.attrs["seed"], []).append(p)
            nus[(p.attrs["year"], p.attrs["seed"])] = p.attrs.get("nu")
        seeds = [pd.concat(v).sort_index() for v in by_seed.values()]
        e = ev.ensemble(seeds)
        x = e.join(d[["r"]], how="inner")
        if name.endswith("_t"):
            nu = np.nanmean([v for v in nus.values() if v])
            a = x["var"] * (nu - 2) / nu
            z = (x.r - x.mu) / np.sqrt(a)
            ll = st.t.logpdf(z, nu) - 0.5 * np.log(a)
            pit = st.t.cdf(z, nu)
        else:
            ll = -bl.nll(x.r - x.mu, x["var"])
            pit = st.norm.cdf((x.r - x.mu) / np.sqrt(x["var"]))
        hourly[name] = pd.DataFrame({"ll": ll, "pit": pit, "var": x["var"]}, index=x.index)
        # day / 30-day forecasts: next-hour variance de-seasonalized, held flat (no term structure in these models)
        rows_all, f1, f30 = [], [], []
        for y in run.YEARS:
            tr0, _, te0, te1 = run.fold(d, y)
            train = np.zeros(len(r), bool)
            train[tr0:te0] = True
            s = bl.seasonal(r, hours, train)
            rows = np.arange(te0, te1 - 1)
            rows = rows[hours[rows] == 23]
            v1 = e["var"].reindex(d.index[rows + 1]).to_numpy() / s[(hours[rows] + 1) % H]
            rows_all.append(rows)
            f1.append(v1 * H)
            f30.append(v1 * 30 * H)
        rows = np.concatenate(rows_all)
        daily[name] = pd.DataFrame({"f1d": np.concatenate(f1), "f30d": np.concatenate(f30)}, index=d.index[rows] + pd.Timedelta(hours=1))
    return hourly, daily


def realized(d):
    """Sum of squared hourly returns over the next 24 / 720 rows, indexed at the 00:00 UTC decision time."""
    r2 = (d.r.where(~d.gap, 0.0) ** 2)
    out = pd.DataFrame(index=d.index + pd.Timedelta(hours=1))
    out["rv1d"] = r2.rolling(H).sum().shift(-H).to_numpy()
    out["rv30d"] = r2.rolling(30 * H).sum().shift(-30 * H).to_numpy()
    out["dvol"] = d.dvol.to_numpy()
    return out[out.index.hour == 0]


def hourly_table(hourly, d):
    ref = hourly["garch"].ll
    rows = {}
    idx = None
    for m in hourly:
        if hourly[m].ll.notna().any():
            i = hourly[m].ll.dropna().index
            idx = i if idx is None else idx.intersection(i)
    idx = idx[~d.gap.reindex(idx).to_numpy()]
    for m, h in hourly.items():
        if h.ll.notna().sum() == 0:
            continue
        h = h.reindex(idx)
        diff = (-h.ll + ref.reindex(idx)).to_numpy()
        hits = (h.pit < 0.01).to_numpy()
        rows[m] = {"nll": float(-h.ll.mean()), "d_nll_vs_garch": float(np.mean(diff)), "t": float(ev.nw_t(diff)),
                   "cover98": float(((h.pit > 0.01) & (h.pit < 0.99)).mean()), "var1_rate": float(hits.mean()),
                   "kupiec_p": float(ev.kupiec(hits, 0.01)), "up1_rate": float((h.pit > 0.99).mean()),
                   "by_year": {int(y): float(v) for y, v in pd.Series(diff, index=idx).groupby(idx.year).mean().items()}}
    return rows, len(idx)


def qlike(rv, f):
    x = rv / f
    return x - np.log(x) - 1


def daily_table(daily, real, col, rcol, lags):
    idx = None
    for m, df in daily.items():
        i = df[col].dropna().index
        idx = i if idx is None else idx.intersection(i)
    idx = idx.intersection(real[rcol].dropna().index)
    rv = real[rcol].reindex(idx).to_numpy() + 1e-6
    ref = qlike(rv, daily["garch"][col].reindex(idx).to_numpy())
    out = {}
    for m, df in daily.items():
        f = df[col].reindex(idx).to_numpy()
        q = qlike(rv, f)
        out[m] = {"qlike": float(q.mean()), "d_vs_garch": float((q - ref).mean()), "t": float(ev.nw_t(q - ref, lags)),
                  "log_mse": float(np.mean((np.log(rv) - np.log(f)) ** 2)),
                  "r2_log": float(np.corrcoef(np.log(rv), np.log(f))[0, 1] ** 2),
                  "bias_log": float(np.mean(np.log(f) - np.log(rv)))}
    return out, len(idx)


# ------------------------------------------------------------------------- follow-up 1: the volatility trade

def vol_trade(daily, real, d, costs=(1.0, 2.0), gain=0.2):
    """Synthetic 30-day BTC variance swap struck at DVOL^2 (Deribit's 30-day implied vol index, a VIX-style fair strike).
    Each day at 00:00 UTC open one tranche (1/30 of capital); it pays position * gain * (K - RV)/K at expiry, minus an
    entry cost of c vol points per side on the strike (cost/K = 2c/DVOL per unit traded). Strategies:
      short_always: sell variance every day (the variance risk premium)
      cond_<model>: sell only when implied variance exceeds the model's 30-day forecast, else flat
      ls_<model>: sell when K > forecast, buy when K < forecast
    gain scales tranche risk (0.2: a realized variance 2x the strike loses 20% of the tranche); Sharpe and t do not
    depend on it."""
    x = real.dropna(subset=["rv30d", "dvol"])
    K = x.dvol ** 2 * 30 / 365                          # implied variance over 30 days, percent^2 (log-return units)
    payoff = (K - x.rv30d) / K                          # short variance, per unit, in strike units
    res = {}
    series = {}
    for c in costs:
        cost = 2 * c / x.dvol
        pos = {"short_always": pd.Series(1.0, index=x.index)}
        for m, df in daily.items():
            f = df["f30d"].reindex(x.index)
            pos[f"cond_{m}"] = (K > f).astype(float).where(f.notna())
            pos[f"ls_{m}"] = np.sign(K - f).where(f.notna())
        for name, p in pos.items():
            p = p.fillna(0.0)
            per_trade = p * payoff - p.abs() * cost         # in strike units, entry date index
            exp_date = per_trade.index + pd.Timedelta(days=30)
            daily_ret = pd.Series((gain / 30) * per_trade.to_numpy(), index=exp_date).groupby(level=0).sum()
            daily_ret = daily_ret.reindex(pd.date_range(x.index[0], exp_date[-1], freq="D"), fill_value=0.0)
            eq = (1 + daily_ret).cumprod()
            tr = per_trade[p != 0]
            row = {"trades": int((p != 0).sum()), "share_short": float((p > 0).mean()),
                   "mean_payoff": float(tr.mean()) if len(tr) else np.nan,
                   "t_nw30": float(ev.nw_t(per_trade.to_numpy(), 30)),
                   "hit": float((tr > 0).mean()) if len(tr) else np.nan,
                   # overlapping tranches make daily ladder returns autocorrelated, so the plain daily Sharpe is inflated;
                   # use the Newey-West t of the per-trade payoffs divided by sqrt(years) instead
                   "sharpe": float(ev.nw_t(per_trade.to_numpy(), 30) / np.sqrt(len(per_trade) / 365)),
                   "cagr": float(eq.iloc[-1] ** (365 / len(eq)) - 1), "max_dd": float((eq / eq.cummax() - 1).min()),
                   "worst_trade": float(tr.min()) if len(tr) else np.nan}
            if name != "short_always":
                dd = per_trade - (payoff - cost)
                row["t_vs_short_always"] = float(ev.nw_t(dd.to_numpy(), 30))
            res[f"{name}|c{c:g}"] = row
            if c == costs[0]:
                series[name] = daily_ret
    names = [k for k in res if k.endswith(f"|c{costs[0]:g}") and k.startswith("cond_")]
    ps = [2 * (1 - st.norm.cdf(abs(res[k]["t_vs_short_always"]))) for k in names]
    for k, a in zip(names, ev.holm(ps)):
        res[k]["holm_p_vs_short"] = float(a)
    return res, pd.DataFrame(series), {"start": str(x.index[0].date()), "end": str(x.index[-1].date()), "days": len(x),
                                        "mean_K_minus_RV_over_K": float(payoff.mean()),
                                        "median_dvol": float(x.dvol.median())}


# ------------------------------------------------------------------------- follow-up 2: smoothed sizing

def sizing(daily, d, start, end, target=2.5, cap=2.0, band=0.25, cost_bp=10):
    """Daily vol-targeted BTC (and vol-scaled Donchian) with true day-ahead variance forecasts and a no-trade band:
    the weight moves only when the new target differs from the current weight by more than 25%."""
    ret = np.expm1(np.log(d.close.resample("D").last()).diff()).loc[start:end].fillna(0.0)
    don, _ = ev.donchian_daily(d, 25, start, end)
    don = don.reindex(ret.index).fillna(0.0)
    # Donchian exposure per day (1 when the 4h rule holds BTC at the day's start): days with a nonzero return
    out = {"btc_hold": ev.perf(ret), "donchian_4h": ev.perf(don, ret)}
    rv30 = (d.r ** 2).rolling(30 * H).mean().resample("D").last() * H        # 30-day hist daily variance
    rv30.index = rv30.index + pd.Timedelta(days=1)
    fc = {"hist30": rv30} | {m: df["f1d"].set_axis(df.index.floor("D")) for m, df in daily.items()}
    series = {}
    for m, f in fc.items():
        f = f[~f.index.duplicated()].reindex(ret.index)
        want = (target / np.sqrt(f)).clip(upper=cap)
        w = np.zeros(len(want))
        cur = 0.0
        for i, v in enumerate(want.to_numpy()):
            if np.isfinite(v) and (cur == 0 or abs(v - cur) > band * cur):
                cur = v
            w[i] = cur
        w = pd.Series(w, index=ret.index)
        turn = w.diff().abs().fillna(w.abs())
        r_bt = w * ret - turn * cost_bp / 1e4
        r_don = w * don - turn * (don != 0) * cost_bp / 1e4
        out[f"voltgt_{m}"] = ev.perf(r_bt, ret) | {"turnover_per_yr": float(turn.sum() / len(turn) * 365), "avg_w": float(w.mean())}
        out[f"donchian_x_{m}"] = ev.perf(r_don, ret) | {"t_vs_donchian": float(ev.nw_t((r_don - don).to_numpy(), 5))}
        series[f"voltgt_{m}"] = r_bt
        series[f"donchian_x_{m}"] = r_don
    return out, pd.DataFrame(series).assign(btc_hold=ret, donchian_4h=don)


# ------------------------------------------------------------------------- main

def fmt(v, nd=4):
    if v is None or (isinstance(v, float) and not np.isfinite(v)):
        return ""
    return f"{v:.{nd}f}" if isinstance(v, (float, np.floating)) else str(v)


def main(folder, out):
    d, _ = run.prepare(folder)
    hourly, daily, info = load_sv(out)
    nh, nd = neural(d, out)
    hourly |= nh
    daily |= nd
    real = realized(d)
    res = {"params": info}
    res["hourly"], res["hourly_n"] = hourly_table(hourly, d)
    res["day"], res["day_n"] = daily_table(daily, real, "f1d", "rv1d", 5)
    res["month"], res["month_n"] = daily_table(daily, real, "f30d", "rv30d", 30)
    res["vol_trade"], vt_series, res["vol_trade_info"] = vol_trade(daily, real, d)
    end = str((d.index[-1] - pd.Timedelta(days=1)).date())
    res["sizing"], sz = sizing(daily, d, "2021-01-01", end)
    # the vol trade against Donchian over the same days
    don = sz["donchian_4h"]
    common = vt_series.index.intersection(don.index)
    res["vol_trade_vs_donchian"] = {k: {"corr": float(vt_series[k].reindex(common).corr(don.reindex(common))),
                                        "sharpe": res["vol_trade"][f"{k}|c1"]["sharpe"]} for k in vt_series}
    dc = don.reindex(common)
    res["vol_trade_vs_donchian"]["donchian_4h"] = {"sharpe": float(ev.nw_t(dc.to_numpy(), 5) / np.sqrt(len(dc) / 365))}
    with open(f"{out}/sv_results.json", "w") as fh:
        json.dump(res, fh, indent=1, default=lambda o: float(o) if np.isscalar(o) else str(o))
    vt_series.to_csv(f"{out}/vol_trade_daily.csv")
    sz.to_csv(f"{out}/sizing_daily.csv")
    md = tables(res)
    with open(f"{out}/sv_tables.md", "w") as fh:
        fh.write(md)
    print(md)


def tables(res):
    L = [f"## Next-hour distribution, 2021-01 to 2026-10 ({res['hourly_n']} hours; lower NLL is better)", "",
         "| model | NLL | dNLL vs GARCH | t | 98% interval cover | 1% VaR breaches | Kupiec p | upper 1% tail |",
         "|---|---|---|---|---|---|---|---|"]
    for m, r in res["hourly"].items():
        L.append(f"| {NAMES.get(m, m)} | {fmt(r['nll'])} | {fmt(r['d_nll_vs_garch'])} | {fmt(r['t'], 1)} | {fmt(r['cover98'], 4)} | "
                 f"{fmt(r['var1_rate'], 4)} | {fmt(r['kupiec_p'], 3)} | {fmt(r['up1_rate'], 4)} |")
    L += ["", "dNLL vs GARCH by test year:", "", "| model | " + " | ".join(str(y) for y in run.YEARS) + " |", "|---|" + "---|" * len(run.YEARS)]
    for m, r in res["hourly"].items():
        L.append(f"| {NAMES.get(m, m)} | " + " | ".join(fmt(r["by_year"].get(y)) for y in run.YEARS) + " |")
    for key, title in (("day", "Next-day realized variance"), ("month", "Next-30-day realized variance")):
        L += ["", f"## {title} ({res[key + '_n']} forecasts; QLIKE, lower is better)", "",
              "| model | QLIKE | vs GARCH | t | log MSE | R2 (logs) | bias (log f - log rv) |", "|---|---|---|---|---|---|---|"]
        for m, r in sorted(res[key].items(), key=lambda kv: kv[1]["qlike"]):
            L.append(f"| {NAMES.get(m, m)} | {fmt(r['qlike'])} | {fmt(r['d_vs_garch'])} | {fmt(r['t'], 1)} | {fmt(r['log_mse'], 3)} | "
                     f"{fmt(r['r2_log'], 3)} | {fmt(r['bias_log'], 3)} |")
    vi = res["vol_trade_info"]
    L += ["", f"## Follow-up 1: 30-day variance swap at DVOL^2 ({vi['start']} to {vi['end']}, {vi['days']} entry days)", "",
          f"Average (K - RV)/K = {fmt(vi['mean_K_minus_RV_over_K'], 3)} (positive = implied above realized); median DVOL {fmt(vi['median_dvol'], 1)}.", "",
          "| strategy, cost (vol pts/side) | trades | share short | mean payoff/trade | t (NW 30) | hit | Sharpe | CAGR | max DD | worst trade | t vs always short | Holm p |",
          "|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for k, r in res["vol_trade"].items():
        L.append(f"| {k} | {r['trades']} | {fmt(r['share_short'], 2)} | {fmt(r['mean_payoff'], 3)} | {fmt(r['t_nw30'], 1)} | {fmt(r['hit'], 2)} | "
                 f"{fmt(r['sharpe'], 2)} | {fmt(r['cagr'], 3)} | {fmt(r['max_dd'], 3)} | {fmt(r['worst_trade'], 2)} | "
                 f"{fmt(r.get('t_vs_short_always'), 1)} | {fmt(r.get('holm_p_vs_short'), 3)} |")
    L += ["", "Same days as the vol trade: Sharpe (Newey-West t / sqrt(years)) and correlation with Donchian 4h (cost 1 vol pt):", "", "| series | Sharpe | corr with Donchian |", "|---|---|---|"]
    for k, r in res["vol_trade_vs_donchian"].items():
        L.append(f"| {k} | {fmt(r['sharpe'], 2)} | {fmt(r.get('corr'), 2)} |")
    L += ["", "## Follow-up 2: sizing with day-ahead forecasts and a 25% no-trade band (2021-01 to 2026-10, 10 bp)", "",
          "| strategy | CAGR | vol | Sharpe | max DD | alpha/yr vs hold | t | turnover/yr | t vs Donchian |", "|---|---|---|---|---|---|---|---|---|"]
    for k, r in res["sizing"].items():
        L.append(f"| {k} | {fmt(r['cagr'], 3)} | {fmt(r['vol'], 3)} | {fmt(r['sharpe'], 2)} | {fmt(r['max_dd'], 3)} | {fmt(r.get('alpha_ann'), 3)} | "
                 f"{fmt(r.get('t_alpha'), 1)} | {fmt(r.get('turnover_per_yr'), 1)} | {fmt(r.get('t_vs_donchian'), 1)} |")
    return "\n".join(L) + "\n"


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
