"""Best achievable tilt: every signal in the atlas at once, walk-forward.

Monthly, point-in-time S&P 500 members. Target = the stock beats the members' median over the next month (50% by
construction). Features are the atlas signals ranked 0..1 across members each month (Coqueret & Guida ch.4), plus
two market-state inputs. Each January a model is refit on every month before it, leaving out the last month whose
target is not yet known (purge, Coqueret & Guida ch.10), and predicts that year. Gradient-boosted trees and a
logistic regression. Scored as hit rate of the top 5% by predicted probability, calibration by predicted bucket, AUC,
and the top-minus-bottom 5% spread after 5 bp a side.

    python combo.py PANEL.pkl OUT_DIR
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

from study import COST, TAIL, nw_mean, sector_demean


def features(P: dict) -> tuple[pd.DataFrame, pd.DatetimeIndex]:
    adj, vol, close, opn, mem, sec = P["adj"], P["volume"], P["close"], P["open"], P["member"], P["sector"]
    r = adj.pct_change(fill_method=None)
    mkt = r.where(mem).mean(axis=1)
    m_mean = mkt.rolling(252, min_periods=200).mean()
    cov = r.mul(mkt, axis=0).rolling(252, min_periods=200).mean() - r.rolling(252, min_periods=200).mean().mul(m_mean, axis=0)
    beta = cov.div(mkt.rolling(252, min_periods=200).var(ddof=0), axis=0)
    days = adj.index
    ME = pd.DatetimeIndex(pd.Series(days, index=days).groupby(days.to_period("M")).max().to_numpy())
    spike = (vol / vol.rolling(50, min_periods=40).median().shift(1) >= 3).astype(float).where(vol.notna())
    mr = adj.loc[ME].pct_change(fill_method=None)
    stack = np.stack([mr.shift(k).to_numpy() for k in range(11, 120, 12)])
    cnt = np.isfinite(stack).sum(0)
    with np.errstate(all="ignore"):
        seas = np.where(cnt >= 3, np.nanmean(stack, 0), np.nan)
    gap = (adj / adj.shift(1)) / (close / close.shift(1)) * (opn / close.shift(1)) - 1
    ret6 = adj / adj.shift(126) - 1
    pos = (r > 0).astype(float).where(r.notna()).rolling(231, min_periods=200).mean().shift(21)
    neg = (r < 0).astype(float).where(r.notna()).rolling(231, min_periods=200).mean().shift(21)
    dm = P["dividends"].gt(0).astype(float).groupby(days.to_period("M")).max()
    sm_ = spike.groupby(days.to_period("M")).max()
    dm.index = sm_.index = ME
    raw = {
        "mom12_1": adj.shift(21) / adj.shift(252) - 1, "high52": adj / adj.rolling(252, min_periods=200).max(),
        "ret1m": adj / adj.shift(21) - 1, "ret1w": adj / adj.shift(5) - 1, "ret1d": r,
        "ind_mom6": ret6 - sector_demean(ret6, sec, mem), "vol3m": r.rolling(63, min_periods=50).std(), "beta": beta,
        "max1m": r.rolling(21, min_periods=18).max(),
        "illiq": (r.abs() / (close * vol).replace(0, np.nan)).rolling(21, min_periods=15).mean(),
        "info_disc": neg - pos, "vol_shock": np.log(vol.rolling(5).sum() / (vol.shift(5).rolling(50, min_periods=40).sum() / 10)),
        "on_minus_id": gap.rolling(21).sum() - (close / opn - 1).rolling(21).sum(),
        "dollar_vol": (close * vol).rolling(63, min_periods=40).mean(),
    }
    frames = {k: v.loc[ME] for k, v in raw.items()}
    frames["seas"] = pd.DataFrame(seas, index=ME, columns=adj.columns)
    frames["div_month"] = dm.shift(11)
    frames["earn_month"] = sm_.shift(11)
    adjf = adj.ffill()
    fwd = adjf.loc[ME].shift(-1) / adjf.loc[ME] - 1
    univ = mem.loc[ME] & adj.loc[ME].notna()
    long = []
    for k, f in frames.items():
        s = f.where(univ).rank(axis=1, pct=True) if k not in ("div_month", "earn_month") else f.where(univ)
        long.append(s.stack().dropna().rename(k))
    X = pd.concat(long, axis=1).rename_axis(["date", "ticker"])
    fw = fwd.where(univ).stack().dropna().rename("fwd").rename_axis(["date", "ticker"])
    X = X.join(fw, how="inner")
    spy = P["spy"]["close"]
    state = pd.DataFrame({"mkt_above200": (spy > spy.rolling(200).mean()).astype(float),
                          "vix": P["vix"]}).reindex(days).ffill().loc[ME]
    X = X.join(state.rename_axis("date"), on="date")
    return X, ME


def main(panel: Path, out: Path) -> None:
    cache = out / "combo_features.pkl"
    if cache.exists():
        X = pd.read_pickle(cache)
    else:
        X, _ = features(pd.read_pickle(panel))
        X.to_pickle(cache)
    X = X[X.index.get_level_values(0) >= "2001-01-01"]
    X["y"] = (X["fwd"] > X.groupby(level=0)["fwd"].transform("median")).astype(int)
    feats = [c for c in X.columns if c not in ("fwd", "y")]
    preds = []
    for year in range(2006, 2027):
        test = X[X.index.get_level_values(0).year == year]
        cut = pd.Timestamp(f"{year}-01-01")
        tr_dates = sorted(set(X.index.get_level_values(0)[X.index.get_level_values(0) < cut]))[:-1]  # purge last month
        train = X[X.index.get_level_values(0).isin(tr_dates)]
        if test.empty:
            continue
        gbm = HistGradientBoostingClassifier(max_iter=200, learning_rate=0.05, max_leaf_nodes=15, min_samples_leaf=500,
                                             l2_regularization=1.0, random_state=0)
        gbm.fit(train[feats], train["y"])
        lr = LogisticRegression(max_iter=500)
        lr.fit(train[feats].fillna(0.5), train["y"])
        p = test[["fwd", "y"]].copy()
        p["p_gbm"] = gbm.predict_proba(test[feats])[:, 1]
        p["p_logit"] = lr.predict_proba(test[feats].fillna(0.5))[:, 1]
        preds.append(p)
        print(year, f"AUC gbm {roc_auc_score(p['y'], p['p_gbm']):.3f} logit {roc_auc_score(p['y'], p['p_logit']):.3f}", flush=True)
    pr = pd.concat(preds)
    pr.to_csv(out / "combo_predictions.csv.gz")
    res = {}
    for model in ("p_gbm", "p_logit"):
        per, prev = [], (None, None)
        for d, g in pr.groupby(level=0):
            lo, hi = g[model].quantile([TAIL, 1 - TAIL])
            top, bot = g[g[model] >= hi], g[g[model] <= lo]
            tt = 1.0 if prev[0] is None else 1 - len(set(top.index.get_level_values(1)) & prev[0]) / len(top)
            tb = 1.0 if prev[1] is None else 1 - len(set(bot.index.get_level_values(1)) & prev[1]) / len(bot)
            prev = (set(top.index.get_level_values(1)), set(bot.index.get_level_values(1)))
            ls = top["fwd"].mean() - bot["fwd"].mean()
            per.append(dict(date=d, hit_top=top["y"].mean(), hit_bot=bot["y"].mean(), ls=ls,
                            ls_net=ls - 2 * COST * (tt + tb), long_net=top["fwd"].mean() - g["fwd"].mean() - 2 * COST * tt,
                            auc=roc_auc_score(g["y"], g[model])))
        per = pd.DataFrame(per).set_index("date")
        cal = pr.groupby(pd.qcut(pr[model], 10, labels=False))[[model, "y"]].mean()
        out_m = {}
        for name, (a, b) in {"2006-2026": ("2006", "2026"), "2013+": ("2013", "2026"), "2019+": ("2019", "2026")}.items():
            q = per.loc[a:b]
            tilt, t = nw_mean(q["hit_top"].to_numpy() - 0.5, 3)
            ls, tls = nw_mean(q["ls_net"].to_numpy(), 3)
            lo, tlo = nw_mean(q["long_net"].to_numpy(), 3)
            out_m[name] = dict(months=len(q), p_beat_top5=float(q["hit_top"].mean()), t=t,
                               p_beat_bottom5=float(q["hit_bot"].mean()), auc=float(q["auc"].mean()),
                               ls_net_yr=ls * 12, t_ls=tls, long_vs_all_net_yr=lo * 12, t_long=tlo)
        out_m["calibration_deciles"] = {f"{a:.3f}": round(float(b), 3) for a, b in zip(cal[model], cal["y"])}
        res[model] = out_m
        per.to_csv(out / f"combo_{model}_monthly.csv")
    (out / "combo.json").write_text(json.dumps(res, indent=1))
    print(json.dumps(res, indent=1))


if __name__ == "__main__":
    main(Path(sys.argv[1]), Path(sys.argv[2]))
