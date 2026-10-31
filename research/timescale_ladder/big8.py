"""Can the time-scale ladder flag the 8% days?

The ladder predicts the size of moves, not their sign. So split the question in two, on every point-in-time
S&P 500 stock-day 2000-2026 (sp500-data daily OHLC; the intraday high-low range is the finest layer here):
  1. Size: P(|next-day return| >= 8%) from the ladder's volatility layers, month -> week -> day -> intraday range,
     plus the quarterly cycle (earnings come about every 63 trading days, so a big move or volume spike ~63
     days ago flags the next report) and the market's own layer (SPY).
  2. Sign: on the days flagged as likely 8% movers, can anything (the drift states at each scale, the gap, the
     market, the size forecast itself) tell an 8% rise from an 8% fall?
Logistic regression and gradient boosting, train 2000-2015, test 2016-2026.

    python big8.py PANEL.pkl OUT_DIR   (PANEL from research/daily_movers/panel.py)
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

BIG = 0.08
TEST = pd.Timestamp("2016-01-01")


def build(P: dict) -> pd.DataFrame:
    C = P["close"].ffill(limit=5)
    O, H, L, V = P["open"], P["high"], P["low"], P["volume"]
    r = np.log(C / C.shift(1))
    gap = np.log(O / C.shift(1))
    pk = np.log(H / L) ** 2 / (4 * np.log(2))           # intraday range variance (Parkinson)
    rv = (pk + gap.fillna(0) ** 2).clip(lower=1e-8)        # day variance: intraday range + overnight gap
    lrv = np.log(rv)
    vrel = V / V.rolling(20, min_periods=10).mean().shift(1)
    spy = np.log(P["etf"]["SPY"].ffill()).diff()
    spy_rv = np.log(spy.pow(2).rolling(5).mean().clip(lower=1e-8))
    f = {
        # volatility ladder: day, week, month, quarter layers of each stock's own variance
        "lrv_d": lrv, "lrv_w": np.log(rv.rolling(5).mean()), "lrv_m": np.log(rv.rolling(22).mean()),
        "lrv_q": np.log(rv.rolling(63).mean()),
        # finest layer: today's range relative to its close-to-close move, and the gap
        "range_vs_cc": np.log(pk.clip(lower=1e-8)) - np.log((r ** 2).clip(lower=1e-8)), "abs_gap": gap.abs(),
        # quarterly (earnings) cycle: biggest move and volume spike 60-66 days ago
        "abs_r_q": r.abs().shift(60).rolling(7).max(), "vrel_q": vrel.shift(60).rolling(7).max(),
        "vrel_d": np.log(vrel.clip(lower=1e-3)), "vrel_w": np.log(vrel.rolling(5).mean().clip(lower=1e-3)),
        # drift layers (direction): day, week, month, year returns, and the market's
        "r_d": r, "r_w": r.rolling(5).sum(), "r_m": r.rolling(22).sum(), "r_y": r.rolling(252).sum().shift(22),
        "gap": gap, "dist_hi": np.log(C / C.rolling(252, min_periods=60).max()),
        "logp": np.log(P["close"]),
    }
    mkt = {"spy_r_d": spy, "spy_r_w": spy.rolling(5).sum(), "spy_lrv_w": spy_rv}
    nxt = r.shift(-1)
    ok = P["member"] & ~P["notrade"] & ~P["notrade"].shift(-1, fill_value=False) & (P["close"] > 5) & nxt.notna()
    st = {k: v.where(ok).stack() for k, v in f.items()}
    D = pd.DataFrame(st)
    D["y_next"] = nxt.where(ok).stack()
    D = D.dropna(subset=["y_next", "lrv_m"]).reset_index().rename(columns={"level_0": "date", "level_1": "ticker"})
    for k, v in mkt.items():
        D[k] = D["date"].map(v)
    D["dow"] = D["date"].dt.dayofweek
    D["big"] = (D["y_next"].abs() >= np.log(1 + BIG)).astype(int)
    D["big_up"] = (D["y_next"] >= np.log(1 + BIG)).astype(int)
    return D


SIZE = ["lrv_d", "lrv_w", "lrv_m", "lrv_q", "range_vs_cc", "abs_gap", "abs_r_q", "vrel_q", "vrel_d", "vrel_w", "spy_lrv_w", "logp", "dow"]
SIGN = ["r_d", "r_w", "r_m", "r_y", "gap", "dist_hi", "spy_r_d", "spy_r_w", "vrel_d", "lrv_d", "lrv_m", "abs_r_q", "p_big"]


def fit_both(X_tr, y_tr, X_te):
    lr = LogisticRegression(max_iter=500).fit((X_tr - X_tr.mean()) / X_tr.std(), y_tr)
    p_lr = lr.predict_proba((X_te - X_tr.mean()) / X_tr.std())[:, 1]
    gb = lgb.LGBMClassifier(n_estimators=300, learning_rate=0.05, num_leaves=31, min_child_samples=500,
                            subsample=0.5, subsample_freq=1, colsample_bytree=0.8, verbose=-1).fit(X_tr, y_tr)
    return p_lr, gb.predict_proba(X_te)[:, 1], gb


def main(panel: str, out: Path) -> None:
    out.mkdir(parents=True, exist_ok=True)
    D = build(pd.read_pickle(panel))
    cols = sorted(set(SIZE + SIGN[:-1]))
    D[cols] = D[cols].replace([np.inf, -np.inf], np.nan)
    D = D.dropna(subset=cols)
    tr, te = D["date"] < TEST, D["date"] >= TEST
    res = {"rows_train": int(tr.sum()), "rows_test": int(te.sum()), "base_rate_big_%": round(100 * D.loc[te, "big"].mean(), 2)}

    # 1. Size
    p_lr, p_gb, gb = fit_both(D.loc[tr, SIZE].to_numpy(), D.loc[tr, "big"].to_numpy(), D.loc[te, SIZE].to_numpy())
    T = D.loc[te].copy()
    T["p_big"] = p_gb
    res["size_auc_logit"] = round(roc_auc_score(T["big"], p_lr), 3)
    res["size_auc_gbm"] = round(roc_auc_score(T["big"], p_gb), 3)
    res["size_auc_vol_only"] = round(roc_auc_score(T["big"], T["lrv_m"]), 3)
    imp = pd.Series(gb.booster_.feature_importance("gain"), SIZE)
    res["size_importance_%"] = (100 * imp / imp.sum()).round(1).sort_values(ascending=False).to_dict()
    T["rank"] = T.groupby("date")["p_big"].rank(ascending=False, method="first")
    for n in (1, 5, 25):
        s = T[T["rank"] <= n]
        res[f"top{n}_per_day"] = {"P(|move|>=8%)_%": round(100 * s["big"].mean(), 1), "P(up>=8%)_%": round(100 * s["big_up"].mean(), 1),
                                  "P(down>=8%)_%": round(100 * (s["big"] - s["big_up"]).mean(), 1),
                                  "mean_next_%": round(100 * np.expm1(s["y_next"]).mean(), 3),
                                  "up_share_%": round(100 * (s["y_next"] > 0).mean(), 1)}
    dec = pd.qcut(T["p_big"], 10, labels=False)
    res["by_decile_P(big)_%"] = (100 * T.groupby(dec)["big"].mean()).round(2).tolist()

    # 2. Sign, given a big move is likely: train on train-period days in the top 5% of predicted size
    D.loc[tr, "p_big"] = gb.predict_proba(D.loc[tr, SIZE].to_numpy())[:, 1]  # in-sample size score for the train sign model
    hi_tr = D.loc[tr & (D["p_big"] >= D.loc[tr, "p_big"].quantile(0.95))]
    hi_te = T[T["p_big"] >= T["p_big"].quantile(0.95)]
    # (a) among flagged days that DID move 8%: was it up or down?
    a_tr, a_te = hi_tr[hi_tr["big"] == 1], hi_te[hi_te["big"] == 1]
    s_lr, s_gb, _ = fit_both(a_tr[SIGN].to_numpy(), a_tr["big_up"].to_numpy(), a_te[SIGN].to_numpy())
    res["sign_given_big"] = {"n_train": len(a_tr), "n_test": len(a_te), "up_share_test_%": round(100 * a_te["big_up"].mean(), 1),
                             "auc_logit": round(roc_auc_score(a_te["big_up"], s_lr), 3), "auc_gbm": round(roc_auc_score(a_te["big_up"], s_gb), 3)}
    # (b) the tradeable version: among all flagged days, predict P(up 8%) and buy the best one each day
    b_lr, b_gb, _ = fit_both(hi_tr[SIGN].to_numpy(), hi_tr["big_up"].to_numpy(), hi_te[SIGN].to_numpy())
    hi_te = hi_te.assign(p_up=b_gb)
    res["sign_flagged_auc_gbm"] = round(roc_auc_score(hi_te["big_up"], b_gb), 3)
    pick = hi_te.loc[hi_te.groupby("date")["p_up"].idxmax()]
    cost = 0.0010  # 5 bp a side
    net = np.expm1(pick["y_next"]) - cost
    days = T["date"].drop_duplicates().sort_values()
    dly = pd.Series(net.values, pick["date"].values).reindex(days).fillna(0)
    eq = (1 + dly).cumprod()
    yrs = len(days) / 252
    spy = pd.read_pickle(panel)["etf"]["SPY"].ffill()
    spy_eq = spy[spy.index >= TEST]
    res["strategy_buy_best_flagged_2016+"] = {"trades": len(pick), "win_%": round(100 * (net > 0).mean(), 1),
                                              "hit_up8_%": round(100 * pick["big_up"].mean(), 1),
                                              "hit_down8_%": round(100 * (pick["big"] - pick["big_up"]).mean(), 1),
                                              "avg_trade_bp": round(1e4 * net.mean(), 1), "t": round(net.mean() / net.std() * np.sqrt(len(net)), 2),
                                              "$20_becomes": round(20 * eq.iloc[-1], 2), "cagr_%": round(100 * (eq.iloc[-1] ** (1 / yrs) - 1), 1),
                                              "max_dd_%": round(100 * (eq / eq.cummax() - 1).min(), 1),
                                              "SPY_$20_becomes": round(20 * spy_eq.iloc[-1] / spy_eq.iloc[0], 2)}
    # 3. A move-size bet that ignores direction (what a straddle pays before its price): mean |move| on flagged days
    for n in (1, 5):
        s = T[T["rank"] <= n]
        res[f"top{n}_mean_abs_move_%"] = round(100 * np.expm1(s["y_next"].abs()).mean(), 2)
        res[f"top{n}_forecast_day_vol_%"] = round(100 * np.sqrt(np.exp(s["lrv_m"])).mean(), 2)
    res["all_mean_abs_move_%"] = round(100 * np.expm1(T["y_next"].abs()).mean(), 2)
    (out / "big8.json").write_text(json.dumps(res, indent=2))
    print(json.dumps(res, indent=2))


if __name__ == "__main__":
    main(sys.argv[1], Path(sys.argv[2]))
