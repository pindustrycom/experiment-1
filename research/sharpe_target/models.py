"""Walk-forward forecasts for the quarterly panel: LightGBM and an LSTM, each trained on two targets
(next-quarter return vs next-quarter Sharpe), both as cross-sectional ranks.

Rules fixed before running (the video's 6-month gap included):
- features ranked 0..1 across stocks each quarter (missing -> 0.5); market features (VIX, VIX change, SPY 3m) raw
- a model used at quarter end d trains only on rows whose target ended at least 6 months before d
- LightGBM refits every quarter (3 seeds); the LSTM refits every December (3 seeds) on the last 4 quarters of features
Args: out_dir [lgbm|lstm|all]
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

OUT = Path(sys.argv[1])
WHICH = sys.argv[2] if len(sys.argv) > 2 else "all"
panel = pd.read_parquet(OUT / "panel.parquet")
MKT = ["vix", "vix_chg63", "spy_r63"]
FEAT = [c for c in panel.columns if c not in ("ticker", "date", "fwd_end", "y_ret", "y_sharpe")]
STOCK = [c for c in FEAT if c not in MKT]
g = panel.groupby("date")
for c in STOCK:
    panel[c] = g[c].rank(pct=True).fillna(0.5)
for y in ("y_ret", "y_sharpe"):
    panel[y + "_rk"] = g[y].rank(pct=True)
panel[MKT] = panel[MKT].fillna(panel[MKT].median())
dates = sorted(panel["date"].unique())
TEST = [d for d in dates if d >= pd.Timestamp("2004-12-01")]
GAP = pd.DateOffset(months=6)


def lgbm_preds() -> pd.DataFrame:
    import lightgbm as lgb
    out = []
    for d in TEST:
        tr = panel[panel["fwd_end"] <= d - GAP]
        te = panel[panel["date"] == d]
        for tgt in ("y_ret", "y_sharpe"):
            t2 = tr.dropna(subset=[tgt + "_rk"])
            s = np.zeros(len(te))
            for seed in range(3):
                prm = {"objective": "regression", "learning_rate": 0.03, "num_leaves": 31, "min_data_in_leaf": 200,
                       "bagging_fraction": 0.8, "bagging_freq": 1, "feature_fraction": 0.8, "seed": seed,
                       "verbose": -1, "num_threads": 4}
                m = lgb.train(prm, lgb.Dataset(t2[FEAT].to_numpy(), t2[tgt + "_rk"].to_numpy()), 300)
                s += m.predict(te[FEAT].to_numpy()) / 3
            out.append(pd.DataFrame({"date": d, "ticker": te["ticker"].values, "model": "lgbm",
                                     "target": tgt[2:], "score": s}))
        print("lgbm", pd.Timestamp(d).date(), len(tr), flush=True)
    return pd.concat(out)


def lstm_preds() -> pd.DataFrame:
    import torch
    from torch import nn
    torch.set_num_threads(4)
    L = 4
    di = {d: k for k, d in enumerate(dates)}
    X = np.full((len(dates), panel["ticker"].nunique(), len(FEAT)), np.nan, dtype=np.float32)
    tix = {t: k for k, t in enumerate(sorted(panel["ticker"].unique()))}
    ti = panel["ticker"].map(tix).to_numpy()
    qi = panel["date"].map(di).to_numpy()
    X[qi, ti] = panel[FEAT].to_numpy(np.float32)
    mk = panel[MKT].to_numpy(np.float32)
    mk = (mk - mk.mean(0)) / mk.std(0)
    X[qi, ti, len(STOCK):] = mk

    def seq(rows: pd.DataFrame) -> np.ndarray:
        q = rows["date"].map(di).to_numpy()
        t = rows["ticker"].map(tix).to_numpy()
        s = np.stack([X[np.maximum(q - L + 1 + k, 0), t] for k in range(L)], axis=1)
        s[:, :L - 1][np.isnan(s[:, :L - 1])] = 0.5
        return np.nan_to_num(s, nan=0.5)

    class Net(nn.Module):
        def __init__(self, n):
            super().__init__()
            self.rnn = nn.LSTM(n, 32, batch_first=True)
            self.head = nn.Sequential(nn.Dropout(0.2), nn.Linear(32, 1))

        def forward(self, x):
            h, _ = self.rnn(x)
            return self.head(h[:, -1]).squeeze(-1)

    out, models = [], {}
    for d in TEST:
        if not models or pd.Timestamp(d).month == 12:
            tr = panel[panel["fwd_end"] <= d - GAP]
            models = {}
            for tgt in ("y_ret", "y_sharpe"):
                t2 = tr.dropna(subset=[tgt + "_rk"])
                xs = torch.tensor(seq(t2))
                ys = torch.tensor(t2[tgt + "_rk"].to_numpy(np.float32))
                models[tgt] = []
                for seed in range(3):
                    torch.manual_seed(seed)
                    net = Net(len(FEAT))
                    opt = torch.optim.Adam(net.parameters(), lr=1e-3, weight_decay=1e-5)
                    for ep in range(8):
                        perm = torch.randperm(len(xs))
                        for b in range(0, len(xs), 512):
                            idx = perm[b:b + 512]
                            opt.zero_grad()
                            loss = ((net(xs[idx]) - ys[idx]) ** 2).mean()
                            loss.backward()
                            opt.step()
                    net.eval()
                    models[tgt].append(net)
            print("lstm refit", pd.Timestamp(d).date(), len(tr), flush=True)
        te = panel[panel["date"] == d]
        xt = torch.tensor(seq(te))
        for tgt, nets in models.items():
            with torch.no_grad():
                s = np.mean([n(xt).numpy() for n in nets], axis=0)
            out.append(pd.DataFrame({"date": d, "ticker": te["ticker"].values, "model": "lstm",
                                     "target": tgt[2:], "score": s}))
    return pd.concat(out)


if WHICH in ("lgbm", "all"):
    lgbm_preds().to_parquet(OUT / "preds_lgbm.parquet")
if WHICH in ("lstm", "all"):
    lstm_preds().to_parquet(OUT / "preds_lstm.parquet")
