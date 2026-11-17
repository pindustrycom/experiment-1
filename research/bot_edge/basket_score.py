"""Score a multi-pair freqtrade run (NostalgiaForInfinity on a 12-coin basket, up to 6 trades open).

  python basket_score.py DATADIR RESULT_ZIP OUT.csv

Equity is freqtrade's own wallet export (cash plus every holding marked to the candle's price), so CAGR, Sharpe and
drawdown are freqtrade's numbers. For the random-timing test each coin's holding weight (its value over equity,
held over the next bar) is shifted in time by its own random offset (at least 20 days), 2000 draws, and the gross
weight x return sum is compared with the real one. Benchmark: equal-weight basket of the same coins, daily rebalanced.
Also reports how much of the profit came from the five best days (crash rebounds, where 5-minute close fills are
least realistic).
"""
import io
import json
import os
import sys
import zipfile

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import evalkit as ek  # noqa: E402

datadir, zpath, out = sys.argv[1:4]
z = zipfile.ZipFile(zpath)
res = json.loads(z.read([n for n in z.namelist() if n.endswith(".json") and not n.endswith("_config.json")][0]))["strategy"]
strat, s = next(iter(res.items()))
w = pd.read_feather(io.BytesIO(z.read([n for n in z.namelist() if n.endswith("_wallet.feather")][0])))
tf = s["timeframe"]
pairs = sorted(p.split("/")[0] for p in s["pairlist"])
closes = pd.DataFrame({b: pd.read_feather(f"{datadir}/{b}_USDT-{tf}.feather").set_index("date").close for b in pairs}).sort_index()
eq = w.groupby("date").total_quote.sum()
closes = closes.reindex(eq.index).ffill()
R = closes.pct_change().fillna(0.0)
val = w[w.currency != "USDT"].pivot_table(index="date", columns="currency", values="total_quote", aggfunc="sum")
W = val.reindex(index=eq.index, columns=pairs).fillna(0.0).div(eq, axis=0).shift(1).fillna(0.0)   # held over the next bar
Wv, Rv = W.to_numpy(), R.to_numpy()
rng = np.random.default_rng(0)
ms = 20 * (288 if tf == "5m" else 24)


def shift_z(m, draws):
    """All circular shifts of each coin's weights at once (FFT), then random offsets combined across coins."""
    Wm, Rm = Wv[m], Rv[m]
    n = len(Rm)
    a = float(np.sum(Wm * Rm))
    S = [np.fft.irfft(np.conj(np.fft.rfft(Wm[:, j])) * np.fft.rfft(Rm[:, j]), n) for j in range(Wm.shape[1])]
    ks = rng.integers(ms, n - ms, size=(draws, len(S)))
    sims = np.array([sum(S[j][k[j]] for j in range(len(S))) for k in ks])
    return a, float((1 + np.sum(sims >= a)) / (1 + draws)), float((a - sims.mean()) / sims.std())


allm = np.ones(len(R), bool)
_, p, zz = shift_z(allm, 2000)
half = pd.Timestamp("2025-01-01", tz="UTC")
_, p1, z1 = shift_z(np.asarray(R.index < half), 500)
_, p2, z2 = shift_z(np.asarray(R.index >= half), 500)
dn = eq.resample("1D").last().pct_change().dropna()
hold = ek.daily(R.mean(axis=1)).reindex(dn.index.tz_convert(None).normalize()).fillna(0.0)
hold.index = dn.index
pf, ph = ek.perf(dn, 365), ek.perf(hold, 365)
X = np.column_stack([np.ones(len(dn)), hold.to_numpy()])
b, tt = ek.nw_t(dn.to_numpy(), X)
tr = pd.DataFrame(s["trades"])
byday = tr.groupby(tr.open_date.str[:10]).profit_abs.sum().sort_values(ascending=False)
row = {"source": "NostalgiaForInfinity (basket)", "strategy": strat, "asset": f"basket of {len(pairs)}", "tf": tf, "native_tf": tf,
       "trades": len(tr), **pf, "gross_cagr": np.nan, "hold_cagr": ph["cagr"], "hold_sharpe": ph["sharpe"], "hold_dd": ph["max_dd"],
       "p_random": p, "z_random": zz, "p_random_h1": p1, "z_random_h1": z1, "p_random_h2": p2, "z_random_h2": z2,
       "alpha_ann": float(b[0] * 365), "alpha_t": float(tt[0]), "beta": float(b[1]), "exposure": float(W.sum(axis=1).mean()),
       "ft_profit_total": s["profit_total"], "win_rate": float((tr.profit_ratio > 0).mean()), "years": len(dn) / 365,
       "top5_days_profit_share": float(byday.head(5).sum() / tr.profit_abs.sum()), "top5_days": ",".join(byday.head(5).index),
       "cagr_h1": ek.perf(dn[dn.index < half], 365)["cagr"], "cagr_h2": ek.perf(dn[dn.index >= half], 365)["cagr"],
       "note": "sample 2023-01 to 2026-10, halves split at 2025-01-01"}
pd.DataFrame([row]).to_csv(out, index=False)
print(json.dumps(row, indent=1, default=str))
