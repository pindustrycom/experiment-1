"""Quarterly stock panel on the point-in-time S&P 500 for the "LSTM picks 25 stocks every 3 months" test.

At each quarter end (last trading day of Mar/Jun/Sep/Dec) every stock that was an S&P 500 member that day gets
price-based features (all known at that close) and two targets over the next quarter:
  ret    = next-quarter total return (delisted stocks stay at their last price)
  sharpe = next-quarter mean daily return / std of daily returns (the video's "return for the risk taken")
Also writes the daily close-to-close return matrix used by the backtest.
Args: papers_data_dir out_dir
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

D, OUT = Path(sys.argv[1]), Path(sys.argv[2])
OUT.mkdir(parents=True, exist_ok=True)
P = D / "prices"
iv = pd.read_parquet(D / "pit" / "membership_intervals.parquet")
iv = iv[(iv["end"].isna()) | (iv["end"] >= "1998-06-01")]
have = {f.stem for f in P.glob("*.csv")}
iv = iv[iv["ticker"].isin(have)]

spy = pd.read_csv(P / "SPY.csv", parse_dates=["date"]).set_index("date")
cal = spy.index[spy.index >= "1998-01-01"]
close, dvol = {}, {}
for t in sorted(set(iv["ticker"])):
    d = pd.read_csv(P / f"{t}.csv", parse_dates=["date"]).set_index("date")
    d = d[~d.index.duplicated()]
    close[t] = d["adj_close"]
    dvol[t] = d["close"] * d["volume"]
C = pd.DataFrame(close).reindex(cal)
V = pd.DataFrame(dvol).reindex(cal)
tick = list(C.columns)
col = {t: i for i, t in enumerate(tick)}
M = np.zeros(C.shape, dtype=bool)
for t, s, e in iv.itertuples(index=False):
    a = cal.searchsorted(s)
    b = cal.searchsorted(e) if pd.notna(e) else len(cal)
    M[a:b, col[t]] = True

Cf = C.ffill()
R = Cf.pct_change(fill_method=None)            # daily returns; 0 after a delisting (held at last price)
R = R.where(Cf.notna())
R.astype("float32").to_parquet(OUT / "daily_returns.parquet")
pd.DataFrame(M, index=cal, columns=tick).to_parquet(OUT / "members.parquet")

spy_r = spy["adj_close"].reindex(cal).pct_change()
vix = pd.read_csv(P / "^VIX.csv", parse_dates=["date"]).set_index("date")["close"].reindex(cal).ffill()

q = cal.to_period("Q")
qend = cal[np.r_[q[1:] != q[:-1], False]]      # last trading day of each complete quarter
qend = qend[qend >= "1999-06-01"]
print("quarter ends:", qend[0].date(), "->", qend[-1].date(), len(qend))

logp = np.log(Cf)
lr = np.log1p(R.fillna(0))
rows = []
for k, d in enumerate(qend):
    i = cal.get_loc(d)
    if i < 260:
        continue
    m = M[i] & C.iloc[i].notna().to_numpy() & C.iloc[i - 252].notna().to_numpy()
    names = [tick[j] for j in np.where(m)[0]]
    win = R.iloc[i - 251:i + 1][names]
    p = Cf.iloc[i][names]
    f = pd.DataFrame(index=names)
    for n in (21, 63, 126):
        f[f"r{n}"] = p / Cf.iloc[i - n][names] - 1
    f["mom12_1"] = Cf.iloc[i - 21][names] / Cf.iloc[i - 252][names] - 1
    f["vol21"] = win.iloc[-21:].std() * np.sqrt(252)
    f["vol63"] = win.iloc[-63:].std() * np.sqrt(252)
    f["vol252"] = win.std() * np.sqrt(252)
    sm = spy_r.iloc[i - 251:i + 1]
    f["beta252"] = win.apply(lambda s: s.cov(sm)) / sm.var()
    f["maxret21"] = win.iloc[-21:].max()
    f["off_high"] = p / Cf.iloc[i - 251:i + 1][names].max() - 1
    f["ma50"] = p / Cf.iloc[i - 49:i + 1][names].mean() - 1
    f["ma200"] = p / Cf.iloc[i - 199:i + 1][names].mean() - 1
    up = win.iloc[-14:].clip(lower=0).mean()
    dn = (-win.iloc[-14:]).clip(lower=0).mean()
    f["rsi14"] = up / (up + dn + 1e-12)
    f["sharpe252"] = win.mean() / win.std()
    f["sharpe63"] = win.iloc[-63:].mean() / win.iloc[-63:].std()
    f["ldvol"] = np.log(V.iloc[i - 62:i + 1][names].mean() + 1)
    f["dvol_ratio"] = V.iloc[i - 20:i + 1][names].mean() / (V.iloc[i - 62:i + 1][names].mean() + 1)
    # market-wide "fear and greed" proxies (same for every stock; trees can interact them with the stock features)
    f["vix"] = vix.iloc[i]
    f["vix_chg63"] = vix.iloc[i] / vix.iloc[i - 63] - 1
    f["spy_r63"] = (1 + spy_r.iloc[i - 62:i + 1]).prod() - 1
    # targets over the next quarter
    if k + 1 < len(qend):
        j = cal.get_loc(qend[k + 1])
        fw = R.iloc[i + 1:j + 1][names].fillna(0)
        f["y_ret"] = (1 + fw).prod() - 1
        sd = fw.std()
        f["y_sharpe"] = fw.mean() / sd.where(sd > 0)
        f["fwd_end"] = qend[k + 1]
    else:
        f["y_ret"] = np.nan
        f["y_sharpe"] = np.nan
        f["fwd_end"] = pd.NaT
    f["date"] = d
    rows.append(f.reset_index(names="ticker"))
    if k % 20 == 0:
        print(d.date(), len(names))

panel = pd.concat(rows, ignore_index=True)
panel.to_parquet(OUT / "panel.parquet")
print(panel.shape, panel["date"].nunique(), "quarters")
