"""The biggest Reddit hype spikes (posts today >= ~7x the 20-day norm and >= 20 posts), one row each, with the
market-adjusted return that day and over the next 1/5/20 days. Shows that the 2023+ "hype spikes fall ~9% next
week" average is First Republic Bank's collapse, not a pattern.

    python hype_events.py SCRATCH_DIR OUT.csv   (SCRATCH_DIR holds panel.pkl and posts/)
"""
import sys, re, numpy as np, pandas as pd
sys.path.insert(0, '.')
import reddit_study as R
from pathlib import Path
S = sys.argv[1]
P = pd.read_pickle(f"{S}/panel.pkl")
close = P["close"].loc["2017-06-01":]; dates = close.index; mem = P["member"].loc[dates]
base = {re.sub(r"-\d{6}$", "", c): c for c in close.columns}
uni = set(base)
df = R.load_posts(Path(f"{S}/posts"), uni)
df["tk"] = df["tk"].map(base)
D = R.daily(df, dates); one = D["one"].reindex(columns=close.columns).fillna(0)
l1 = np.log1p(one); att1 = l1 - l1.shift(1).rolling(20).mean()
r = close.pct_change(fill_method=None); spy = P["etf"]["SPY"].reindex(dates).pct_change()
lr, ls = np.log1p(r), np.log1p(spy)
out = []
big = (att1 >= 2) & (one >= 20) & mem
for d, t in big.stack().loc[lambda x: x].index:
    i = dates.get_loc(d)
    row = {"date": d.date(), "ticker": t, "posts": int(one.at[d, t]), "ret_that_day_%": round(100 * (r.at[d, t] - spy.at[d]), 1)}
    for h in (1, 5, 20):
        if i + h < len(dates):
            row[f"next_{h}d_%"] = round(100 * float(lr[t].iloc[i + 1:i + 1 + h].sum() - ls.iloc[i + 1:i + 1 + h].sum()), 1)
    out.append(row)
E = pd.DataFrame(out)
E.to_csv(sys.argv[2], index=False)
T = E[E.date >= pd.Timestamp("2023-01-01").date()]
print(T.ticker.value_counts().head(15))
print(T.groupby("ticker")["next_5d_%"].agg(["count", "mean"]).sort_values("count", ascending=False).head(15))
# one event per ticker per 20 days, to avoid counting the same hype episode many times
T = T.sort_values("date"); keep, last = [], {}
for x in T.itertuples():
    if x.ticker not in last or (pd.Timestamp(x.date) - last[x.ticker]).days > 30:
        keep.append(x.Index); last[x.ticker] = pd.Timestamp(x.date)
K = T.loc[keep]
for h in (1, 5, 20):
    v = K[f"next_{h}d_%"].dropna(); print(h, len(v), round(v.mean(), 2), round(v.median(), 2), round(v.mean() / v.std() * np.sqrt(len(v)), 2), "share down", round((v < 0).mean(), 2))
print(K[["date","ticker","posts","ret_that_day_%","next_5d_%","next_20d_%"]].to_string())
