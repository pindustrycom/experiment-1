"""Weekly reversal on the point-in-time S&P 500 since 1998 (the live pot's rule: at each week's last close buy the 20
members that fell most over 5 sessions, hold a week), with realistic costs, a next-open fill, and its own universe
(equal-weight members) as a second benchmark. Args: papers_data_dir out_dir"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from common import alpha, beta_matched, perf, rf_daily

D, OUT = Path(sys.argv[1]), Path(sys.argv[2])
P = D / "prices"
iv = pd.read_parquet(D / "pit" / "membership_intervals.parquet")
iv = iv[(iv["end"].isna()) | (iv["end"] >= "1997-06-01")]
have = {f.stem for f in P.glob("*.csv")}
print("members since mid-1997:", len(iv), "with a price file:", iv["ticker"].isin(have).sum())
iv = iv[iv["ticker"].isin(have)]

spy = pd.read_csv(P / "SPY.csv", parse_dates=["date"]).set_index("date")
cal = spy.index[spy.index >= "1997-12-01"]
close, opn = {}, {}
for t in sorted(set(iv["ticker"])):
    d = pd.read_csv(P / f"{t}.csv", parse_dates=["date"]).set_index("date")
    d = d[~d.index.duplicated()]
    close[t] = d["adj_close"]
    opn[t] = d["open"] * d["adj_close"] / d["close"]
C = pd.DataFrame(close).reindex(cal)
O = pd.DataFrame(opn).reindex(cal)
last_seen = C.apply(lambda s: s.last_valid_index())
Cf = C.ffill()                                   # a delisted stock stays at its last price (its fall is already in it)
tick = list(C.columns)
col = {t: i for i, t in enumerate(tick)}

# membership matrix on the calendar
M = np.zeros(C.shape, dtype=bool)
for t, s, e in iv.itertuples(index=False):
    a = cal.searchsorted(s)
    b = cal.searchsorted(e) if pd.notna(e) else len(cal)
    M[a:b, col[t]] = True
M &= C.notna().to_numpy() | (Cf.notna().to_numpy() & False)
print("members priced per day: min", M[250:].sum(1).min(), "median", int(np.median(M[250:].sum(1))))

wk = cal.to_period("W")
wend = np.r_[wk[1:] != wk[:-1], False]
Cv, Ov = Cf.to_numpy(), O.to_numpy()
ret_cc = Cv[1:] / Cv[:-1] - 1                    # close to close, row i = day i+1
N = 20


def run(cost: float, fill: str = "close", n: int = N, best: bool = False) -> pd.Series:
    """Daily returns. fill='open': the book chosen at the week's last close is traded at the next session's open."""
    w = np.zeros(len(tick))
    pend = None
    out = np.zeros(len(cal))
    for i in range(6, len(cal)):
        if fill == "open" and pend is not None:
            # old book earns close->open, new book open->close
            co = np.where(np.isfinite(Ov[i]), Ov[i] / Cv[i - 1] - 1, 0.0)
            oc = np.where(np.isfinite(Ov[i]), Cv[i] / Ov[i] - 1, ret_cc[i - 1])
            r = np.nansum(w * co) + np.nansum(pend * oc) - cost * np.abs(pend - w).sum()
            w, pend = pend, None
        else:
            r = np.nansum(w * np.nan_to_num(ret_cc[i - 1]))
        if wend[i]:
            m = M[i] & np.isfinite(Cv[i - 5]) & (Cv[i - 5] > 0)
            r5 = np.where(m, Cv[i] / Cv[i - 5] - 1, np.nan)
            idx = np.argsort(np.where(np.isnan(r5), np.inf, -r5 if best else r5))[:n]
            nw = np.zeros(len(tick))
            nw[idx] = 1.0 / n
            if fill == "open":
                pend = nw
            else:
                r -= cost * np.abs(nw - w).sum()
                w = nw
        out[i] = r
    return pd.Series(out, index=cal)


def equal_weight(cost: float) -> pd.Series:
    w = np.zeros(len(tick))
    out = np.zeros(len(cal))
    for i in range(6, len(cal)):
        r = np.nansum(w * np.nan_to_num(ret_cc[i - 1]))
        if wend[i]:
            nw = M[i] / M[i].sum()
            drift = w * (1 + np.nan_to_num(ret_cc[i - 1]))
            drift = drift / drift.sum() if drift.sum() else drift
            r -= cost * np.abs(nw - drift).sum()
            w = nw
        out[i] = r
    return pd.Series(out, index=cal)


R = pd.DataFrame({
    "SPY": spy["adj_close"].reindex(cal).pct_change(),
    "ew_members": equal_weight(0.0005),
    "rev_1bp_close": run(0.0001),
    "rev_5bp_close": run(0.0005),
    "rev_10bp_close": run(0.0010),
    "rev_5bp_next_open": run(0.0005, "open"),
    "winners_5bp_close": run(0.0005, best=True),
}).loc["1998-01-05":]
rf = rf_daily(P, R.index)
R.to_csv(OUT / "reversal_long_daily.csv", float_format="%.6f")

res = {}
spans = {"1998-2026": ("1998", "2026"), "1998-2008": ("1998", "2008"), "2009-2018": ("2009", "2018"), "2019-2026": ("2019", "2026")}
for name, (a, b) in spans.items():
    X = R.loc[a:b]
    print(f"\n{name}   {'cagr':>6} {'vol':>6} {'sharpe':>6} {'maxdd':>7} {'beta':>5} {'alpha':>6} {'t':>5} | vs equal-weight members: alpha, t")
    for k in X:
        p, al = perf(X[k], rf), alpha(X[k], X["SPY"], rf)
        ae = alpha(X[k], X["ew_members"], rf)
        res[f"{k}|{name}"] = {**{x: float(v) for x, v in p.items()}, **{x: float(v) for x, v in al.items()},
                              "alpha_vs_ew": float(ae["alpha"]), "alpha_vs_ew_t": float(ae["alpha_t"]), "beta_vs_ew": float(ae["beta"])}
        print(f"{k:18} {100*p['cagr']:6.1f} {100*p['vol']:6.1f} {p['sharpe']:6.2f} {100*p['max_dd']:7.1f} {al['beta']:5.2f} "
              f"{100*al['alpha']:6.1f} {al['alpha_t']:5.1f} | {100*ae['alpha']:6.1f} {ae['alpha_t']:5.1f} (beta {ae['beta']:.2f})")
# turnover and what SPY at the reversal's beta would have made
b = res["rev_5bp_close|1998-2026"]["beta"]
bm = perf(beta_matched(R["SPY"], rf, b), rf)
res["spy_at_reversal_beta|1998-2026"] = {x: float(v) for x, v in bm.items()}
print(f"\nSPY held at beta {b:.2f}: cagr {100*bm['cagr']:.1f} maxdd {100*bm['max_dd']:.1f}")
yr = (1 + R[["SPY", "ew_members", "rev_5bp_close"]]).groupby(R.index.year).prod() - 1
print((100 * yr).round(1).to_string())
res["years"] = {str(y): {k: float(v) for k, v in row.items()} for y, row in yr.iterrows()}
json.dump(res, open(OUT / "reversal_long.json", "w"), indent=1)
