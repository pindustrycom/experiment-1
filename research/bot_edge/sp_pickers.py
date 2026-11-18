"""Stock pickers from GitHub run on the point-in-time S&P 500 (release papers-data), each against random picks.

  python sp_pickers.py PAPERS_DIR rows.csv

PAPERS_DIR holds prices/<TICKER>.csv (date,open,close,adj_close,volume) and pit/membership_intervals.parquet.
Pickers:
  - nickmccullum/algorithmic-trading-python (freeCodeCamp course) quantitative momentum: top 51 by 1-year return, and
    the 'HQM' version (mean percentile of 1y/6m/3m/1m returns). Rebalanced monthly (the notebook is one-shot).
    The notebook's HQM cell sorts without inplace, so it really takes the first 51 tickers alphabetically; the
    intended ranking is what is tested here.
  - myhhub/stock (instock) daily screens: turtle_trade (60-day closing high), enter (volume surge up day),
    keep_increasing (rising MA30, +20%), low_backtrace_increase (+60% in 60 days without a deep drop),
    breakthrough_platform (crosses MA60 on a volume surge after a flat base). Each hit is bought at the next open and
    held 20 trading days (assumed: the repo screens and leaves the exit to the user).
Null: the same number of picks on the same days drawn at random from that day's members (200 draws).
Costs 5bp per side. Benchmark: equal-weight point-in-time S&P 500 and SPY.
"""
import glob
import os
import sys

import numpy as np
import pandas as pd
from scipy.stats import norm

sys.path.insert(0, os.path.dirname(__file__))
import evalkit as ek  # noqa: E402

FEE = 0.0005
HOLD = 20
START = "2003-01-01"
N_RAND = 200


def load(papers):
    fr = {}
    for f in sorted(glob.glob(f"{papers}/prices/*.csv")):
        t = os.path.basename(f)[:-4]
        if t.startswith("^"):
            continue
        fr[t] = pd.read_csv(f, parse_dates=["date"]).set_index("date")
    cols = {k: pd.concat({t: v[k] for t, v in fr.items()}, axis=1).sort_index() for k in ("open", "close", "adj_close", "volume")}
    return cols


def members(papers, index, tickers):
    m = pd.read_parquet(f"{papers}/pit/membership_intervals.parquet")
    M = pd.DataFrame(False, index=index, columns=tickers)
    for t, s, e in m.itertuples(index=False):
        if t in M.columns:
            M.loc[(index >= s) & ((index <= e) if pd.notna(e) else True), t] = True
    return M


def portfolio(E, Ro, hold):
    """E[t, j] = 1 if stock j is picked at the close of day t: bought at the open of t+1, sold at the open of
    t+1+hold. Ro[t] = open(t) -> open(t+1) return. Equal weight across open positions; flat when none."""
    En = np.nan_to_num(E.astype(float))
    c = np.cumsum(En, axis=0)
    H = c.copy()
    H[hold:] -= c[:-hold]                       # positions opened in the last `hold` decision days
    H = np.vstack([np.zeros((1, H.shape[1])), H[:-1]])        # held over open(t) -> open(t+1) from the next day
    Hn = H.sum(axis=1)
    R = np.nan_to_num(Ro)
    g = np.where(Hn > 0, (H * R).sum(axis=1) / np.maximum(Hn, 1), 0.0)
    ent = np.vstack([np.zeros((1, En.shape[1])), En[:-1]]).sum(axis=1)
    ext = np.r_[np.zeros(hold + 1), En.sum(axis=1)[:-hold - 1]]
    cost = FEE * (ent + ext) / np.maximum(Hn, 1)
    return g - np.where(Hn > 0, cost, 0.0), g, (Hn > 0).astype(float)


def monthly_top(score, M, k=51):
    """Top-k by score among members at each month end, held for the next month (entries via a 1-month hold)."""
    s = score.where(M)
    me = s.index.to_series().groupby(s.index.to_period("M")).transform("max") == s.index.to_series()
    E = np.zeros(s.shape, bool)
    rk = s[me.to_numpy()].rank(axis=1, ascending=False, method="first")
    E[me.to_numpy()] = (rk <= k).to_numpy()
    return E


def random_like(E, M, rng):
    """Same count of picks on the same days, drawn from that day's members."""
    Er = np.zeros(E.shape, bool)
    cnt = E.sum(axis=1)
    Mv = M.to_numpy()
    for t in np.where(cnt > 0)[0]:
        pool = np.where(Mv[t])[0]
        Er[t, rng.choice(pool, min(cnt[t], len(pool)), replace=False)] = True
    return Er


def screens(C):
    o, c, v = C["open"], C["close"], C["volume"]
    pc = c.pct_change() * 100
    out = {}
    out["turtle_trade (60-day closing high)"] = c >= c.rolling(60).max()
    vma5 = v.rolling(5).mean()
    enter = (pc >= 2) & (c >= o) & (c * v >= 2e8) & (v / vma5.shift() >= 2)
    out["enter (volume surge up day)"] = enter
    ma30 = c.rolling(30).mean()
    out["keep_increasing (rising MA30, +20%)"] = ((ma30.shift(29) < ma30.shift(19)) & (ma30.shift(19) < ma30.shift(9)) &
                                                 (ma30.shift(9) < ma30) & (ma30 > 1.2 * ma30.shift(29)))
    intraday = (c - o) / o * 100
    two = pc + pc.shift()
    gap2 = (c - o.shift()) / o.shift() * 100
    bad = (pc < -7) | (intraday < -7) | (two < -10) | (gap2 < -10)
    out["low_backtrace_increase (+60% in 60d, no deep drop)"] = ((c / c.shift(59) - 1 >= 0.6) &
                                                                 (bad.astype(float).rolling(59).sum() == 0))
    ma60 = c.rolling(60).mean()
    brk = (o < ma60) & (ma60 <= c) & enter
    gap = (ma60 - c) / ma60
    base_ok = ((gap > -0.05) & (gap < 0.2)) | ma60.isna()
    # first breakthrough inside the last 60 days, with every earlier day in the window inside the base
    B, ok = brk.to_numpy(), base_ok.to_numpy()
    res = np.zeros(B.shape, bool)
    for j in range(B.shape[1]):
        ev = np.where(B[:, j])[0]
        if len(ev) == 0:
            continue
        bad_cum = np.cumsum(~ok[:, j])
        for t in range(B.shape[0]):
            lo = t - 59
            k = np.searchsorted(ev, max(lo, 0))
            if k >= len(ev) or ev[k] > t:
                continue
            b = ev[k]
            nbad = bad_cum[b - 1] - (bad_cum[lo - 1] if lo >= 1 else 0) if b > max(lo, 0) else 0
            res[t, j] = nbad == 0
    out["breakthrough_platform (MA60 cross on volume after a flat base)"] = pd.DataFrame(res, c.index, c.columns)
    return out


def score_row(name, src, E, M, Ro, idx, hold, bench_d, rng, note):
    net, gross, held = portfolio(E, Ro, hold)
    s_net, s_gross = pd.Series(net, idx), pd.Series(gross, idx)
    m = idx >= pd.Timestamp(START)
    dn = s_net[m]
    p = ek.perf(dn, 252)
    gp = ek.perf(s_gross[m], 252)
    hb = bench_d.reindex(dn.index).fillna(0.0)
    hp = ek.perf(hb, 252)
    b, t = ek.nw_t(dn.to_numpy(), np.column_stack([np.ones(len(dn)), hb.to_numpy()]))
    split = pd.Timestamp("2015-01-01")
    act = dn.mean()
    act_h = [dn[dn.index < split].mean(), dn[dn.index >= split].mean()]
    sims, sims_h = [], []
    for _ in range(N_RAND):
        rn, _, _ = portfolio(random_like(E, M, rng), Ro, hold)
        rs = pd.Series(rn, idx)[m]
        sims.append(rs.mean())
        sims_h.append([rs[rs.index < split].mean(), rs[rs.index >= split].mean()])
    sims, sims_h = np.array(sims), np.array(sims_h)
    p_emp = (1 + np.sum(sims >= act)) / (1 + N_RAND)
    z = (act - sims.mean()) / sims.std()
    pr = float(norm.sf(z))          # 200 draws floor the empirical p at 0.005, too coarse for Holm over ~1,000 rows
    zh = [(act_h[k] - sims_h[:, k].mean()) / sims_h[:, k].std() for k in (0, 1)]
    ph = [(1 + np.sum(sims_h[:, k] >= act_h[k])) / (1 + N_RAND) for k in (0, 1)]
    r = {"source": src, "strategy": name, "asset": "S&P 500 (point-in-time)", "tf": "1d", "native_tf": "1d", "fee": FEE,
         "cagr": p["cagr"], "sharpe": p["sharpe"], "max_dd": p["max_dd"], "vol": p["vol"], "gross_cagr": gp["cagr"],
         "gross_sharpe": gp["sharpe"], "hold_cagr": hp["cagr"], "hold_sharpe": hp["sharpe"], "hold_dd": hp["max_dd"],
         "alpha_ann": float(b[0] * 252), "alpha_t": float(t[0]), "beta": float(b[1]),
         "p_random": float(pr), "p_random_empirical": float(p_emp), "z_random": float(z), "p_random_h1": float(ph[0]), "z_random_h1": float(zh[0]),
         "p_random_h2": float(ph[1]), "z_random_h2": float(zh[1]),
         "trades": int(E[m].sum()), "exposure": float(held[m].mean()), "years": len(dn) / 252,
         "note": note + f" random null = same number of picks per day from that day's members ({N_RAND} draws); "
                        f"halves split 2015-01-01; benchmark = equal-weight point-in-time S&P 500; 5bp per side"}
    print(f"{name[:50]:50} n {r['trades']:6d} cagr {r['cagr']:+.1%} sh {r['sharpe']:.2f} EW {r['hold_cagr']:+.1%}/"
          f"{r['hold_sharpe']:.2f} p {pr:.3f} z {z:+.1f} h1 {zh[0]:+.1f} h2 {zh[1]:+.1f} a_t {t[0]:+.1f}", flush=True)
    return r


if __name__ == "__main__":
    papers, outp = sys.argv[1], sys.argv[2]
    C = load(papers)
    idx = C["close"].index
    idx = idx[idx >= pd.Timestamp("2001-01-01")]
    C = {k: v.loc[idx] for k, v in C.items()}
    adj = C["adj_close"] / C["close"]
    ao = C["open"] * adj
    Ro = (ao.shift(-1) / ao - 1).where(lambda x: x.abs() < 1.0).to_numpy()      # open(t) -> open(t+1)
    M = members(papers, idx, C["close"].columns) & C["close"].notna() & ao.shift(-1).notna()
    ew = pd.Series(np.nanmean(np.where(M.to_numpy(), np.nan_to_num(Ro), np.nan), axis=1), idx)
    rng = np.random.default_rng(0)
    rows = []
    ac = C["adj_close"]
    ret = {k: ac / ac.shift(n) - 1 for k, n in (("1y", 252), ("6m", 126), ("3m", 63), ("1m", 21))}
    pct = {k: v.where(M).rank(axis=1, pct=True) for k, v in ret.items()}
    hqm = sum(pct.values()) / 4
    for name, sc in (("quantitative momentum, top 51 by 1-year return", ret["1y"]),
                     ("HQM momentum (1y/6m/3m/1m percentiles), top 51", hqm)):
        E = monthly_top(sc, M)
        rows.append(score_row(name, "algorithmic-trading-python (freeCodeCamp)", E, M, Ro, idx, 21, ew, rng,
                              "monthly rebalance assumed;"))
    for name, sig in screens(C).items():
        E = (sig.fillna(False).astype(bool) & M).to_numpy()
        rows.append(score_row(name, "myhhub/stock (instock)", E, M, Ro, idx, HOLD, ew, rng,
                              "hold 20 trading days assumed; amount filter 200M kept in USD;"))
    df = pd.DataFrame(rows)
    df["ew_cagr"] = df["hold_cagr"]
    spy = pd.read_csv(f"{papers}/prices/SPY.csv", parse_dates=["date"]).set_index("date").adj_close.pct_change()
    df["spy_cagr"] = ek.perf(spy[spy.index >= pd.Timestamp(START)].dropna(), 252)["cagr"]
    df.to_csv(outp, index=False)
