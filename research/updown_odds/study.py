"""Which observable information tilts the odds of a stock going up or down away from 50%?

Every test is scored the same way: a hit rate against a coin flip.
- Stock signals (cross-section, point-in-time S&P 500 members): each period, the 5% of members the signal likes most
  (Sharma ch.3 tails). Hit = the stock beats the members' median return over the next period, so 50% is a coin flip
  and the market's own drift cancels out. Also reported: the same with beta-adjusted returns (Sharma ch.4), the
  plain up-rate of the picks vs all members, and the top-minus-bottom 5% spread after 5 bp a side (Sharma ch.9).
- Market conditions (SPY): P(SPY up next day / month | condition) against P(up | not condition).
t-stats are Newey-West; Holm across every test in the atlas. A delisted stock keeps its last price (no delisting
return), so losers are, if anything, flattered.

    python study.py PANEL.pkl SPY_1MIN_DIR OUT_DIR
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy import stats

TAIL = 0.05
COST = 0.0005  # per side, S&P 500 all-in (Sharma ch.9)
FIRST = "2001-01-01"
PERIODS = {"full": ("2001-01-01", "2026-12-31"), "2013+": ("2013-01-01", "2026-12-31"),
           "2019+": ("2019-01-01", "2026-12-31")}
LAGS = {"month": 3, "week": 4, "day": 5}
PER_YEAR = {"month": 12, "week": 52, "day": 252}


def nw_mean(x: np.ndarray, lags: int) -> tuple[float, float]:
    """Mean and Newey-West t of a series."""
    x = x[np.isfinite(x)]
    if len(x) < 24:
        return float("nan"), float("nan")
    fit = sm.OLS(x, np.ones(len(x))).fit(cov_type="HAC", cov_kwds={"maxlags": lags})
    return float(fit.params[0]), float(fit.tvalues[0])


def xs_rows(sig: pd.DataFrame, fwd: pd.DataFrame, fwd_beta: pd.DataFrame, univ: pd.DataFrame,
            binary: bool = False) -> pd.DataFrame:
    """Per-period stats of the top (and bottom) 5% by signal."""
    S, F, B, U = (x.to_numpy(float) for x in (sig, fwd, fwd_beta, univ.astype(float)))
    out, prev_top, prev_bot = [], None, None
    for i in range(len(S)):
        ok = (U[i] > 0) & np.isfinite(S[i]) & np.isfinite(F[i])
        if ok.sum() < 100:
            out.append([np.nan] * 9)
            prev_top = prev_bot = None
            continue
        s, f, b = S[i], F[i], B[i]
        if binary:
            top, bot = ok & (s > 0.5), ok & (s <= 0.5)
        else:
            lo, hi = np.nanquantile(s[ok], [TAIL, 1 - TAIL])
            top, bot = ok & (s >= hi), ok & (s <= lo)
        if top.sum() < 5 or bot.sum() < 5:
            out.append([np.nan] * 9)
            continue
        med = np.median(f[ok])
        okb = ok & np.isfinite(b)
        medb = np.median(b[okb]) if okb.sum() > 50 else np.nan
        tb = top & np.isfinite(b)
        turn_t = 1.0 if prev_top is None else 1 - (top & prev_top).sum() / top.sum()
        turn_b = 1.0 if prev_bot is None else 1 - (bot & prev_bot).sum() / bot.sum()
        prev_top, prev_bot = top, bot
        ls = f[top].mean() - f[bot].mean()
        out.append([(f[top] > med).mean(), (b[tb] > medb).mean() if tb.sum() >= 5 else np.nan,
                    (f[top] > 0).mean(), (f[ok] > 0).mean(), (f[bot] > med).mean(), ls,
                    ls - 2 * COST * (turn_t + turn_b), f[top].mean() - f[ok].mean() - 2 * COST * turn_t, turn_t])
    return pd.DataFrame(out, index=sig.index, columns=["hit", "hit_beta", "up_top", "up_all", "hit_bot", "ls",
                                                       "ls_net", "long_net", "turn"])


def summarize(name: str, family: str, freq: str, rows: pd.DataFrame, note: str) -> list[dict]:
    res = []
    for per, (a, b) in PERIODS.items():
        r = rows.loc[a:b].dropna(subset=["hit"])
        lags, k = LAGS[freq], PER_YEAR[freq]
        tilt, t = nw_mean(r["hit"].to_numpy() - 0.5, lags)
        tb, ttb = nw_mean(r["hit_beta"].to_numpy() - 0.5, lags)
        ls, tls = nw_mean(r["ls_net"].to_numpy(), lags)
        lo, tlo = nw_mean(r["long_net"].to_numpy(), lags)
        res.append(dict(test=name, family=family, freq=freq, period=per, n=len(r), p_up_pick=r["hit"].mean(),
                        tilt=tilt, t=t, beta_adj_tilt=tb, t_beta=ttb, up_rate_picks=r["up_top"].mean(),
                        up_rate_all=r["up_all"].mean(), hit_bottom=r["hit_bot"].mean(),
                        ls_gross_yr=r["ls"].mean() * k, ls_net_yr=ls * k, t_ls=tls, long_vs_all_net_yr=lo * k,
                        t_long=tlo, turnover=r["turn"].mean(), note=note))
    return res


def cond_test(y: pd.Series, c: pd.Series, lags: int) -> dict:
    """P(up | c) vs P(up | not c) by OLS of the up indicator on c, Newey-West."""
    d = pd.concat([y.rename("y"), c.rename("c").astype(float)], axis=1).dropna()
    fit = sm.OLS(d["y"], sm.add_constant(d["c"])).fit(cov_type="HAC", cov_kwds={"maxlags": lags})
    return dict(n=int(d["c"].sum()), p_up_pick=float(d.loc[d["c"] > 0, "y"].mean()),
                p_up_else=float(d.loc[d["c"] == 0, "y"].mean()), tilt=float(fit.params["c"]),
                t=float(fit.tvalues["c"]))


def sector_demean(x: pd.DataFrame, sec: pd.Series, mask: pd.DataFrame) -> pd.DataFrame:
    xm = x.where(mask)
    g = xm.T.groupby(sec).transform("mean").T
    return x - g


def main(panel: Path, minute_dir: Path, out: Path) -> None:
    out.mkdir(parents=True, exist_ok=True)
    P = pd.read_pickle(panel)
    adj, vol, close, opn, mem, sec = P["adj"], P["volume"], P["close"], P["open"], P["member"], P["sector"]
    r = adj.pct_change(fill_method=None)
    mkt = r.where(mem).mean(axis=1)  # equal-weight members
    adjf = adj.ffill()  # a stock that leaves keeps its last price
    # 252-day beta to the equal-weight market
    m_mean = mkt.rolling(252, min_periods=200).mean()
    cov = (r.mul(mkt, axis=0)).rolling(252, min_periods=200).mean() - r.rolling(252, min_periods=200).mean().mul(m_mean, axis=0)
    beta = cov.div(mkt.rolling(252, min_periods=200).var(ddof=0), axis=0)
    dvol = (close * vol).replace(0, np.nan)
    spike = (vol / vol.rolling(50, min_periods=40).median().shift(1) >= 3).astype(float).where(vol.notna())
    divs = P.get("dividends")

    days = adj.index
    month_ends = pd.Series(days, index=days).groupby(days.to_period("M")).max().to_numpy()
    week_ends = pd.Series(days, index=days).groupby(days.to_period("W-FRI")).max().to_numpy()
    rows_all: list[dict] = []
    keep: dict[str, pd.DataFrame] = {}

    def run(name, family, freq, sig, dates, note, binary=False):
        nxt = np.append(dates[1:], pd.NaT)
        d, n = pd.DatetimeIndex(dates[:-1]), pd.DatetimeIndex(nxt[:-1])
        fwd = pd.DataFrame(adjf.loc[n].to_numpy() / adjf.loc[d].to_numpy() - 1, index=d, columns=adj.columns)
        mf = pd.Series(np.exp(np.log1p(mkt).cumsum()).loc[n].to_numpy()
                       / np.exp(np.log1p(mkt).cumsum()).loc[d].to_numpy() - 1, index=d)
        fb = fwd - beta.loc[d].mul(mf, axis=0)
        rows = xs_rows(sig.reindex(d), fwd, fb, mem.loc[d] & adj.loc[d].notna(), binary)
        keep[name] = rows
        rows_all.extend(summarize(name, family, freq, rows, note))
        s = rows_all[-3]
        print(f"{name:22s} P(beat median) {s['p_up_pick']:.3f} t {s['t']:5.1f} | beta-adj t {s['t_beta']:5.1f} "
              f"| L/S net {s['ls_net_yr']:+.1%}/yr t {s['t_ls']:4.1f} | 2019+ {rows_all[-1]['p_up_pick']:.3f}", flush=True)

    ME = month_ends
    # ---- monthly stock signals (hold to next month end)
    run("momentum_12_1", "price", "month", adj.shift(21) / adj.shift(252) - 1, ME, "Jegadeesh-Titman; tested before as a pot")
    run("high_52w", "price", "month", adj / adj.rolling(252, min_periods=200).max(), ME, "George-Hwang 52-week-high proximity")
    run("reversal_1m", "price", "month", -(adj / adj.shift(21) - 1), ME, "Jegadeesh 1990 one-month reversal")
    ret6 = adj / adj.shift(126) - 1
    run("industry_mom_6m", "price", "month", (ret6 - sector_demean(ret6, sec, mem)), ME, "Moskowitz-Grinblatt (GICS sector, EW members)")
    run("low_vol_3m", "risk", "month", -r.rolling(63, min_periods=50).std(), ME, "Ang et al. / Baker et al. low-volatility anomaly")
    run("low_beta_1y", "risk", "month", -beta, ME, "Frazzini-Pedersen betting against beta")
    run("low_max_1m", "risk", "month", -r.rolling(21, min_periods=18).max(), ME, "Bali-Cakici-Whitelaw lottery (MAX) effect")
    run("illiquidity_1m", "risk", "month", (r.abs() / dvol).rolling(21, min_periods=15).mean(), ME, "Amihud illiquidity premium")
    # Heston-Sadka: the coming calendar month's return in each of the last 10 years
    mr = adj.loc[ME].pct_change(fill_method=None)
    stack = np.stack([mr.shift(k).to_numpy() for k in range(11, 120, 12)])
    cnt = np.isfinite(stack).sum(0)
    seas = pd.DataFrame(np.where(cnt >= 3, np.nanmean(np.where(np.isfinite(stack), stack, np.nan), 0), np.nan),
                        index=mr.index, columns=mr.columns)
    run("seasonality_same_month", "calendar", "month", seas, ME, "Heston-Sadka: same calendar month in past 10 years")
    # frog in the pan: momentum that came in small steps (Da, Gurun, Warachka)
    pos = (r > 0).astype(float).where(r.notna()).rolling(231, min_periods=200).mean().shift(21)
    neg = (r < 0).astype(float).where(r.notna()).rolling(231, min_periods=200).mean().shift(21)
    mom = adj.shift(21) / adj.shift(252) - 1
    info_d = np.sign(mom) * (neg - pos)
    fip = mom.rank(axis=1, pct=True) + (-info_d).rank(axis=1, pct=True)
    run("frog_in_pan_mom", "price", "month", fip.where(mom.notna() & info_d.notna()), ME, "Da-Gurun-Warachka smooth momentum")
    if divs is not None:
        dm = divs.gt(0).astype(float).groupby(days.to_period("M")).max()
        dm.index = pd.DatetimeIndex(ME)
        run("dividend_month", "calendar", "month", dm.shift(11), ME, "Hartzmark-Solomon: paid a dividend in this month last year", binary=True)
    sm_ = spike.groupby(days.to_period("M")).max()
    sm_.index = pd.DatetimeIndex(ME)
    run("earnings_month_proxy", "calendar", "month", sm_.shift(11), ME, "Frazzini-Lamont: volume spike (proxy for earnings) in this month last year", binary=True)

    # ---- weekly stock signals (Friday close to Friday close)
    WE = week_ends
    r1w = adj / adj.shift(5) - 1
    run("reversal_1w", "price", "week", -r1w, WE, "Lehmann / Lo-MacKinlay; our reversal pot")
    run("reversal_1w_in_sector", "price", "week", -sector_demean(r1w, sec, mem), WE, "Da-Liu-Schaumburg: industry-adjusted reversal")
    vshock = np.log(vol.rolling(5).sum() / (vol.shift(5).rolling(50, min_periods=40).sum() / 10))
    run("volume_shock_1w", "flow", "week", vshock, WE, "Gervais-Kaniel-Mingelgrin high-volume return premium")
    run("earnings_week_proxy", "calendar", "week", spike.rolling(9, min_periods=5).max().shift(245), WE,
        "volume spike 1 year ago lands in the coming week", binary=True)
    gap_ret = (adj / adj.shift(1)) / (close / close.shift(1)) * (opn / close.shift(1)) - 1  # dividend-adjusted overnight
    intra = close / opn - 1
    run("overnight_vs_intraday_1m", "flow", "week", gap_ret.rolling(21).sum() - intra.rolling(21).sum(), WE,
        "Lou-Polk-Skouras tug of war: who buys overnight (retail) vs intraday (institutions)")

    # ---- daily stock signals (close to next close)
    D = days[days >= "2000-06-01"].to_numpy()
    run("reversal_1d", "price", "day", -r, D, "one-day reversal")
    run("reversal_1d_in_sector", "price", "day", -sector_demean(r, sec, mem), D, "one-day sector-relative reversal")
    run("gap_fade_1d", "flow", "day", -gap_ret, D, "fade today's overnight gap")

    # ---- market conditions (SPY)
    spy = P["spy"]
    sr = spy["adj_close"].pct_change()
    y = (sr.shift(-1) > 0).astype(float).where(sr.shift(-1).notna())
    idx = spy.index
    nxt = pd.Series(idx[1:].append(pd.DatetimeIndex([pd.NaT])), index=idx)
    nxt2 = nxt.shift(-1)
    bd = pd.Series(np.busday_count(nxt.dropna().values.astype("datetime64[D]"),
                                   nxt2.dropna().reindex(nxt.dropna().index).fillna(nxt.dropna()).values.astype("datetime64[D]")),
                   index=nxt.dropna().index)
    mon = idx.to_period("M")
    pos_in_m = pd.Series(1, index=idx).groupby(mon).cumcount() + 1
    left_in_m = pd.Series(1, index=idx).groupby(mon).cumcount(ascending=False)
    tom_day = (pos_in_m <= 3) | (left_in_m == 0)
    vix = P["vix"].reindex(idx).ffill()
    above50 = (adj > adj.rolling(50, min_periods=40).mean()).where(mem & adj.notna()).mean(axis=1).reindex(idx)
    conds = {
        "turn_of_month": (tom_day.shift(-1).fillna(False), "Ariel / Lakonishok-Smidt: last day + first 3 days"),
        "pre_holiday": ((bd > 1).reindex(idx).fillna(False), "Ariel 1990: day before a market holiday"),
        "monday": (pd.Series(nxt.dt.dayofweek == 0, index=idx), "weekend effect"),
        "after_2pct_drop": (sr < -0.02, "big down day yesterday"),
        "after_up_day": (sr > 0, "yesterday's sign"),
        "above_200dma": (spy["close"] > spy["close"].rolling(200).mean(), "trend regime"),
        "vix_above_30": (vix > 30, "fear regime"),
        "vix_jump_20pct": (vix.pct_change() > 0.2, "fear spike"),
        "breadth_below_20pct": (above50 < 0.2, "under 20% of members above their 50-day average"),
    }
    for per, (a, b) in PERIODS.items():
        for k, (c, note) in conds.items():
            d = cond_test(y.loc[a:b], c.loc[a:b], 5)
            rows_all.append(dict(test=k, family="market", freq="day", period=per, note=note, **d))
    # monthly market conditions
    mspy = spy["adj_close"].resample("ME").last()
    mret = mspy.pct_change()
    ym = (mret.shift(-1) > 0).astype(float).where(mret.shift(-1).notna())
    mconds = {"halloween_nov_apr": (pd.Series((mret.index.month.isin([10, 11, 12, 1, 2, 3])), index=mret.index),
                                    "Bouman-Jacobsen: next month is Nov-Apr"),
              "above_10m_sma": (mspy > mspy.rolling(10).mean(), "Faber: SPY above its 10-month average")}
    for per, (a, b) in PERIODS.items():
        for k, (c, note) in mconds.items():
            d = cond_test(ym.loc[a:b], c.loc[a:b], 3)
            rows_all.append(dict(test=k, family="market", freq="month", period=per, note=note, **d))
    # overnight vs intraday up-rate (SPY)
    on = (spy["open"] / spy["close"].shift(1) - 1)
    idr = spy["close"] / spy["open"] - 1
    for per, (a, b) in PERIODS.items():
        o, i_ = on.loc[a:b].dropna(), idr.loc[a:b].dropna()
        diff, t = nw_mean(((o > 0).astype(float) - (i_ > 0).astype(float).reindex(o.index)).to_numpy(), 5)
        rows_all.append(dict(test="overnight_vs_intraday_spy", family="market", freq="day", period=per, n=len(o),
                             p_up_pick=float((o > 0).mean()), p_up_else=float((i_ > 0).mean()), tilt=diff, t=t,
                             note="share of overnight moves up vs share of open-to-close moves up"))
    # intraday momentum (Gao-Han-Li-Zhou): first half hour (incl. overnight) and 12th half hour -> last half hour
    if minute_dir.exists():
        rows_all.extend(intraday(minute_dir))

    df = pd.DataFrame(rows_all)
    full = df[df["period"] == "full"].copy()
    full["p"] = 2 * stats.norm.sf(full["t"].abs())
    m = len(full)
    order = full["p"].sort_values().index
    holm = np.maximum.accumulate([min(1, (m - i) * full.loc[j, "p"]) for i, j in enumerate(order)])
    full.loc[order, "holm_p"] = holm
    df = df.merge(full[["test", "holm_p"]], on="test", how="left")
    df.to_csv(out / "atlas.csv", index=False)
    pd.concat({k: v for k, v in keep.items()}, names=["test", "date"]).to_csv(out / "xs_periods.csv.gz")
    print(f"\n{m} tests, Holm-significant (p<0.05): {sorted(full.loc[full['holm_p'] < 0.05, 'test'])}")


def intraday(minute_dir: Path) -> list[dict]:
    res = []
    for sym in ("SPY", "QQQ"):
        files = sorted((minute_dir / sym).glob("*.csv"))
        if not files:
            continue
        b = pd.concat(pd.read_csv(f, parse_dates=["date"]) for f in files).drop_duplicates("date").set_index("date")
        b.index = b.index.tz_convert("America/New_York")
        b = b.between_time("09:30", "15:59")
        day = b.index.normalize()
        g = b.groupby(day)
        first_open, close = g["open"].first(), g["close"].last()

        def px_at(hm):
            s = b.between_time(hm, hm)["open"]
            s.index = s.index.normalize()
            return s
        p10, p1500, p1530 = px_at("10:00"), px_at("15:00"), px_at("15:30")
        prev_close = close.shift(1)
        df = pd.DataFrame({"r1": p10 / prev_close - 1, "r12": p1530 / p1500 - 1, "rl": close / p1530 - 1}).dropna()
        df = df[df.index.dayofweek < 5]
        yl = (df["rl"] > 0).astype(float)
        for nm, c, note in [("intraday_mom_first30_" + sym, df["r1"] > 0, "Gao-Han-Li-Zhou: first half hour (incl. overnight) up -> last half hour up"),
                            ("intraday_mom_12th30_" + sym, df["r12"] > 0, "12th half hour up -> last half hour up")]:
            d = cond_test(yl, c, 5)
            # also the sign-agreement hit rate (go long or short the last half hour with the signal)
            hit = ((df["rl"] > 0) == c).astype(float)
            pnl = np.where(c, df["rl"], -df["rl"]) - 2 * 0.00003  # ~0.3 bp a side for SPY
            # the minute data starts in 2019, so "full" is 2019-2026 here
            res.append(dict(test=nm, family="intraday", freq="day", period="full", note=note,
                            p_hit_sign=float(hit.mean()), yr_net=float(np.mean(pnl) * 252),
                            t_pnl=nw_mean(pnl, 5)[1], **d))
    return res


if __name__ == "__main__":
    main(Path(sys.argv[1]), Path(sys.argv[2]), Path(sys.argv[3]))
