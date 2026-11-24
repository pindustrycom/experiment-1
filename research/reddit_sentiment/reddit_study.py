"""Does Reddit chatter about a stock predict its next-day or next-week return, beyond what its price already says?

    python reddit_study.py PANEL.pkl POSTS_DIR OUT_DIR

POSTS_DIR holds posts_*.jsonl.gz from collect_reddit.py. Steps:
1. Tickers: a post names a stock if its title has the ticker in capitals as a word ("TSLA") or as $ticker ("$tsla");
   tickers that are everyday words (ALL, NOW, LOW...) count only as $ticker. One row per (post, ticker).
2. Time: a post counts for trading day d if made before 15:30 New York on d (or after 15:30 on the previous trading
   day); we trade at d's close. Archived scores / comment counts are never used (recorded after the post).
3. Signals per stock-day, over all point-in-time S&P 500 members (unmentioned = 0 posts):
   att1   = log(1+posts today) - mean of that over the previous 20 days (attention spike)
   att5   = log(1+posts last 5d) - log(1+average 5-day posts over the previous 60 days)
   wsb5   = att5 counting r/wallstreetbets only
   sent5  = (bullish - bearish words)/(bullish + bearish + 1) in titles of the last 5 days (WSB word list)
   sent20 = the same over 20 days ("slowed-down" sentiment, Sharma ch.11)
   dd20   = log(1+ "DD"-flaired research posts in the last 20 days) (Bradley et al. 2024 found DD posts informative)
   Controls: today's, 5-day and 21-day return, 21-day volatility, log dollar volume, dollar volume vs its 20-day mean,
   and the stock's usual Reddit level (posts per day over the 60 days before the last 5), so a signal is not just
   'popular on Reddit' (the mega caps).
4. Tests (train 2018-06..2022-12, test 2023-01..2026-11, nothing tuned on test):
   a. Fama-MacBeth: each day regress next-day (and next-5-day) market-adjusted return on cross-sectional z-scores of
      each signal plus all controls; Newey-West t of the daily coefficients.
   b. Trading rules, direction fixed by the train-period sign: daily top-2% attention spikes (long or short, 1 day),
      weekly top-vs-bottom quintile of sent5 / sent20 among stocks with >=5 posts, long-only versions vs SPY,
      5 bp per side. Holm across the test-period p-values.
"""
from __future__ import annotations

import glob
import gzip
import json
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.api as sm

TRAIN_END, TEST_START = "2022-12-31", "2023-01-01"
START = "2018-06-01"
COST = 0.0005
BULL = set("""moon mooning moons rocket rockets calls call bull bulls bullish buy buying bought long longs undervalued squeeze
squeezing tendies yolo breakout rip ripping soar soaring surge surging rally rallying beat beats upgrade upgraded green ath lambo
diamond hodl gains printing cheap oversold rebound bounce bouncing upside runup winner winners""".split())
BEAR = set("""puts put bear bears bearish short shorts shorting shorted sell selling sold overvalued crash crashing crashed dump
dumping drill drilling tank tanking tanked plunge plunging drop dropping red bubble fraud bankrupt bankruptcy miss misses missed
downgrade downgraded loss losses rug bagholder bagholders bagholding dead worst fall falling overbought scam downside collapse
collapsing""".split())
EMOJI_BULL = ("🚀", "💎", "🌙", "📈", "🐂")
EMOJI_BEAR = ("📉", "🌈🐻", "🐻", "🩸")


# tickers that are also everyday words or common all-caps words: counted only when written as $TICKER
WORDS = set("""ALL ARE BALL BEN CAT COST DAY DOC DOW ECHO FAST FIX FLEX GAP GEN HAS HUM ICE KEY KEYS KIM LOW NOW PEG POOL
SEE TAP TECH TEL TRIP WAT WELL AMP APP ARES BRO COO EXE LEG LEN MET MOS PSA UPS USB VAR RTX DTE TXT PPL ROL INFO MAT MAR MAS
DASH NVR BIO AWK COP COR DIS FANG HON MAC MMM CASH EXPE IEX CEO CFO IPO ETF USA FED SEC GDP CPI EPS ATH YOLO FOMO IMO TLDR
EOD AMA LOL WTF FYI EDIT UPDATE BUY SELL HOLD PUT PUTS CALL CALLS LONG SHORT BIG NEW ONE TWO OUT WAY RUN""".split())


def load_posts(d: Path, universe: set[str]) -> pd.DataFrame:
    """One row per (post, ticker named in its title). Plain capital tokens of 3+ letters count unless they are in
    WORDS; $TICKER counts at any length."""
    rows, n_all = [], 0
    caps = re.compile(r"(?<![A-Za-z$])([A-Z]{3,5})(?![A-Za-z])")
    dollar = re.compile(r"\$([A-Za-z]{1,5})(?![A-Za-z])")
    for f in sorted(glob.glob(str(d / "posts_*.jsonl.gz"))):
        with gzip.open(f, "rt") as fh:
            for line in fh:
                try:
                    p = json.loads(line)
                except json.JSONDecodeError:
                    continue
                n_all += 1
                ti = p.get("title") or ""
                tk = {t for t in caps.findall(ti) if t not in WORDS} | {t.upper() for t in dollar.findall(ti)}
                for t in tk & universe:
                    rows.append((t, p["id"], p["ts"], p["sub"], ti, p.get("flair") or ""))
    df = pd.DataFrame(rows, columns=["tk", "id", "ts", "sub", "title", "flair"]).drop_duplicates(["id", "tk"])
    df.attrs["n_all"] = n_all
    return df


def tone(title: str) -> tuple[int, int]:
    w = re.findall(r"[a-z]+", title.lower())
    b = sum(x in BULL for x in w) + sum(title.count(e) for e in EMOJI_BULL)
    s = sum(x in BEAR for x in w) + sum(title.count(e) for e in EMOJI_BEAR)
    return b, s


def daily(df: pd.DataFrame, dates: pd.DatetimeIndex) -> dict[str, pd.DataFrame]:
    t = pd.to_datetime(df["ts"], unit="s", utc=True).dt.tz_convert("America/New_York")
    day = t.dt.tz_localize(None).dt.normalize()
    late = (t.dt.hour * 60 + t.dt.minute) >= 15 * 60 + 30
    cand = day + pd.to_timedelta(late.astype(int), unit="D")
    pos = dates.searchsorted(cand.to_numpy())  # first trading day on or after
    ok = pos < len(dates)
    df = df.loc[ok].copy()
    df["d"] = dates[pos[ok]]
    bs = [tone(x) for x in df["title"]]
    df["bull"], df["bear"] = [b for b, _ in bs], [s for _, s in bs]
    df["wsb"] = (df["sub"] == "wallstreetbets").astype(int)
    df["dd"] = df["flair"].str.contains(r"^\s*DD\b|due diligence", case=False, regex=True).astype(int)
    df["one"] = 1
    g = df.groupby(["d", "tk"])[["one", "wsb", "bull", "bear", "dd"]].sum()
    out = {}
    for c in ["one", "wsb", "bull", "bear", "dd"]:
        out[c] = g[c].unstack().reindex(dates).fillna(0)
    return out


def zs(x: pd.DataFrame, mask: pd.DataFrame) -> pd.DataFrame:
    x = x.where(mask)
    return x.sub(x.mean(axis=1), axis=0).div(x.std(axis=1), axis=0).clip(-4, 4)


def fama_macbeth(y: pd.DataFrame, X: dict[str, pd.DataFrame], dates, lags: int) -> dict:
    coefs = []
    for d in dates:
        yy = y.loc[d]
        M = pd.DataFrame({k: v.loc[d] for k, v in X.items()})
        M["y"] = yy
        M = M.dropna()
        if len(M) < 100:
            continue
        b = np.linalg.lstsq(np.column_stack([np.ones(len(M)), M.drop(columns="y").to_numpy()]), M["y"].to_numpy(), rcond=None)[0]
        coefs.append(pd.Series(b[1:], index=list(X)))
    C = pd.DataFrame(coefs)
    res = {}
    for k in C.columns:
        m = sm.OLS(C[k].to_numpy(), np.ones(len(C))).fit(cov_type="HAC", cov_kwds={"maxlags": lags})
        res[k] = {"coef_bp": round(1e4 * float(m.params[0]), 2), "t": round(float(m.tvalues[0]), 2)}
    return res


def perf(r: pd.Series, spy: pd.Series | None = None, per_year: int = 252) -> dict:
    r = r.dropna()
    if len(r) < 20:
        return {"n": int(len(r))}
    ann = (1 + r).prod() ** (per_year / len(r)) - 1
    out = {"n": int(len(r)), "ann_%": round(100 * ann, 1), "sharpe": round(float(r.mean() / r.std() * np.sqrt(per_year)), 2),
           "t_mean": round(float(r.mean() / r.std() * np.sqrt(len(r))), 2)}
    if spy is not None:
        s = spy.reindex(r.index)
        out["spy_ann_%"] = round(100 * ((1 + s).prod() ** (per_year / len(s)) - 1), 1)
        m = sm.OLS(r.to_numpy(), sm.add_constant(s.to_numpy())).fit(cov_type="HAC", cov_kwds={"maxlags": 5})
        out["alpha_ann_%"] = round(100 * per_year * float(m.params[0]), 1)
        out["alpha_t"] = round(float(m.tvalues[0]), 2)
        out["beta"] = round(float(m.params[1]), 2)
    return out


def holm(p: dict[str, float]) -> dict[str, float]:
    items = sorted(p.items(), key=lambda kv: kv[1])
    m, run, out = len(items), 0.0, {}
    for i, (k, v) in enumerate(items):
        run = max(run, min(1.0, (m - i) * v))
        out[k] = round(run, 3)
    return out


def main(panel: str, posts_dir: str, out: Path) -> None:
    from scipy import stats
    out.mkdir(parents=True, exist_ok=True)
    P = pd.read_pickle(panel)
    close, dvol = P["close"].loc["2017-06-01":], P["dvol"].loc["2017-06-01":]
    mem = P["member"].loc[close.index]
    dates = close.index
    base = {re.sub(r"-\d{6}$", "", c): c for c in close.columns}
    universe = set(base) | {t.replace("-B", "") for t in base if t.endswith("-B")}
    df = load_posts(Path(posts_dir), universe)
    res = {"posts_all": int(df.attrs["n_all"]), "post_ticker_mentions": int(len(df))}
    df["tk"] = df["tk"].map(lambda t: base.get(t, base.get(t + "-B", t)))
    df = df[df["tk"].isin(close.columns)]
    res["posts_by_sub"] = df["sub"].value_counts().to_dict()
    res["posts_by_year"] = pd.to_datetime(df["ts"], unit="s").dt.year.value_counts().sort_index().to_dict()
    res["top_tickers"] = df["tk"].value_counts().head(25).to_dict()
    # coverage: the collector fills each year in time order, subreddit by subreddit (state_<year>.json = how far it got).
    # Use only subreddits that cover most of r/wallstreetbets' span, and only days with 65 covered days behind them.
    cov = {}
    for f in glob.glob(str(Path(posts_dir) / "state_*.json")):
        key = Path(f).stem.split("_")[1]
        st = json.loads(Path(f).read_text())
        y0 = pd.Timestamp(f"{key}-01" if "-" in key else ("2018-06-01" if key == "2018" else f"{key}-01-01"))
        for sub, t in st.items():
            m = cov.setdefault(sub, pd.Series(False, index=dates))
            m[(dates >= y0) & (dates < pd.Timestamp(int(t), unit="s").normalize())] = True
    if cov:
        wsb = cov.get("wallstreetbets", pd.Series(False, index=dates))
        subs = [k for k, m in cov.items() if (m & wsb).sum() >= 0.9 * wsb.sum()]
        df = df[df["sub"].isin(subs)]
        covered = pd.concat([cov[k] for k in subs], axis=1).all(axis=1)
        covered = covered.astype(int).rolling(65).min().fillna(0).astype(bool)
        res["coverage_subs_used"] = subs
        res["coverage_days_usable"] = int(covered.sum())
    else:
        covered = pd.Series(True, index=dates)
    D = daily(df, dates)
    D = {k: v.reindex(columns=close.columns).fillna(0) for k, v in D.items()}

    r = close.pct_change(fill_method=None)
    spy = P["etf"]["SPY"].reindex(dates).pct_change()
    ar = r.sub(spy, axis=0)  # market-adjusted (beta 1); the FM regressions are cross-sectional so beta matters little
    y1 = ar.shift(-1)
    y5 = (np.log1p(r).rolling(5).sum().shift(-5)) - np.log1p(spy).rolling(5).sum().shift(-5).to_numpy()[:, None]
    l1 = np.log1p(D["one"])
    S = {}
    S["att1"] = l1 - l1.shift(1).rolling(20).mean()
    n5 = D["one"].rolling(5).sum()
    S["att5"] = np.log1p(n5) - np.log1p(n5.shift(5).rolling(60).mean())
    w5 = D["wsb"].rolling(5).sum()
    S["wsb5"] = np.log1p(w5) - np.log1p(w5.shift(5).rolling(60).mean())
    b5, s5 = D["bull"].rolling(5).sum(), D["bear"].rolling(5).sum()
    S["sent5"] = (b5 - s5) / (b5 + s5 + 1)
    b20, s20 = D["bull"].rolling(20).sum(), D["bear"].rolling(20).sum()
    S["sent20"] = (b20 - s20) / (b20 + s20 + 1)
    S["dd20"] = np.log1p(D["dd"].rolling(20).sum())
    Cn = {"ret1": r, "ret5": close.pct_change(5, fill_method=None), "ret21": close.pct_change(21, fill_method=None),
          "vol21": r.rolling(21).std(), "ldvol": np.log(dvol.rolling(20).mean()), "adv": np.log(dvol / dvol.rolling(20).mean()),
          "lvl20": np.log1p(D["one"].shift(5).rolling(60).mean())}  # how talked-about the stock usually is
    ok = mem & close.notna() & y1.notna() & pd.DataFrame({c: covered for c in close.columns}, index=dates)
    mem = mem & ok
    Z = {k: zs(v, ok) for k, v in {**S, **Cn}.items()}
    # sentiment only means something where people talk: zero it where the stock had <3 posts in the window
    Z["sent5"] = Z["sent5"].where(n5 >= 3, 0).where(ok)
    Z["sent20"] = Z["sent20"].where(D["one"].rolling(20).sum() >= 3, 0).where(ok)
    tr = dates[(dates >= pd.Timestamp(START) + pd.Timedelta(days=100)) & (dates <= TRAIN_END)]
    te = dates[(dates >= TEST_START) & (dates <= dates[-6])]
    res["coverage"] = {"mean_stocks_with_post_per_day": round(float((D["one"].where(mem) > 0).sum(axis=1).loc[te].mean()), 1),
                       "mean_posts_per_day_test": round(float(D["one"].sum(axis=1).loc[te].mean()), 1),
                       "mean_posts_per_day_train": round(float(D["one"].sum(axis=1).loc[tr].mean()), 1)}

    fm = {}
    ctrl = {k: Z[k] for k in Cn}
    for sig in S:
        for name, yy, lags in [("next_day", y1, 5), ("next_5d", y5, 10)]:
            for per, ds in [("train", tr), ("test", te)]:
                ds2 = ds if name == "next_day" else ds[::5]
                fm[f"{sig}|{name}|{per}|alone"] = fama_macbeth(yy, {sig: Z[sig]}, ds2, lags)[sig]
                fm[f"{sig}|{name}|{per}|with_controls"] = fama_macbeth(yy, {sig: Z[sig], **ctrl}, ds2, lags)[sig]
    res["fama_macbeth"] = fm

    # trading rules
    rules, pvals = {}, {}
    fwd1 = r.shift(-1)
    fwd5 = close.shift(-5) / close - 1

    def spike_rule(sig: str, per) -> pd.Series:
        x = S[sig].where(mem & (D["one"] >= 3))
        thr = x.rank(axis=1, pct=True)
        pick = (thr >= 0.98) & x.notna()
        port = fwd1.where(pick).mean(axis=1)
        return port.loc[per]

    for sig in ["att1", "wsb5"]:
        trr = spike_rule(sig, tr).dropna()
        sign = 1 if (trr - spy.shift(-1).reindex(trr.index)).mean() > 0 else -1
        ter = spike_rule(sig, te).dropna()
        ex = sign * (ter - spy.shift(-1).reindex(ter.index)) - 2 * COST  # hedged with SPY, traded daily
        rules[f"{sig} spike top2% (sign {sign:+d} from train), 1 day, SPY-hedged, net"] = perf(ex)
        pvals[f"{sig}_spike"] = float(2 * stats.norm.sf(abs(ex.mean() / ex.std() * np.sqrt(len(ex)))))
        if sign > 0:
            lo = ter - 2 * COST
            rules[f"{sig} spike long-only, net, vs SPY"] = perf(lo, spy.shift(-1))

    fri = dates[dates.weekday == 4]
    for sig in ["sent5", "sent20", "att5", "dd20"]:
        x = S[sig].where(mem & (D["one"].rolling(20).sum() >= 5))
        q = x.rank(axis=1, pct=True)
        top, bot = fwd5.where(q >= 0.8).mean(axis=1), fwd5.where(q <= 0.2).mean(axis=1)
        ls = (top - bot)
        trr, ter = ls.reindex(fri).loc[tr[0]:TRAIN_END].dropna(), ls.reindex(fri).loc[TEST_START:].dropna()
        sign = 1 if trr.mean() > 0 else -1
        net = sign * ter - 4 * COST  # both legs turned over weekly (upper bound on cost)
        rules[f"{sig} weekly top-bottom quintile (sign {sign:+d} from train), net"] = {**perf(net, per_year=52),
                                                                                     "train_gross": perf(sign * trr, per_year=52)}
        pvals[f"{sig}_ls"] = float(2 * stats.norm.sf(abs(net.mean() / net.std() * np.sqrt(len(net)))))
        lo = (top if sign > 0 else bot).reindex(fri).loc[TEST_START:].dropna() - 2 * COST
        spy5 = (P["etf"]["SPY"].reindex(dates).shift(-5) / P["etf"]["SPY"].reindex(dates) - 1).reindex(lo.index)
        rules[f"{sig} long-only {'top' if sign > 0 else 'bottom'} quintile weekly, net, vs SPY"] = perf(lo, spy5, per_year=52)
    res["rules_test_2023plus"] = rules
    res["holm_p_rules"] = holm(pvals)
    fm_p = {k: float(2 * stats.norm.sf(abs(v["t"]))) for k, v in fm.items() if "|test|with_controls" in k}
    res["holm_p_fm_test_with_controls"] = holm(fm_p)
    (out / "reddit.json").write_text(json.dumps(res, indent=1, default=str))
    print(json.dumps(res, indent=1, default=str))


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2], Path(sys.argv[3]))
