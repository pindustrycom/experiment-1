"""Weekly walk-forward of the vol-and-curve market map, per strategy, then a 500 CAD test on 2026.

Market state at each close, from data known at that close:
  vix    = VIX level (what options say volatility will be, next 30 days)
  spread = VIX minus SPY's realized 21-day volatility (the curve between implied and realized vol:
           positive = options price more fear than the market is showing; negative = the market is moving more
           than options expected, which is what stress looks like)
Every Friday the K-Means map (3 groups on standardized log VIX and spread, all history since 1998 up to that Friday)
is refitted, every past day is relabelled with it, and each strategy's record per group from 2020 up to that Friday
is recomputed. Next week's days are labelled with that week's map only (no peeking).
"""
from pathlib import Path
import json

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm

S = Path(__file__).resolve().parents[1]
OUT = Path("/mnt/project-files/research/regime_walkforward/")
OUT.mkdir(parents=True, exist_ok=True)
NAMES = ("Calm", "Normal", "Stressed")
TRAIN0, TRAIN1, TEST0, TEST1 = "2020-01-01", "2025-12-31", "2026-01-01", "2026-10-01"
CAP, MIN_DAYS, T_MIN, SWITCH = 0.4, 40, 1.0, 0.0002
LABEL = {"trend": "Trend (momentum)", "reversal": "Weekly reversal", "best_ideas": "Best ideas (13F copy)",
         "conviction": "Conviction (13F copy)", "pelosi": "Pelosi copy", "exp1": "Exp1 (pairs divergence)",
         "exp2": "Exp2 (vol sleeve + stocks)", "exp7": "Exp7 (SPY VWAP+EMA)", "SPY": "SPY (buy and hold)"}
COL = {"Calm": "#199e70", "Normal": "#c98500", "Stressed": "#d55181"}

R = pd.read_csv(S / "alloc/returns.csv", index_col=0, parse_dates=True)
pots = list(R.columns)
px = lambda t: pd.read_csv(S / f"dl/papers/prices/{t}.csv", index_col="date", parse_dates=True)["adj_close"]
spy, vix, irx = px("SPY"), px("^VIX"), px("^IRX")
rv = np.log(spy).diff().rolling(21).std() * np.sqrt(252) * 100
F = pd.DataFrame({"vix": vix, "spread": vix - rv}).dropna()
F = F[F.index.isin(spy.index)]
X = pd.DataFrame({"lvix": np.log(F["vix"]), "spread": F["spread"]})


def kmeans(x, k=3, runs=6, seed=0, iters=60):
    rng = np.random.default_rng(seed)
    best = (np.inf, None)
    for _ in range(runs):
        c = x[[rng.integers(len(x))]]
        while len(c) < k:
            d2 = ((x[:, None] - c[None]) ** 2).sum(-1).min(1)
            c = np.vstack([c, x[rng.choice(len(x), p=d2 / d2.sum())]])
        for _ in range(iters):
            lab = ((x[:, None] - c[None]) ** 2).sum(-1).argmin(1)
            new = np.array([x[lab == j].mean(0) if (lab == j).any() else c[j] for j in range(k)])
            if np.allclose(new, c):
                break
            c = new
        sse = ((x - c[lab]) ** 2).sum()
        if sse < best[0]:
            best = (sse, c)
    return best[1]


def fit(upto):
    """Centres (standardized space) and the scaler, from all days up to `upto`, named by VIX level."""
    past = X.loc[:upto]
    mu, sd = past.mean().to_numpy(), past.std().to_numpy()
    c = kmeans(((past - mu) / sd).to_numpy())
    order = np.argsort(c[:, 0])
    return c[order], mu, sd


def label(fitted, rows):
    c, mu, sd = fitted
    z = ((rows - mu) / sd).to_numpy()
    return pd.Series(np.array(NAMES)[((z[:, None] - c[None]) ** 2).sum(-1).argmin(1)], index=rows.index)


def stats(x):
    n = len(x)
    if n < 5 or x.std() == 0:
        return {"n": n, "ann": np.nan, "sharpe": np.nan, "t": np.nan, "mean": np.nan, "var": np.nan}
    return {"n": n, "ann": x.mean() * 252 * 100, "sharpe": x.mean() / x.std() * np.sqrt(252),
            "t": x.mean() / x.std() * np.sqrt(n), "mean": x.mean(), "var": x.var()}


# ---------- weekly walk-forward over 2020-2025 (and on into the test, same procedure) ----------
days = R.index[(R.index >= TRAIN0) & (R.index <= TEST1)]
fridays = [d for i, d in enumerate(days) if i + 1 == len(days) or days[i + 1].isocalendar()[1] != d.isocalendar()[1]]
start_fit = R.index[R.index < TRAIN0][-1]
weeks = [start_fit] + fridays
live_lab = pd.Series(index=days, dtype=object)      # regime each close, from the latest Friday map before or on it
table = []                                           # per week, per strategy, per regime: record from 2020 to that week
fits = {}
import pickle
CACHE = Path(__file__).with_name("wf.pkl")
cached = pickle.load(open(CACHE, "rb")) if CACHE.exists() else None
for i, w in enumerate(weeks if cached is None else []):
    f = fit(w)
    fits[w] = f
    nxt = days[(days > w) & (days <= (weeks[i + 1] if i + 1 < len(weeks) else days[-1]))]
    if len(nxt):
        live_lab[nxt] = label(f, X.loc[nxt]).to_numpy()
    if w < pd.Timestamp(TRAIN0):
        continue
    hist_lab = label(f, X.loc[TRAIN0:w]).shift(1)                 # a day's return scored in the previous close's regime
    for k in pots:
        r = R.loc[TRAIN0:w, k].dropna()
        lab_k = hist_lab.reindex(r.index)
        for g in NAMES:
            table.append({"week": w, "pot": k, "regime": g, **stats(r[lab_k == g])})
    if i % 50 == 0:
        print("fit", w.date(), flush=True)
if cached is None:
    pickle.dump((table, live_lab, fits), open(CACHE, "wb"))
else:
    table, live_lab, fits = cached
T = pd.DataFrame(table)
T.to_csv(OUT / "weekly_refit_table.csv", index=False)

# point-in-time label for the day a return is earned = regime at the previous close
prev = live_lab.shift(1)
prev.iloc[0] = label(fits[start_fit], X.loc[[start_fit]]).iloc[0]

# ---------- does it hold, or was it a fluke? ----------
summary = {}
train_weeks = [w for w in fridays if w <= pd.Timestamp(TRAIN1)]
for k in pots:
    t = T[(T.pot == k) & (T.week <= TRAIN1)]
    pv = t.pivot(index="week", columns="regime", values="sharpe")[list(NAMES)]
    pv = pv[pv.notna().any(axis=1)]
    best = pv.fillna(-99).idxmax(axis=1)
    final_best = best.iloc[-1]
    later = best[best.index >= "2021-01-01"]
    # out of sample: was next week's money made in the regimes the past called "good" (mean > 0 and t >= 1)?
    good_days, bad_days = [], []
    for i, w in enumerate(train_weeks[:-1]):
        tw = t[t.week == w].set_index("regime")
        good = {g for g in NAMES if tw.at[g, "n"] >= MIN_DAYS and tw.at[g, "t"] >= T_MIN}
        nxt = days[(days > w) & (days <= train_weeks[i + 1])]
        for d in nxt:
            x = R.at[d, k]
            if pd.notna(x):
                (good_days if prev[d] in good else bad_days).append(x)
    last = t[t.week == t.week.max()].set_index("regime")
    summary[k] = {
        "label": LABEL[k], "best_regime_end_2025": final_best,
        "weeks_best_regime_same_as_final_pct": round(100 * (later == final_best).mean(), 1),
        "weeks_best_regime_switched": int((best != best.shift()).sum() - 1),
        "end_2025": {g: {"days": int(last.at[g, "n"]), "ann_pct": round(last.at[g, "ann"], 1),
                         "sharpe": round(last.at[g, "sharpe"], 2), "t": round(last.at[g, "t"], 2)} for g in NAMES},
        "oos_next_week": {"in_called_good_regimes": stats(pd.Series(good_days, dtype=float)),
                          "in_other_regimes": stats(pd.Series(bad_days, dtype=float))},
    }
    for s in summary[k]["oos_next_week"].values():
        for kk in ("ann", "sharpe", "t"):
            s[kk] = None if pd.isna(s[kk]) else round(float(s[kk]), 2)
        s.pop("mean"); s.pop("var")

# ---------- spread diagrams ----------
cmap = LinearSegmentedColormap.from_list("div", ["#d95926", "#f0efec", "#3987e5"])
fin = fits[train_weeks[-1]]
cen = fin[0] * fin[2] + fin[1]                       # back to raw units (log vix, spread)
tr = F.loc[TRAIN0:TRAIN1]
plt.rcParams.update({"font.size": 10, "axes.edgecolor": "#bbb", "axes.labelcolor": "#333", "xtick.color": "#555",
                     "ytick.color": "#555", "axes.spines.top": False, "axes.spines.right": False})
for k in pots:
    fig, (a, b) = plt.subplots(1, 2, figsize=(13, 5.2), gridspec_kw={"width_ratios": [1, 1.25]})
    r = R[k].reindex(tr.index)
    nx = r.shift(-1)                                 # the return earned the day after that state was seen
    m = nx.notna()
    xs, ys, vs = tr["vix"][m], tr["spread"][m], nx[m] * 1e4
    lim = np.nanpercentile(np.abs(vs), 90) or 1
    hb = a.hexbin(xs, ys, C=vs, reduce_C_function=np.mean, gridsize=(30, 18), xscale="log", mincnt=3, extent=(1, np.log10(60), -20, 20), cmap=cmap,
                  norm=TwoSlopeNorm(0, -lim / 3, lim / 3), linewidths=0.3, edgecolors="#fcfcfb")
    for j, g in enumerate(NAMES):
        a.scatter(np.exp(cen[j, 0]), cen[j, 1], s=90, color=COL[g], edgecolor="white", linewidth=2, zorder=5)
        a.annotate(g, (np.exp(cen[j, 0]), cen[j, 1]), xytext=(8, 6), textcoords="offset points", fontsize=10,
                   color="#222", weight="bold")
    a.axhline(0, color="#999", lw=0.8, ls=":")
    a.set_xlim(10, 60); a.set_ylim(*np.nanpercentile(tr["spread"], [0.5, 99.5]))
    a.set_xticks([10, 15, 20, 30, 40, 60]); a.set_xticklabels(["10", "15", "20", "30", "40", "60"]); a.set_xlim(10, 60)
    a.set_xlabel("VIX (implied vol, log scale)"); a.set_ylabel("Spread: VIX minus SPY 21-day realized vol (points)")
    a.set_title("Next-day return by market state, 2020–2025", loc="left", fontsize=11)
    cb = fig.colorbar(hb, ax=a, shrink=0.8, pad=0.02); cb.set_label("avg next-day return (bp)")
    t = T[(T.pot == k) & (T.week <= TRAIN1)].pivot(index="week", columns="regime", values="sharpe")
    for g in NAMES:
        b.plot(t.index, t[g].clip(-4, 6), color=COL[g], lw=2, label=g)
        b.annotate(f"{g} {t[g].iloc[-1]:.2f}", (t.index[-1], np.clip(t[g].iloc[-1], -4, 6)), xytext=(4, 0),
                   textcoords="offset points", fontsize=9, color="#222", va="center")
    b.axhline(0, color="#999", lw=0.8)
    b.set_ylabel("Sharpe in that regime, 2020 → that week")
    b.set_title("Re-fitted every Friday: does the best regime hold?", loc="left", fontsize=11)
    b.legend(frameon=False, loc="upper left", ncol=3)
    b.set_xlim(t.index[0], t.index[-1] + pd.Timedelta(days=200))
    s = summary[k]
    fig.suptitle(f"{LABEL[k]}  ·  best at end of 2025: {s['best_regime_end_2025']}  ·  same best in "
                 f"{s['weeks_best_regime_same_as_final_pct']}% of weeks since 2021", x=0.01, ha="left", fontsize=12.5,
                 weight="bold")
    fig.tight_layout()
    fig.savefig(OUT / f"spread_{k}.png", dpi=110, facecolor="#fcfcfb")
    plt.close(fig)

# ---------- the test: 500 CAD from Jan 1 2026 to Oct 1 2026, same weekly refit ----------
fx = pd.read_csv("/mnt/project-files/research/portfolio_2019/fx_usdcad_boc.txt", sep=" ", header=None, index_col=0,
                 parse_dates=True)[1]
cash_daily = (irx / 100 / 252).reindex(R.index).ffill()
tdays = R.index[(R.index > TRAIN1) & (R.index <= TEST1)]


def weights_for(w, regime, use_filter, only=None):
    tw = T[(T.week == w) & (T.regime == regime)].set_index("pot")
    allr = {k: R.loc[TRAIN0:w, k].dropna() for k in pots}
    sc = {}
    for k in (only or pots):
        n, mean, var, t = tw.at[k, "n"], tw.at[k, "mean"], tw.at[k, "var"], tw.at[k, "t"]
        if n < MIN_DAYS:
            mean, var, t = allr[k].mean(), allr[k].var(), allr[k].mean() / allr[k].std() * np.sqrt(len(allr[k]))
        if mean > 0 and (not use_filter or t >= T_MIN):
            sc[k] = mean / var
    if not sc:
        return {}
    s = pd.Series(sc); s = s / s.sum()
    for _ in range(10):
        over = s > CAP
        if not over.any():
            break
        extra = (s[over] - CAP).sum(); s[over] = CAP
        free = s < CAP
        if not free.any():
            break
        s[free] += extra * s[free] / s[free].sum()
    return s.to_dict()


def initial(use_filter=True, only=None):
    """Weights set at the Dec 31 2025 close, before the first test day."""
    d0 = R.index[R.index <= TRAIN1][-1]
    return weights_for(max(w for w in fits if w <= d0), live_lab[d0], use_filter, only)


def simulate_from_start(use_filter=True, only=None):
    w0 = initial(use_filter, only)
    w, cur_reg = w0, live_lab[R.index[R.index <= TRAIN1][-1]]
    out, log = [], [{"date": str(R.index[R.index <= TRAIN1][-1].date()), "regime": cur_reg,
                     "weights": {k: round(v, 3) for k, v in w0.items()}}]
    last_fri = max(w for w in fits if w <= R.index[R.index <= TRAIN1][-1])
    for d in tdays:
        ret = sum(x * R.at[d, k] for k, x in w.items() if pd.notna(R.at[d, k]))
        ret += (1 - sum(w.values())) * cash_daily[d]
        if d in fits:
            last_fri = d
        reg = live_lab[d]
        if d in fits or reg != cur_reg:
            nw = weights_for(last_fri, reg, use_filter, only)
            ret -= SWITCH * sum(abs(nw.get(k, 0) - w.get(k, 0)) for k in set(nw) | set(w))
            if {k: round(v, 3) for k, v in nw.items()} != {k: round(v, 3) for k, v in w.items()}:
                log.append({"date": str(d.date()), "regime": reg, "weights": {k: round(v, 3) for k, v in nw.items()}})
            w, cur_reg = nw, reg
        out.append(ret)
    return pd.Series(out, index=tdays), log


fx_d = fx.reindex(R.index.union(fx.index)).ffill()
d0 = R.index[R.index <= TRAIN1][-1]
fx0, fx1 = fx_d[:d0].iloc[-1], fx_d[:tdays[-1]].iloc[-1]


def cad(ret):
    usd = 500 / fx0 * (1 + ret.fillna(0)).cumprod()
    return usd * fx_d.reindex(ret.index).ffill()


runs = {"Regime method (with fluke filter)": simulate_from_start(True),
        "Regime method, no fluke filter": simulate_from_start(False)}
eq = R.loc[tdays, pots].drop(columns="SPY").mean(axis=1)
res = {}
curves = {}
for name, (ret, log) in runs.items():
    c = cad(ret); curves[name] = c
    res[name] = {"end_cad": round(c.iloc[-1], 2), "made_cad": round(c.iloc[-1] - 500, 2), "trades": len(log), "log": log}
for name, ret in {"Equal weight, all strategies": eq, "SPY buy and hold": R.loc[tdays, "SPY"]}.items():
    c = cad(ret); curves[name] = c
    res[name] = {"end_cad": round(c.iloc[-1], 2), "made_cad": round(c.iloc[-1] - 500, 2)}
for k in pots:
    c = cad(R.loc[tdays, k])
    res[f"only {LABEL[k]}"] = {"end_cad": round(c.iloc[-1], 2), "made_cad": round(c.iloc[-1] - 500, 2)}
for name, c in curves.items():
    r = c.pct_change().dropna()
    res[name]["worst_drop_pct"] = round(100 * (c / c.cummax() - 1).min(), 1)
    res[name]["sharpe_cad"] = round(r.mean() / r.std() * np.sqrt(252), 2)

test_regime_days = live_lab[tdays].value_counts().to_dict()
json.dump({"features": "VIX level and VIX minus SPY 21d realized vol; k=3, refit every Friday on 1998..that Friday",
           "train": [TRAIN0, TRAIN1], "test": [str(tdays[0].date()), str(tdays[-1].date())],
           "fx_usdcad": [fx0, fx1], "test_regime_days": test_regime_days,
           "summary": summary, "test": res}, open(OUT / "result.json", "w"), indent=1, default=float)
pd.DataFrame(curves).to_csv(OUT / "test_equity_cad.csv")

# test chart
fig, a = plt.subplots(figsize=(12, 5))
pal = ["#3987e5", "#d95926", "#199e70", "#c98500"]
for (name, c), col in zip(curves.items(), pal):
    a.plot(c.index, c, color=col, lw=2, label=name)
    a.annotate(f"{c.iloc[-1]:.0f}", (c.index[-1], c.iloc[-1]), xytext=(4, 0), textcoords="offset points",
               fontsize=9, color="#222", va="center")
lo = a.get_ylim()[0]
for d in tdays:
    a.axvspan(d - pd.Timedelta(hours=12), d + pd.Timedelta(hours=12), ymin=0, ymax=0.03,
              color=COL[live_lab[d]], lw=0)
a.axhline(500, color="#999", lw=0.8, ls=":")
a.set_ylabel("CAD"); a.legend(frameon=False, loc="upper left")
a.set_title("Test, Jan 2 – Oct 1 2026: 500 CAD (strip at bottom = regime each day: green Calm, amber Normal, pink Stressed)",
            loc="left", fontsize=11)
fig.tight_layout(); fig.savefig(OUT / "test_500cad.png", dpi=110, facecolor="#fcfcfb")
print(json.dumps({k: {kk: v for kk, v in r.items() if kk != "log"} for k, r in res.items()}, indent=1))
print(test_regime_days)
for k, s in summary.items():
    print(k, s["best_regime_end_2025"], s["weeks_best_regime_same_as_final_pct"], s["weeks_best_regime_switched"],
          {g: (v["ann_pct"], v["sharpe"], v["t"]) for g, v in s["end_2025"].items()},
          {g: (v["ann"], v["sharpe"], v["n"]) for g, v in s["oos_next_week"].items()})
