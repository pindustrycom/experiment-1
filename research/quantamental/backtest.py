"""Monthly backtests of the quantamental slides on point-in-time S&P 500 members, 2010 to 2026-09.

- Piotroski F-score (slide 3): nine yes/no accounting checks; long high scores, short low scores, among cheap stocks.
- Quality minus junk (slide 1, Asness-Frazzini-Pedersen): profitability, growth, safety and payout z-scores.
- 10-K tone (slide 4): Loughran-McDonald finance word lists vs the generic Harvard list.
- One composite of all three (slide 2's breadth argument), and three ways to size each book (slide 5).
- Edge statistics (slide 6): Newey-West t, alpha vs SPY and Fama-French factors, deflated Sharpe over every variant.
Every signal uses only numbers filed before the month's last close; returns are the next month's.
"""
from __future__ import annotations

import io
import json
import logging
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd
import requests
from scipy.stats import norm

import stats as st

log = logging.getLogger("quantamental.backtest")
START, END = "2010-01-31", "2026-09-30"
COST = 0.0010           # per unit traded, each side
BORROW = 0.005 / 12     # per month on the short side
STALE_DAYS = 456        # a 10-K older than 15 months is no longer used


# ---------------------------------------------------------------- universe and prices

def membership(cur: pd.DataFrame, changes: pd.DataFrame, months: pd.DatetimeIndex) -> pd.DataFrame:
    """Month x ticker S&P 500 membership, walking the change list back from today (as in exp2/papers.py)."""
    members = set(cur["ticker"])
    events = list(changes.sort_values("date", ascending=False).itertuples(index=False))
    rows, k = {}, 0
    for m in sorted(months, reverse=True):
        while k < len(events) and events[k].date > m:
            if isinstance(events[k].added, str):
                members.discard(events[k].added)
            if isinstance(events[k].removed, str):
                members.add(events[k].removed)
            k += 1
        rows[m] = set(members)
    tickers = sorted(set().union(*rows.values()))
    return pd.DataFrame({t: [t in rows[m] for m in sorted(rows)] for t in tickers}, index=sorted(rows))


def load_prices(data: Path, tickers: list[str]) -> tuple[pd.DataFrame, pd.DataFrame]:
    adj, close = {}, {}
    for t in tickers:
        f = data / "prices" / f"{t}.csv"
        if f.exists():
            d = pd.read_csv(f, parse_dates=["date"]).drop_duplicates("date").set_index("date").sort_index()
            if len(d) > 60:
                adj[t], close[t] = d["adj_close"], d["close"]
    return pd.DataFrame(adj).sort_index(), pd.DataFrame(close).sort_index()


def risk(adj: pd.DataFrame, spy: pd.Series, months: pd.DatetimeIndex) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Trailing one-year daily beta and idiosyncratic volatility at each month end."""
    r = adj.pct_change(fill_method=None)
    s = spy.reindex(r.index).pct_change(fill_method=None)
    mr, ms = r.rolling(252, min_periods=150).mean(), s.rolling(252, min_periods=150).mean()
    cov = r.mul(s, axis=0).rolling(252, min_periods=150).mean().sub(mr.mul(ms, axis=0))
    vs = s.rolling(252, min_periods=150).var(ddof=0)
    vr = r.pow(2).rolling(252, min_periods=150).mean() - mr.pow(2)
    beta = cov.div(vs, axis=0)
    ivol = np.sqrt((vr - beta.pow(2).mul(vs, axis=0)).clip(lower=0) * 252)
    tot = np.sqrt(vr.clip(lower=0) * 252)
    at = lambda x: x.reindex(months, method="ffill")
    return at(beta), at(ivol), at(tot)


# ---------------------------------------------------------------- signals

def zs(x: pd.Series) -> pd.Series:
    """Cross-sectional rank turned into a z-score (Asness et al.); missing stays missing."""
    r = x.rank()
    return (r - r.mean()) / r.std()


def piotroski(f: pd.DataFrame) -> pd.DataFrame:
    g = lambda k: f[f"gp_{k}"].fillna(f[f"rev_{k}"] - f[f"cogs_{k}"])
    ltd = lambda k: f[f"ltd_{k}"].fillna(0.0)
    roa0, roa1 = f["ni_0"] / f["assets_1"], f["ni_1"] / f["assets_2"]
    lev0 = ltd(0) / ((f["assets_0"] + f["assets_1"]) / 2)
    lev1 = ltd(1) / ((f["assets_1"] + f["assets_2"]) / 2)
    checks = pd.DataFrame({
        "roa_pos": roa0 > 0, "cfo_pos": f["cfo_0"] > 0, "roa_up": roa0 > roa1, "cfo_gt_ni": f["cfo_0"] > f["ni_0"],
        "lev_down": lev0 < lev1, "cr_up": f["ca_0"] / f["cl_0"] > f["ca_1"] / f["cl_1"],
        "no_issue": f["shares_0"] <= f["shares_1"] * 1.001, "gm_up": g(0) / f["rev_0"] > g(1) / f["rev_1"],
        "ato_up": f["rev_0"] / f["assets_1"] > f["rev_1"] / f["assets_2"]})
    ok = f[["ni_0", "ni_1", "cfo_0", "assets_0", "assets_1", "assets_2"]].notna().all(axis=1)
    return checks.assign(fscore=checks.sum(axis=1).where(ok))


def quality(p: pd.DataFrame) -> pd.DataFrame:
    """Quality minus junk inputs per month (p holds one month's cross-section)."""
    gp = lambda k: p[f"gp_{k}"].fillna(p[f"rev_{k}"] - p[f"cogs_{k}"])
    eq = lambda k: p[f"eq_{k}"].where(p[f"eq_{k}"] > 0)
    liab = p["liab_0"].fillna(p["assets_0"] - p["eq_0"])
    roe_hist = pd.concat([p[f"ni_{k}"] / eq(k) for k in range(5)], axis=1)
    comp = {
        "prof": [gp(0) / p["assets_0"], p["ni_0"] / eq(0), p["ni_0"] / p["assets_0"], p["cfo_0"] / p["assets_0"],
                 gp(0) / p["rev_0"], (p["cfo_0"] - p["ni_0"]) / p["assets_0"]],
        "growth": [(gp(0) - gp(3)) / p["assets_3"], (p["ni_0"] - p["ni_3"]) / eq(3), (p["ni_0"] - p["ni_3"]) / p["assets_3"],
                   (p["cfo_0"] - p["cfo_3"]) / p["assets_3"], (gp(0) - gp(3)) / p["rev_3"]],
        "safety": [-p["beta"], -p["ivol"], -liab / p["assets_0"], -roe_hist.std(axis=1).where(roe_hist.notna().sum(axis=1) >= 3)],
        "payout": [-np.log(p["shares_0"] / p["shares_1"]), (p["ni_0"] - (p["eq_0"] - p["eq_1"])) / p["assets_0"]],
    }
    out = pd.DataFrame(index=p.index)
    for k, xs in comp.items():
        z = pd.concat([zs(x.replace([np.inf, -np.inf], np.nan)) for x in xs], axis=1)
        out[k] = zs(z.mean(axis=1).where(z.notna().sum(axis=1) >= max(1, len(xs) // 2)))
    out["quality"] = zs(out.fillna(0).sum(axis=1).where(out[["prof", "safety"]].notna().all(axis=1)))
    return out


# ---------------------------------------------------------------- panel

def build_panel(data: Path, fund: pd.DataFrame, text: pd.DataFrame | None) -> tuple[pd.DataFrame, pd.Series, dict]:
    cur = pd.read_csv(data / "sp500_current.csv")
    changes = pd.read_csv(data / "sp500_changes.csv", parse_dates=["date"])
    months = pd.date_range(START, END, freq="ME")
    member = membership(cur, changes, months)
    tickers = sorted(set(member.columns) & set(fund["ticker"]))
    adj, close = load_prices(data, tickers + ["SPY"])
    spy = adj.pop("SPY")
    close = close.drop(columns="SPY")
    beta, ivol, _ = risk(adj, spy, months)
    madj = adj.reindex(adj.index.union(months)).ffill(limit=5).reindex(months)
    last_obs = adj.apply(lambda c: c.last_valid_index())
    last_px = pd.Series({t: adj.at[d, t] for t, d in last_obs.items() if pd.notna(d)})
    nxt = madj.shift(-1) / madj - 1
    # no price at next month's end: the stock left the data (often a takeover, sometimes a failure). Use its last price,
    # so the move until it left still counts instead of the stock silently dropping out.
    gone = pd.DataFrame({t: [pd.notna(last_obs.get(t)) and last_obs[t] < m + pd.offsets.MonthEnd(1) for m in months] for t in madj.columns},
                        index=months)
    nxt = nxt.where(~(nxt.isna() & madj.notna() & gone), last_px.reindex(madj.columns) / madj - 1)

    fund = fund.sort_values("avail")
    pf = piotroski(fund)
    fund = pd.concat([fund, pf], axis=1)
    rows = []
    for m in months:
        mem = [t for t in member.columns[member.loc[m].values] if t in madj.columns and pd.notna(madj.at[m, t])]
        snap = fund[(fund["avail"] < m) & (fund["avail"] >= m - pd.Timedelta(days=STALE_DAYS)) & fund["ticker"].isin(mem)]
        snap = snap.groupby("ticker").tail(1).set_index("ticker")
        if snap.empty:
            continue
        snap = snap.assign(month=m, ret=nxt.loc[m].reindex(snap.index), beta=beta.loc[m].reindex(snap.index),
                           ivol=ivol.loc[m].reindex(snap.index))
        # market value: the public float on the 10-K cover (dollars, so splits cannot distort it) moved with the price since
        c_now = close.loc[:m].iloc[-1].reindex(snap.index)
        c_then = pd.Series({t: _price_on(close, t, d) for t, d in snap["float_date"].items()})
        snap["mcap"] = snap["float_usd"] * c_now / c_then
        snap["bm"] = snap["eq_0"].where(snap["eq_0"] > 0) / snap["mcap"]
        snap = snap.join(quality(snap))
        rows.append(snap.reset_index())
    P = pd.concat(rows, ignore_index=True)
    if text is not None and len(text):
        P = add_text(P, text)
    info = {"months": int(P["month"].nunique()), "avg_names": float(P.groupby("month").size().mean()),
            "tickers_ever": len(tickers), "tickers_with_prices": int(adj.shape[1]),
            "share_with_fscore": float(P["fscore"].notna().mean()), "share_with_quality": float(P["quality"].notna().mean()),
            "share_with_mcap": float(P["mcap"].notna().mean())}
    return P, spy, info


def _price_on(close: pd.DataFrame, t: str, d) -> float:
    if pd.isna(d) or t not in close:
        return np.nan
    s = close[t].loc[:d].dropna()
    return float(s.iloc[-1]) if len(s) else np.nan


def text_signals(text: pd.DataFrame) -> pd.DataFrame:
    t = text.copy()
    t["filed"] = pd.to_datetime(t["filed"])
    t = t[t["words"] > 2000].sort_values(["cik", "filed"])
    for k in ("lm_neg", "lm_pos", "lm_unc", "h4_neg", "h4_pos"):
        t[f"{k}_sh"] = t[k] / t["words"]
    t["lm_tone"] = t["lm_pos_sh"] - t["lm_neg_sh"]
    t["h4_tone"] = t["h4_pos_sh"] - t["h4_neg_sh"]
    for k in ("lm_neg_sh", "h4_neg_sh", "lm_tone", "h4_tone"):
        t[f"d_{k}"] = t.groupby("cik")[k].diff().where(t.groupby("cik")["filed"].diff().dt.days.between(250, 500))
    return t


def add_text(P: pd.DataFrame, text: pd.DataFrame) -> pd.DataFrame:
    t = text_signals(text)[["cik", "filed", "lm_neg_sh", "h4_neg_sh", "lm_tone", "h4_tone", "d_lm_neg_sh", "d_h4_neg_sh", "d_lm_tone", "d_h4_tone"]]
    P = P.sort_values("month")
    t = t.sort_values("filed").assign(cik=lambda x: x["cik"].astype(int))
    out = pd.merge_asof(P.assign(cik=P["cik"].astype(int)), t, left_on="month", right_on="filed", by="cik",
                        allow_exact_matches=False, tolerance=pd.Timedelta(days=STALE_DAYS))
    return out


# ---------------------------------------------------------------- portfolios

def weights(g: pd.DataFrame, how: str) -> pd.Series:
    if how == "vw":
        w = g["mcap"].where(g["mcap"] > 0)
    elif how == "iv":
        w = 1 / g["ivol"].where(g["ivol"] > 0).fillna(g["ivol"].median())
    else:
        w = pd.Series(1.0, index=g.index)
    w = w.fillna(0)
    return w / w.sum() if w.sum() > 0 else w


def book(P: pd.DataFrame, pick_long, pick_short=None, how: str = "ew", min_names: int = 8) -> pd.Series:
    """Monthly returns after costs of a long book (minus a short book when pick_short is given)."""
    out, prev = {}, pd.Series(dtype=float)
    for m, g in P.groupby("month"):
        g = g.set_index("ticker")
        g = g[g["ret"].notna()]
        L = g[pick_long(g)]
        if len(L) < min_names:
            continue
        w = weights(L, how)
        r = float((w * L["ret"]).sum())
        if pick_short is not None:
            S = g[pick_short(g)]
            if len(S) < min_names:
                continue
            ws = weights(S, how)
            r -= float((ws * S["ret"]).sum()) + BORROW
            w = w.sub(ws, fill_value=0)
        turn = w.sub(prev, fill_value=0).abs().sum()
        out[m + pd.offsets.MonthEnd(1)] = r - COST * turn
        prev = w
    return pd.Series(out).sort_index()


def q(col: str, lo: float, hi: float):
    """Members whose month's percentile of col lies in [lo, hi]."""
    return lambda g: g[col].rank(pct=True).between(lo, hi) & g[col].notna()


def ic(P: pd.DataFrame, col: str) -> dict:
    s = P.dropna(subset=[col, "ret"]).groupby("month").apply(lambda g: g[col].corr(g["ret"], method="spearman") if len(g) > 20 else np.nan,
                                                              include_groups=False).dropna()
    n = P.dropna(subset=[col, "ret"]).groupby("month").size().mean()
    mean = s.mean()
    return {"ic_mean": mean, "ic_t": mean / st.nw_se(s.values, st.default_lags(len(s))), "names": n,
            "law_ir": mean * np.sqrt(12 * n), "months": len(s)}


def strategies(P: pd.DataFrame) -> tuple[dict[str, pd.Series], pd.DataFrame]:
    P = P.copy()
    P["value"] = P.groupby("month")["bm"].rank(pct=True)
    cheap = lambda g: g["value"] >= 2 / 3
    hiF = lambda g: g["fscore"] >= 7
    loF = lambda g: g["fscore"] <= 3
    S = {
        "F-score: cheap high minus cheap low": book(P, lambda g: cheap(g) & hiF(g), lambda g: cheap(g) & loF(g), min_names=5),
        "F-score: cheap high, long only": book(P, lambda g: cheap(g) & hiF(g), min_names=5),
        "F-score: all high minus all low": book(P, hiF, loF),
        "F-score: all high, long only": book(P, hiF),
        "Cheap stocks (top third B/M), long only": book(P, cheap),
        "QMJ: top 30% minus bottom 30%, value weighted": book(P, q("quality", .7, 1), q("quality", 0, .3), "vw"),
        "QMJ: top 30% minus bottom 30%, equal weighted": book(P, q("quality", .7, 1), q("quality", 0, .3)),
        "QMJ: top 30%, long only, value weighted": book(P, q("quality", .7, 1), how="vw"),
        "QMJ: top 30%, long only, equal weighted": book(P, q("quality", .7, 1)),
        "QMJ: top 30%, long only, inverse vol": book(P, q("quality", .7, 1), how="iv"),
    }
    for k in ("prof", "growth", "safety", "payout"):
        S[f"QMJ part {k}: top minus bottom 30%"] = book(P, q(k, .7, 1), q(k, 0, .3))
    if "d_lm_neg_sh" in P:
        for lab, col in (("finance list (LM)", "lm"), ("generic list (Harvard)", "h4")):
            P[f"neg_fall_{col}"] = -P[f"d_{col}_neg_sh"]
            S[f"10-K negativity falling, {lab}: top minus bottom 20%"] = book(P, q(f"neg_fall_{col}", .8, 1), q(f"neg_fall_{col}", 0, .2))
            P[f"low_neg_{col}"] = -P[f"{col}_neg_sh"]
            S[f"10-K least negative, {lab}: top minus bottom 20%"] = book(P, q(f"low_neg_{col}", .8, 1), q(f"low_neg_{col}", 0, .2))
    # composite: equal parts of each signal's z-score (slide 2: more independent signals, more breadth)
    parts = ["fscore", "quality"] + (["neg_fall_lm"] if "neg_fall_lm" in P else [])
    P["composite"] = P.groupby("month")[parts].transform(zs).mean(axis=1, skipna=True)
    S["Composite (F + QMJ + LM): top minus bottom 30%"] = book(P, q("composite", .7, 1), q("composite", 0, .3))
    S["Composite (F + QMJ + LM): top 30%, long only"] = book(P, q("composite", .7, 1))
    S["Composite (F + QMJ + LM): top 30%, long only, inverse vol"] = book(P, q("composite", .7, 1), how="iv")
    S["All members, equal weighted (benchmark)"] = book(P, lambda g: g["ret"].notna())
    return S, P


# ---------------------------------------------------------------- filing-window reaction (Loughran-McDonald 2011)

def filing_reaction(text: pd.DataFrame, fund: pd.DataFrame, data: Path) -> dict:
    """Return from the close before the 10-K to three sessions after, minus SPY, regressed each month on each
    negativity measure (Fama-MacBeth). Slopes per one standard deviation, in percent."""
    t = text_signals(text)
    tick = fund.drop_duplicates("cik").set_index("cik")["ticker"]
    t["ticker"] = t["cik"].map(tick)
    t = t.dropna(subset=["ticker"])
    adj, _ = load_prices(data, sorted(t["ticker"].unique()) + ["SPY"])
    rows = []
    for r in t.itertuples(index=False):
        if r.ticker not in adj:
            continue
        s, b = adj[r.ticker].dropna(), adj["SPY"]
        i = s.index.searchsorted(r.filed)
        if i < 1 or i + 3 >= len(s):
            continue
        d0, d3 = s.index[i - 1], s.index[i + 3]
        rows.append({"filed": r.filed, "x": s[d3] / s[d0] - b[d3] / b[d0], **{k: getattr(r, k) for k in ("lm_neg_sh", "h4_neg_sh", "d_lm_neg_sh", "d_h4_neg_sh")}})
    D = pd.DataFrame(rows)
    out = {"filings": len(D)}
    D["m"] = D["filed"].dt.to_period("M")
    for k in ("lm_neg_sh", "h4_neg_sh", "d_lm_neg_sh", "d_h4_neg_sh"):
        sl = []
        for _, g in D.dropna(subset=[k]).groupby("m"):
            if len(g) >= 20:
                z = (g[k] - g[k].mean()) / g[k].std()
                sl.append(np.polyfit(z, g["x"], 1)[0])
        sl = np.array(sl)
        out[k] = {"slope_pct_per_sd": 100 * sl.mean(), "t": sl.mean() / (sl.std(ddof=1) / np.sqrt(len(sl))), "months": len(sl)}
    return out


# ---------------------------------------------------------------- outside factor data

def french_factors() -> pd.DataFrame | None:
    base = "https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/ftp/"
    try:
        def grab(name):
            z = zipfile.ZipFile(io.BytesIO(requests.get(base + name, timeout=60).content))
            raw = z.read(z.namelist()[0]).decode("latin1").splitlines()
            start = next(i for i, ln in enumerate(raw) if ln.strip().startswith(",") or ln.lower().startswith(",mkt") or "Mkt-RF" in ln or ",Mom" in ln)
            rows = []
            for ln in raw[start + 1:]:
                p = [x.strip() for x in ln.split(",")]
                if len(p[0]) != 6 or not p[0].isdigit():
                    break
                rows.append(p)
            cols = [c.strip() for c in raw[start].split(",")][1:]
            df = pd.DataFrame([r[1:] for r in rows], index=pd.to_datetime([r[0] for r in rows], format="%Y%m") + pd.offsets.MonthEnd(0),
                              columns=cols).astype(float) / 100
            return df
        f5 = grab("F-F_Research_Data_5_Factors_2x3_CSV.zip")
        mom = grab("F-F_Momentum_Factor_CSV.zip")
        return f5.join(mom.rename(columns=lambda c: "MOM"), how="inner")
    except Exception as e:  # noqa: BLE001 - the factor check is optional
        log.warning("French factors: %s", e)
        return None


def factor_alpha(r: pd.Series, F: pd.DataFrame, long_only: bool) -> dict:
    cols = ["Mkt-RF", "SMB", "HML", "RMW", "CMA", "MOM"]
    d = F[cols + ["RF"]].join(r.rename("r"), how="inner").dropna()
    if len(d) < 36:
        return {}
    y = d["r"] - (d["RF"] if long_only else 0)
    X = np.c_[np.ones(len(d)), d[cols].values]
    coef = np.linalg.lstsq(X, y.values, rcond=None)[0]
    u = y.values - X @ coef
    L = st.default_lags(len(d))
    Xu, T = X * u[:, None], len(d)
    S = Xu.T @ Xu / T
    for k in range(1, L + 1):
        G = Xu[k:].T @ Xu[:-k] / T
        S += (1 - k / (L + 1)) * (G + G.T)
    Q = np.linalg.inv(X.T @ X / T)
    se = np.sqrt(np.diag(Q @ S @ Q / T))
    return {"ff_alpha_ann": coef[0] * 12, "ff_alpha_t": coef[0] / se[0], "ff_end": str(d.index[-1].date()),
            **{f"load_{c}": round(float(b), 2) for c, b in zip(cols, coef[1:])}}


# ---------------------------------------------------------------- report

def evaluate(S: dict[str, pd.Series], spy: pd.Series, F: pd.DataFrame | None, prior_trials: int = 300) -> pd.DataFrame:
    spy_m = spy.resample("ME").last().pct_change()
    rows = {}
    srs = [s.mean() / s.std() for s in S.values() if len(s) > 24]
    var = float(np.var(srs, ddof=1))
    for k, s in S.items():
        if len(s) < 24:
            continue
        long_only = "minus" not in k
        row = st.summary(s, spy_m.reindex(s.index), ppy=12)
        row["dsr_here"] = st.deflated_sharpe(s, len(srs), var)["dsr"]
        row["dsr_project"] = st.deflated_sharpe(s, len(srs) + prior_trials, var)["dsr"]
        if long_only:
            ex = s - spy_m.reindex(s.index)
            row["vs_spy_ann"], row["vs_spy_t"] = ex.mean() * 12, ex.mean() / st.nw_se(ex.values, st.default_lags(len(ex)))
        if F is not None:
            row.update(factor_alpha(s, F, long_only))
        for a, b in (("2010-2017", ("2010", "2017")), ("2018-2026", ("2018", "2026"))):
            part = s[b[0]:b[1]]
            row[f"sharpe_{a}"] = part.mean() / part.std() * np.sqrt(12) if len(part) > 12 else np.nan
        rows[k] = row
    R = pd.DataFrame(rows).T
    key = "alpha_t"
    p = {k: float(2 * norm.sf(abs(v))) for k, v in R[key].astype(float).items()}
    R["alpha_holm_p"] = pd.Series(st.holm(p))
    first = min(s.index[0] for s in S.values() if len(s))
    spy_row = st.summary(spy_m[first:END], ppy=12)
    R.loc["SPY"] = pd.Series(spy_row)
    return R


def run(data: Path, fund_csv: Path, text_csv: Path | None, out: Path) -> dict:
    out.mkdir(parents=True, exist_ok=True)
    fund = pd.read_csv(fund_csv, parse_dates=["end", "avail", "float_date"])
    text = pd.read_csv(text_csv) if text_csv and text_csv.exists() else None
    P, spy, info = build_panel(data, fund, text)
    S, P = strategies(P)
    F = french_factors()
    R = evaluate(S, spy, F)
    ics = {c: ic(P, c) for c in ("fscore", "quality", "prof", "growth", "safety", "payout", "neg_fall_lm", "neg_fall_h4", "composite") if c in P}
    react = filing_reaction(text, fund, data) if text is not None and len(text) else {}
    dist = P.groupby("month")["fscore"].apply(lambda s: s.value_counts(normalize=True)).groupby(level=1).mean().sort_index()
    rep = {"universe": info, "ic": ics, "filing_reaction": react, "fscore_distribution": dist.round(3).to_dict(),
           "french_factors": F is not None, "results": json.loads(R.to_json(orient="index"))}
    pd.DataFrame(S).to_csv(out / "monthly_returns.csv", float_format="%.6g")
    P[["month", "ticker", "ret", "fscore", "quality", "prof", "growth", "safety", "payout", "bm", "mcap"]
      + [c for c in ("lm_neg_sh", "h4_neg_sh", "d_lm_neg_sh", "d_h4_neg_sh", "composite") if c in P]].to_csv(out / "panel.csv.gz", index=False, float_format="%.5g")
    (out / "results.json").write_text(json.dumps(rep, indent=1, default=str))
    return rep
