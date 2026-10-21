"""Experiment 2, live: the combined portfolio of exp2/collapse.py, tracked on paper every trading day.

Each morning before the open it predicts the day's portfolio return (a number and a 68% band) from data up to the
last close; each evening after the close it measures what happened, logs the error, and learns from it.

The portfolio is frozen as the backtest chose it on 2005-2018 data (minimum-variance mix, levered to 10% volatility):
  12.5% HRP book of S&P 500 members without the 10% most collapse-prone   (weights set at each month end)
  35.9% SPY when the 2-state Markov regime model says calm, else cash     (decided at each close)
  51.7% synthetic short one-month SPY variance swap at VIX^2, only when VIX^2 is above our forecast of the month's
        variance and the regime is calm                                     (struck at each month end)
  all times 1.35.

The day's P&L, in % of capital, is P = a r + h e + c - b r^2 - cost, where r is SPY's return, e the HRP book's return
beyond its SPY beta, c the variance swap's daily premium and b r^2 its daily realised-variance payout. With
r ~ N(mu, s^2) and s from a daily HAR model of SPY variance (with VIX), the morning prediction is
  E[P] = 100 a mu + c - b (s^2 + mu^2) - cost,   Var[P] = (100 a s)^2 + (100 h s_e)^2 + 2 b^2 s^4.

Learning: two factors are updated from the errors of the last 120 sessions (shrunk towards 1 with 20 pseudo-days)
and used the next morning: a volatility multiplier (realised r^2 against forecast s^2) and a band multiplier
(how wide the errors really were against the predicted band). Both are logged with every prediction.

Paper tracking only: no orders are placed anywhere.
"""
from __future__ import annotations

import json
import logging
import math
import time
from datetime import UTC, datetime
from pathlib import Path
from statistics import NormalDist
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd

from . import collapse, papers, volpred

log = logging.getLogger("exp2.live")
NY = ZoneInfo("America/New_York")
TRAIN_END = "2018-12-31"
WEIGHTS = {"hrp": 0.125, "regime": 0.359, "varswap": 0.517}     # min-variance mix, estimated on 2005-2018
LEVERAGE = 1.35
VEGA = 0.5            # % of capital per vol point of the variance swap
ROLL_COST = 0.5       # vol points paid to strike each month's swap
MU = 0.0004           # SPY's expected daily return (about 10% a year), used for every prediction
LEARN_WINDOW, PRIOR_DAYS = 120, 20
BACKFILL_FROM = "2019-01-01"
LOG_COLS = ["date", "source", "made_at", "last_close", "sigma_raw", "vol_mult", "sigma", "p_hi", "regime_in", "varswap_on", "K",
            "n_days", "first_day", "hrp_beta", "hrp_idio", "band_mult", "exp_ret", "sd", "band_lo", "band_hi", "p_up",
            "exp_hrp", "exp_regime", "exp_varswap", "evaluated_at", "status", "spy_ret", "hrp_ret", "real_hrp", "real_regime",
            "real_varswap", "real_ret", "error_pp", "z", "in68", "in95", "vol_ratio", "up_right"]
N = NormalDist()


# ---------------------------------------------------------------- inputs

def spy_frame(data: Path) -> pd.DataFrame:
    px = papers.load(data, ["SPY", "^VIX"])
    spy = px["SPY"]["adj_close"].astype(float)
    df = pd.DataFrame({"r": spy.pct_change(), "vix": px["^VIX"]["close"].reindex(spy.index).ffill()})
    return df.dropna(subset=["vix"])


def closed_sessions(df: pd.DataFrame, today: pd.Timestamp) -> pd.DataFrame:
    """Drop a bar dated today: before the close, Yahoo's latest row is a live quote, not a close."""
    return df[df.index < today]


def har_inputs(df: pd.DataFrame) -> pd.DataFrame:
    r2 = df["r"] ** 2
    return pd.DataFrame({"one": 1.0, "d": r2, "w": r2.rolling(5).mean(), "m": r2.rolling(22).mean(), "vix": (df["vix"] / 100) ** 2 / 252})


def fit_har(df: pd.DataFrame) -> np.ndarray:
    """Daily HAR with VIX for tomorrow's squared SPY return, fitted on the training years only. Weighted by 1/VIX^2 so
    that errors count relative to the level: plain least squares is dominated by 2008 and forecasts zero on a third
    of calm days."""
    X = har_inputs(df)
    y = (df["r"] ** 2).shift(-1)
    ok = X.notna().all(axis=1) & y.notna() & (df.index <= pd.Timestamp(TRAIN_END))
    w = 1 / X.loc[ok, "vix"].to_numpy()
    return np.linalg.lstsq(X[ok].to_numpy() * w[:, None], y[ok].to_numpy() * w, rcond=None)[0]


def sigma_next(df: pd.DataFrame, b: np.ndarray) -> pd.Series:
    """Forecast of the next session's SPY volatility, made at each close (indexed by that close); never below half
    the volatility VIX implies."""
    X = har_inputs(df)
    s2 = np.maximum(X.to_numpy() @ b, 0.25 * X["vix"].to_numpy())
    return pd.Series(np.sqrt(s2), index=df.index)


def regime(df: pd.DataFrame) -> pd.Series:
    """Filtered probability of the turbulent regime at each close (parameters fitted on the training years)."""
    lr = np.log1p(df["r"].fillna(0.0)).to_numpy()
    p, _ = collapse.hmm2(lr, (df.index <= pd.Timestamp(TRAIN_END)) & df["r"].notna().to_numpy())
    return pd.Series(p, index=df.index)


def month_plans(data: Path, work: Path, months: list[pd.Period], df: pd.DataFrame, p_hi: pd.Series) -> dict:
    """For each month: the HRP book (from the month-end before it), its beta and residual volatility, and whether the
    variance swap is struck (and at what VIX)."""
    if not months:
        return {}
    P, C, member, spy, vix, rf, sectors = collapse.load_all(data, work, TRAIN_END)
    P["p_hi"] = P["month"].map(p_hi.groupby(p_hi.index.to_period("M")).last())
    P = collapse.collapse_labels(P, C, spy)
    _, P = collapse.collapse_model(P, TRAIN_END)
    _, sdf = volpred.spy_timing(data, TRAIN_END)
    R = np.log(C).diff()
    me = df.groupby(df.index.to_period("M")).apply(lambda d: d.index[-1])
    out = {}
    for m in months:
        prev = m - 1
        if prev not in me.index:
            continue
        end = me[prev]
        w = hrp_weights(P, R, prev, end)
        if w is None:
            continue
        hr = (np.expm1(R.loc[:end].iloc[-252:][w.index]).fillna(0.0) @ w)
        sr = df["r"].reindex(hr.index)
        beta = float(np.cov(hr, sr)[0, 1] / sr.var())
        idio = float((hr - beta * sr).std())
        K = float(df.loc[end, "vix"])
        fvar = float(np.exp(2 * sdf["fvol"].get(prev, np.nan)) * 252 * 1e4)
        on = bool(K ** 2 > fvar and p_hi.loc[end] < 0.5) if not math.isnan(fvar) else False
        sessions = df.index[df.index.to_period("M") == m]
        weekdays = len(pd.bdate_range(m.start_time, m.end_time))
        n = len(sessions) if m < df.index[-1].to_period("M") else weekdays - _holidays(m)
        out[str(m)] = {"weights": w.round(6).to_dict(), "beta": beta, "idio": idio, "K": K, "fvar": fvar, "on": on,
                       "p_hi_end": float(p_hi.loc[end]), "n_days": int(n), "formed": str(end.date())}
    (work / "panel_stage1.pkl").unlink(missing_ok=True)          # large; rebuilt from prices when next needed
    return out


def _holidays(m: pd.Period) -> int:
    """NYSE full-day closures on weekdays in a month: federal holidays except Columbus and Veterans Day, plus Good Friday."""
    from pandas.tseries.holiday import GoodFriday, USFederalHolidayCalendar
    h = USFederalHolidayCalendar().holidays(m.start_time, m.end_time, return_name=True)
    days = {d for d, name in h.items() if name not in ("Columbus Day", "Veterans Day")}
    days |= set(GoodFriday.dates(m.start_time, m.end_time))
    return sum(1 for d in days if d.weekday() < 5)


def hrp_weights(P: pd.DataFrame, R: pd.DataFrame, month: pd.Period, end: pd.Timestamp) -> pd.Series | None:
    """The backtest's 'HRP without collapse-prone stocks' book formed at the end of `month`."""
    from sklearn.covariance import LedoitWolf
    g = P[(P["month"] == month) & P["pc_gbt"].notna()]
    if len(g) < 50:
        return None
    W = R.loc[:end].iloc[-252:][[t for t in g["ticker"] if t in R.columns]]
    ok = W.notna().sum() >= 240
    W = W.loc[:, ok].fillna(0.0)
    g = g.set_index("ticker").loc[W.columns]
    safe = (g["pc_gbt"].rank(pct=True) <= 0.9).to_numpy()
    S = LedoitWolf().fit(W.to_numpy()).covariance_
    ix = np.where(safe)[0]
    w = collapse._hrp(S[np.ix_(ix, ix)])
    return pd.Series(w, index=W.columns[ix])


# ---------------------------------------------------------------- one day

def predict(plan: dict, sigma_raw: float, p_hi: float, first_day: bool, vol_mult: float, band_mult: float) -> dict:
    s = sigma_raw * vol_mult
    reg = 1.0 if p_hi < 0.5 else 0.0
    a = LEVERAGE * (WEIGHTS["hrp"] * plan["beta"] + WEIGHTS["regime"] * reg)
    h = LEVERAGE * WEIGHTS["hrp"]
    g = LEVERAGE * WEIGHTS["varswap"] * VEGA / (2 * plan["K"] * plan["n_days"]) if plan["on"] else 0.0
    c, b = g * plan["K"] ** 2, g * 252e4
    cost = LEVERAGE * WEIGHTS["varswap"] * VEGA * ROLL_COST if plan["on"] and first_day else 0.0
    exp_v = c - b * (s ** 2 + MU ** 2) - cost
    exp = 100 * a * MU + exp_v
    sd = math.sqrt((100 * a * s) ** 2 + (100 * h * plan["idio"]) ** 2 + 2 * b ** 2 * s ** 4)
    band = sd * band_mult
    return {"sigma_raw": sigma_raw, "vol_mult": vol_mult, "sigma": s, "p_hi": p_hi, "regime_in": bool(reg), "varswap_on": plan["on"],
            "K": plan["K"], "n_days": plan["n_days"], "first_day": first_day, "hrp_beta": plan["beta"], "hrp_idio": plan["idio"],
            "band_mult": band_mult, "exp_ret": exp, "sd": sd, "band_lo": exp - band, "band_hi": exp + band,
            "p_up": N.cdf(exp / band) if band > 0 else 0.5, "exp_hrp": 100 * LEVERAGE * WEIGHTS["hrp"] * plan["beta"] * MU,
            "exp_regime": 100 * LEVERAGE * WEIGHTS["regime"] * reg * MU, "exp_varswap": exp_v}


def realize(row: dict, plan: dict, spy_ret: float, stock_rets: pd.Series) -> dict:
    w = pd.Series(plan["weights"])
    hrp = float((stock_rets.reindex(w.index).fillna(0.0) * w).sum())
    g = LEVERAGE * WEIGHTS["varswap"] * VEGA / (2 * plan["K"] * plan["n_days"]) if plan["on"] else 0.0
    cost = LEVERAGE * WEIGHTS["varswap"] * VEGA * ROLL_COST if plan["on"] and row["first_day"] else 0.0
    rv = g * plan["K"] ** 2 - g * 252e4 * spy_ret ** 2 - cost
    rh = 100 * LEVERAGE * WEIGHTS["hrp"] * hrp
    rr = 100 * LEVERAGE * WEIGHTS["regime"] * spy_ret * (1.0 if row["regime_in"] else 0.0)
    real = rh + rr + rv
    err = real - row["exp_ret"]
    band = (row["band_hi"] - row["band_lo"]) / 2
    z = err / band if band > 0 else 0.0
    return {"status": "scored", "spy_ret": 100 * spy_ret, "hrp_ret": 100 * hrp, "real_hrp": rh, "real_regime": rr, "real_varswap": rv,
            "real_ret": real, "error_pp": err, "z": z, "in68": abs(z) <= 1, "in95": abs(z) <= 1.96,
            "vol_ratio": abs(spy_ret) / row["sigma"] if row["sigma"] > 0 else None, "up_right": (real > 0) == (row["p_up"] > 0.5)}


def learned(log_df: pd.DataFrame) -> tuple[float, float]:
    """Volatility and band multipliers from the last 120 scored sessions, shrunk towards 1."""
    s = log_df[log_df["status"] == "scored"].tail(LEARN_WINDOW)
    if s.empty:
        return 1.0, 1.0
    ratio = ((s["spy_ret"].astype(float) / 100) ** 2).sum() / (s["sigma_raw"].astype(float) ** 2).sum()
    vol = math.sqrt((PRIOR_DAYS + len(s) * ratio) / (PRIOR_DAYS + len(s)))
    z2 = (s["error_pp"].astype(float) / s["sd"].astype(float)) ** 2
    band = math.sqrt((PRIOR_DAYS + z2.sum()) / (PRIOR_DAYS + len(s)))
    return vol, band


# ---------------------------------------------------------------- the run

def load_log(state: Path) -> pd.DataFrame:
    f = state / "log" / "predictions.csv"
    if f.exists():
        return pd.read_csv(f, dtype={"date": str})
    return pd.DataFrame(columns=LOG_COLS)


def save_log(state: Path, df: pd.DataFrame) -> None:
    (state / "log").mkdir(parents=True, exist_ok=True)
    df[LOG_COLS].to_csv(state / "log" / "predictions.csv", index=False, float_format="%.6g")


def backfill(df: pd.DataFrame, sig: pd.Series, p_hi: pd.Series, plans: dict, R: pd.DataFrame, L: pd.DataFrame | None = None,
             source: str = "backfill") -> pd.DataFrame:
    """Replay every session since 2019 that the log lacks, as if the tracker had run: same plans, same learning,
    outcomes known. Used once to build the history and later to fill sessions a scheduled run missed."""
    L = L if L is not None else pd.DataFrame(columns=LOG_COLS)
    have = set(L["date"].astype(str))
    rows: list[dict] = []
    days = df.index[df.index >= pd.Timestamp(BACKFILL_FROM)]
    start = pd.Timestamp(max(have)) if have else days[0] - pd.Timedelta(days=1)
    for d in days[days > start]:
        plan = plans.get(str(d.to_period("M")))
        if plan is None or str(d.date()) in have or not _book_priced(plan, R, d):
            continue
        prev = df.index[df.index.get_loc(d) - 1]
        cur = pd.concat([L, pd.DataFrame(rows, columns=LOG_COLS)], ignore_index=True) if rows else L
        vm, bm = learned(cur)
        first = prev.to_period("M") != d.to_period("M")
        row = {"date": str(d.date()), "source": source, "made_at": "", "last_close": str(prev.date()),
               **predict(plan, float(sig.loc[prev]), float(p_hi.loc[prev]), first, vm, bm)}
        row.update(realize(row, plan, float(df.loc[d, "r"]), R.loc[d] if d in R.index else pd.Series(dtype=float)))
        row["evaluated_at"] = ""
        rows.append(row)
    return pd.concat([L, pd.DataFrame(rows, columns=LOG_COLS)], ignore_index=True) if rows else L


def run(state: Path, site: Path, mode: str, now: datetime | None = None, fetch: bool = True) -> dict:
    t0 = time.time()
    now = now or datetime.now(UTC)
    today = pd.Timestamp(now.astimezone(NY).date())
    data, work = state / "data", state / "model"
    work.mkdir(parents=True, exist_ok=True)
    rep = {"mode": mode, "now": now.isoformat(timespec="seconds"), "ny_date": str(today.date()), "warnings": []}
    plans = json.loads((work / "plans.json").read_text()) if (work / "plans.json").exists() else {}
    if fetch:
        rep["fetch"] = refresh(data, today, everything=mode == "evening" or str(today.to_period("M")) not in plans)
    df = spy_frame(data)
    if mode == "morning":
        df = closed_sessions(df, today)
    b = fit_har(df)
    sig = sigma_next(df, b)
    p_hi = regime(df)
    last = df.index[-1]
    rep["last_close"] = str(last.date())
    # monthly plans: everything since the backfill start, plus the month being traded
    need = pd.period_range(BACKFILL_FROM, max(today, last).to_period("M"), freq="M")
    missing = [m for m in need if str(m) not in plans and (m - 1) <= last.to_period("M") and (m < last.to_period("M") or _month_closed(m - 1, df, today))]
    if missing:
        log.info("building plans for %s", ", ".join(map(str, missing)))
        plans.update(month_plans(data, work, missing, df, p_hi))
        (work / "plans.json").write_text(json.dumps(plans))
    C = pd.DataFrame({t: d["adj_close"] for t, d in papers.load(data, _tickers(data)).items()}).sort_index()
    R = C.pct_change(fill_method=None)
    L = load_log(state)
    # score whatever has an outcome by now (normally in the evening; the morning catches a late price feed)
    L, rep["scored"] = evening(L, df, plans, R, now)
    n0 = len(L)
    L = backfill(df, sig, p_hi, plans, R, L, "backfill" if L.empty else "catch-up")
    rep["backfilled_days"] = len(L) - n0
    if mode == "morning":
        rep["prediction"] = morning(L, df, sig, p_hi, plans, today, now)
        if rep["prediction"].get("row"):
            L = pd.concat([L, pd.DataFrame([rep["prediction"]["row"]])], ignore_index=True)
    save_log(state, L)
    from .live_site import build
    rep["site"] = build(site, L, plans, df)
    rep["seconds"] = round(time.time() - t0, 1)
    return rep


def _book_priced(plan: dict, R: pd.DataFrame, d: pd.Timestamp, need: float = 0.9) -> bool:
    """True when stocks holding at least 90% of the book's weight have a return for day d."""
    if d not in R.index:
        return False
    w = pd.Series(plan["weights"])
    return float(w[R.loc[d].reindex(w.index).notna().to_numpy()].sum() / w.sum()) >= need


def _month_closed(m: pd.Period, df: pd.DataFrame, today: pd.Timestamp) -> bool:
    """True once month m has ended (today is in a later month) and its last session is in the data."""
    return today.to_period("M") > m and df.index[-1].to_period("M") >= m


def _tickers(data: Path) -> list[str]:
    return sorted(p.stem for p in (data / "prices").glob("*.csv") if not p.stem.startswith("^") and p.stem != "SPY")


def refresh(data: Path, today: pd.Timestamp, everything: bool = True) -> dict:
    """Prices for SPY, VIX, T-bills and every past or present member; the member list itself once a month.
    The morning only needs SPY and VIX, unless a new month's book has to be formed."""
    if not everything:
        out = papers.update(data, ["SPY", "^VIX"], pause=0.2, max_age_days=0.2)
        out["spy_last"] = _last_row(data, "SPY")
        return out
    out = {}
    cur_f = data / "sp500_current.csv"
    if not cur_f.exists() or time.time() - cur_f.stat().st_mtime > 25 * 86400:
        try:
            cur, changes = papers.sp500_tables()
            if len(cur) > 400:
                cur.to_csv(cur_f, index=False)
                changes.to_csv(data / "sp500_changes.csv", index=False)
        except Exception as ex:  # noqa: BLE001 - keep last month's list
            out["members_warning"] = f"{type(ex).__name__}: {str(ex)[:200]}"
    cur = pd.read_csv(cur_f)
    tick = sorted(set(cur["ticker"]) | set(_tickers(data)))
    out.update(papers.update(data, ["SPY", "^VIX", "^IRX", *tick], pause=0.2, max_age_days=0.2))
    out["failed"] = out.get("failed", [])[:50]
    out["spy_last"] = _last_row(data, "SPY")
    return out


def _last_row(data: Path, ticker: str) -> dict:
    try:
        return pd.read_csv(data / "prices" / f"{ticker}.csv").iloc[-1].to_dict()
    except (OSError, ValueError, IndexError):
        return {}


def morning(L: pd.DataFrame, df: pd.DataFrame, sig: pd.Series, p_hi: pd.Series, plans: dict, today: pd.Timestamp, now: datetime) -> dict:
    last = df.index[-1]
    if today.weekday() >= 5:
        return {"skipped": "weekend"}
    if today <= last:
        return {"skipped": f"the close of {today.date()} is already in the data; predictions are made before the open"}
    if (L["date"] == str(today.date())).any():
        return {"skipped": "already predicted"}
    plan = plans.get(str(today.to_period("M")))
    if plan is None:
        return {"skipped": f"no plan for {today.to_period('M')} (the month-end data is missing)"}
    first = last.to_period("M") != today.to_period("M")
    vm, bm = learned(L)
    row = {"date": str(today.date()), "source": "live", "made_at": now.isoformat(timespec="seconds"), "last_close": str(last.date()),
           **predict(plan, float(sig.loc[last]), float(p_hi.loc[last]), first, vm, bm), "status": "pending"}
    return {"row": row}


def evening(L: pd.DataFrame, df: pd.DataFrame, plans: dict, R: pd.DataFrame, now: datetime) -> tuple[pd.DataFrame, list]:
    done = []
    L = L.copy()
    for i in L.index[(L["source"] == "live") & (L["status"] == "pending")]:
        d = pd.Timestamp(L.at[i, "date"])
        plan = plans.get(str(d.to_period("M")))
        if d in df.index and plan is not None and not _book_priced(plan, R, d):
            continue                                                   # SPY is in but most stocks are not yet: wait
        if d in df.index and plan is not None:
            row = L.loc[i].to_dict()
            for k in ("first_day", "regime_in", "varswap_on"):
                row[k] = str(row[k]).lower() == "true"
            res = realize(row, plan, float(df.loc[d, "r"]), R.loc[d] if d in R.index else pd.Series(dtype=float))
            res["evaluated_at"] = now.isoformat(timespec="seconds")
            for k, v in res.items():
                L.at[i, k] = v
            done.append(str(d.date()))
        elif df.index[-1] > d:                                         # later sessions exist, so the market was closed
            L.at[i, "status"] = "market closed"
            done.append(f"{d.date()} closed")
    return L, done
