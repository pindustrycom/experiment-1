"""Take-Two (TTWO) around the GTA VI launch (Nov 19, 2026). Paper research only.

    python analysis.py DATA_DIR OUT_DIR       # DATA_DIR = unpacked ttwo-gta-data release (fetch.py)

1. Event study: TTWO return (market-model abnormal, vs SPY) and realized vol around every GTA / Red Dead launch, the
   delays and reveals, and other big Take-Two launches.
2. Volatility: HAR-RV, GARCH(1,1)-t, GJR-GARCH-t and a log-RV Kalman level (the timescale-ladder day layer) on TTWO
   daily data; out-of-sample QLIKE 2015-2026; forecasts for each listed option expiry vs Cboe implied vol; the launch
   "event move" the options price vs the launch-window moves TTWO actually made; chi-square p-value.
3. 500 CAD: block bootstrap of joint daily (TTWO, SPY, USD/CAD) returns, TTWO scaled either to the option-implied vol
   (market view) or to the model vol + a launch jump from history (history view); straddles priced ask -> bid as in
   the straddle8 pot. Distributions in CAD: median, 5-95%, chance of loss.
"""
from __future__ import annotations

import json
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

warnings.filterwarnings("ignore")
RNG = np.random.default_rng(20261119)
LAUNCH = pd.Timestamp("2026-11-19")
ASOF = pd.Timestamp("2026-11-13")

# Event day = first session the market could react (after-close news -> next day). CAR uses days 0..+1 anyway.
EVENTS = [
    # kind, date, label
    ("delay", "2007-08-03", "GTA IV delayed (Oct 2007 -> 2008)"),
    ("launch_rockstar", "2008-04-29", "GTA IV launch"),
    ("launch_rockstar", "2010-05-18", "Red Dead Redemption launch"),
    ("reveal", "2011-10-25", "GTA V announced"),
    ("reveal", "2011-11-02", "GTA V trailer 1"),
    ("launch_other", "2012-05-15", "Max Payne 3 launch"),
    ("launch_other", "2012-09-18", "Borderlands 2 launch"),
    ("delay", "2013-01-31", "GTA V delayed (spring -> Sep 17 2013)"),
    ("launch_other", "2013-03-26", "BioShock Infinite launch"),
    ("launch_rockstar", "2013-09-17", "GTA V launch (PS3/360)"),
    ("launch_rockstar", "2014-11-18", "GTA V re-release (PS4/XB1)"),
    ("delay", "2015-01-13", "GTA V PC delayed (Jan -> Mar)"),
    ("delay", "2015-02-27", "GTA V PC delayed (Mar -> Apr)"),
    ("launch_rockstar", "2015-04-14", "GTA V PC launch"),
    ("reveal", "2016-10-17", "Red Dead Redemption 2 revealed"),
    ("launch_other", "2016-10-07", "Mafia III launch"),
    ("delay", "2017-05-22", "RDR2 delayed (fall 2017 -> spring 2018)"),
    ("delay", "2018-02-08", "RDR2 delayed to Oct 26 2018 (with earnings)"),
    ("launch_rockstar", "2018-10-26", "Red Dead Redemption 2 launch"),
    ("launch_other", "2019-09-13", "Borderlands 3 launch"),
    ("reveal", "2023-11-08", "GTA VI trailer date announced"),
    ("reveal", "2023-12-05", "GTA VI trailer 1"),
    ("launch_other", "2025-02-11", "Civilization VII launch"),
    ("delay", "2025-05-02", "GTA VI delayed (fall 2025 -> May 2026)"),
    ("reveal", "2025-05-06", "GTA VI trailer 2"),
    ("delay", "2025-11-07", "GTA VI delayed (May -> Nov 19 2026)"),
    ("reveal", "2026-06-18", "GTA VI pre-order date announced"),
]


# ------------------------------------------------------------------------------------------------ data
def load(d: Path) -> dict:
    px = {t: pd.read_csv(d / f"prices_{t}.csv", index_col=0, parse_dates=True) for t in ("TTWO", "SPY", "EA")}
    T = px["TTWO"]
    df = pd.DataFrame({"r": np.log(T.adjClose).diff(), "spy": np.log(px["SPY"].adjClose).diff(),
                       "ea": np.log(px["EA"].adjClose).diff()})
    # Garman-Klass daily variance + overnight gap squared (close-to-close variance proxy)
    gk = 0.5 * np.log(T.adjHigh / T.adjLow) ** 2 - (2 * np.log(2) - 1) * np.log(T.adjClose / T.adjOpen) ** 2
    gap = np.log(T.adjOpen / T.adjClose.shift())
    df["rv"] = (gk + gap ** 2).clip(lower=1e-8)
    df = df.loc["1998":ASOF].dropna(subset=["r", "spy"])
    fx = pd.read_csv(d / "usdcad.csv", index_col=0, parse_dates=True)["usdcad"]
    earn = pd.to_datetime(pd.read_csv(d / "earnings_TTWO.csv")["acceptanceDateTime"], utc=True).dt.tz_convert("America/New_York")
    # release after 16:00 -> next session reacts
    eday = [df.index[df.index.searchsorted((t + pd.Timedelta(hours=8)).tz_localize(None).normalize() if t.hour >= 16
                                           else t.tz_localize(None).normalize())] for t in earn
            if t.tz_localize(None) < df.index[-1]]
    return {"df": df, "fx": fx, "earn": pd.DatetimeIndex(sorted(set(eday)))}


def chain(d: Path) -> pd.DataFrame:
    js = json.loads((d / "chain_TTWO.json").read_text())
    o = pd.DataFrame(js["data"]["options"])
    o["exp"] = pd.to_datetime(o.option.str[-15:-9], format="%y%m%d")
    o["cp"] = o.option.str[-9]
    o["K"] = o.option.str[-8:].astype(int) / 1000
    o.attrs["spot"] = js["data"]["close"]
    return o


def atm(o: pd.DataFrame, S: float) -> pd.DataFrame:
    rows = []
    for e, g in o.groupby("exp"):
        k = g.K.iloc[(g.K - S).abs().argsort().iloc[0]]
        c, p = g[(g.K == k) & (g.cp == "C")].iloc[0], g[(g.K == k) & (g.cp == "P")].iloc[0]
        n = int(np.busday_count((ASOF + pd.Timedelta(days=1)).date(), (e + pd.Timedelta(days=1)).date(),
                                holidays=["2026-11-26", "2026-12-25", "2027-01-01", "2027-01-18", "2027-02-15", "2027-04-02",
                                          "2027-05-31", "2027-06-18", "2027-07-05", "2027-09-06", "2027-11-25", "2027-12-24"]))
        rows.append({"exp": e.date().isoformat(), "days": n, "K": k, "iv": (c.iv + p.iv) / 2,
                     "strad_bid": c.bid + p.bid, "strad_ask": c.ask + p.ask, "strad_mid": (c.bid + c.ask + p.bid + p.ask) / 2,
                     "call_ask": c.ask, "call_bid": c.bid, "put_ask": p.ask, "put_bid": p.bid,
                     "oi": c.open_interest + p.open_interest})
    a = pd.DataFrame(rows)
    a["move_mid_pct"] = a.strad_mid / S * 100
    a["spread_pct_of_mid"] = (a.strad_ask - a.strad_bid) / a.strad_mid * 100
    return a


# ------------------------------------------------------------------------------------------------ 1. event study
def event_study(df: pd.DataFrame, earn: pd.DatetimeIndex) -> pd.DataFrame:
    out = []
    idx = df.index
    for kind, ds, label in EVENTS:
        i = idx.searchsorted(pd.Timestamp(ds))
        est = df.iloc[max(0, i - 260): i - 20]
        b = np.polyfit(est.spy, est.r, 1)
        ab = df.r - (b[1] + b[0] * df.spy)
        base_vol = est.r.std() * np.sqrt(252)

        def car(a, z):
            return float(ab.iloc[max(0, i + a): i + z + 1].sum()) if i + z < len(df) else np.nan

        def vol(a, z):
            s = df.r.iloc[max(0, i + a): i + z + 1]
            return float(s.std() * np.sqrt(252)) if i + z < len(df) else np.nan
        near_earn = bool(((earn >= idx[max(0, i - 2)]) & (earn <= idx[min(len(idx) - 1, i + 2)])).any())
        out.append({"kind": kind, "date": idx[i].date().isoformat(), "event": label, "beta": round(b[0], 2),
                    "day0_raw": float(df.r.iloc[i]), "car_0_1": car(0, 1), "car_m60_m1": car(-60, -1),
                    "car_m20_m1": car(-20, -1), "car_m2_p2": car(-2, 2), "raw_m3_p1": float(df.r.iloc[i - 3: i + 2].sum()), "car_0_20": car(0, 20), "car_0_60": car(0, 60),
                    "car_0_250": car(0, 250), "ret5_launchweek": float(df.r.iloc[i - 2: i + 3].sum()),
                    "vol_base": base_vol, "vol_m20_m1": vol(-20, -1), "vol_0_20": vol(0, 20), "earnings_within_2d": near_earn})
    return pd.DataFrame(out)


def earnings_moves(df: pd.DataFrame, earn: pd.DatetimeIndex) -> dict:
    m = df.r.reindex(earn).dropna()
    m = m.loc["2005":]
    return {"n": int(len(m)), "abs_mean": float(m.abs().mean()), "sd": float(m.std()), "median_abs": float(m.abs().median()),
            "last8": {d.date().isoformat(): round(float(v), 4) for d, v in m.iloc[-8:].items()}}


# ------------------------------------------------------------------------------------------------ 2. volatility models
def har_design(rv: pd.Series) -> pd.DataFrame:
    lv = np.log(rv)
    return pd.DataFrame({"d": lv, "w": np.log(rv.rolling(5).mean()), "m": np.log(rv.rolling(22).mean())})


def har_fit_forecast(rv: pd.Series, h: int) -> float:
    """HAR on log RV for the average variance over the next h days (direct regression); returns annualized vol."""
    X = har_design(rv)
    y = np.log(rv[::-1].rolling(h).mean()[::-1].shift(-1))
    D = pd.concat([X, y.rename("y")], axis=1).dropna()
    A = np.c_[np.ones(len(D)), D[["d", "w", "m"]].values]
    coef, *_ = np.linalg.lstsq(A, D.y.values, rcond=None)
    res = D.y.values - A @ coef
    x = np.r_[1, X.iloc[-1].values]
    return float(np.sqrt(np.exp(x @ coef + res.var() / 2) * 252))


def garch_forecast(r: pd.Series, h: int, o: int = 0) -> tuple[float, object]:
    from arch import arch_model
    m = arch_model(r * 100, mean="Constant", vol="GARCH", p=1, o=o, q=1, dist="t").fit(disp="off")
    f = m.forecast(horizon=h, reindex=False).variance.values[-1]
    return float(np.sqrt(f.mean() * 252) / 100), m


def kalman_logrv(rv: pd.Series, h: int) -> float:
    """Day layer of the timescale ladder: AR(1) log-variance state observed with noise (local level, MLE)."""
    import statsmodels.api as sm
    y = np.log(rv).values
    try:
        m = sm.tsa.SARIMAX(y, order=(1, 0, 1), trend="c").fit(disp=False)  # ARMA(1,1) == AR(1) state + noise
        f = m.get_forecast(h).predicted_mean
        s2 = m.resid[-500:].var()
        return float(np.sqrt(np.exp(f + s2 / 2).mean() * 252))
    except Exception:  # noqa: BLE001
        return np.nan


def oos_qlike(df: pd.DataFrame, start: str = "2015-01-01", h: int = 5) -> dict:
    """Monthly refits, forecast next-h-day average variance; QLIKE vs realized squared returns (close-to-close)."""
    from arch import arch_model
    r, rv = df.r, df.rv
    target = (r ** 2)[::-1].rolling(h).mean()[::-1].shift(-1)
    months = pd.date_range(start, df.index[-1 - h], freq="MS")
    rows = []
    for m0 in months:
        tr = df.index[df.index < m0]
        te = df.index[(df.index >= m0) & (df.index < m0 + pd.offsets.MonthBegin(1))]
        if len(te) == 0:
            continue
        # HAR coefficients from training window
        X = har_design(rv)
        y = np.log(rv[::-1].rolling(h).mean()[::-1].shift(-1))
        D = pd.concat([X, y.rename("y")], axis=1).loc[tr].dropna()
        A = np.c_[np.ones(len(D)), D[["d", "w", "m"]].values]
        coef, *_ = np.linalg.lstsq(A, D.y.values, rcond=None)
        s2 = (D.y.values - A @ coef).var()
        g = arch_model(r.loc[:tr[-1]].iloc[-3000:] * 100, vol="GARCH", p=1, q=1, dist="t").fit(disp="off")
        gj = arch_model(r.loc[:tr[-1]].iloc[-3000:] * 100, vol="GARCH", p=1, o=1, q=1, dist="t").fit(disp="off")
        mrows = []
        for t in te:
            j = df.index.get_loc(t)
            if j == 0 or pd.isna(target.iloc[j - 1]):
                continue
            x = np.r_[1, X.iloc[j - 1].values]
            if np.isnan(x).any():
                continue
            fh = np.exp(x @ coef + s2 / 2)
            # GARCH: filter forward with fixed params to day j-1, then h-step forecast
            mrows.append({"date": t, "y": target.iloc[j - 1], "har": fh,
                          "rw22": float((r.iloc[j - 22:j] ** 2).mean())})
        # GARCH forecasts for the month, params fixed at month start (fast: refit with fixed params over data to t)
        for name, mod in (("garch", g), ("gjr", gj)):
            full = arch_model(r.loc[:te[-1]].iloc[-3100:] * 100, vol="GARCH", p=1, o=1 if name == "gjr" else 0, q=1, dist="t")
            fc = full.fix(mod.params).forecast(horizon=h, reindex=False, start=tr[-1]).variance
            fc = fc.mean(axis=1) / 1e4
            for rr in mrows:
                k = df.index[df.index.get_loc(rr["date"]) - 1]
                rr[name] = float(fc.get(k, np.nan))
        rows += mrows
    R = pd.DataFrame(rows).dropna()

    def ql(f):
        return R.y / f - np.log(R.y / f) - 1
    base = ql(R.har)
    out = {"n_days": int(len(R)), "h": h}
    for c in ("har", "garch", "gjr", "rw22"):
        d = ql(R[c]) - base
        import statsmodels.api as sm
        ols = sm.OLS(d.values, np.ones(len(d))).fit(cov_type="HAC", cov_kwds={"maxlags": 10})
        out[c] = {"qlike": float(ql(R[c]).mean()), "vs_har_pct": float(ql(R[c]).mean() / base.mean() * 100 - 100),
                  "dm_t_vs_har": float(ols.tvalues[0]) if c != "har" else 0.0}
    return out


def launch_jump_history(ev: pd.DataFrame, df: pd.DataFrame) -> dict:
    """Launch-window moves: 5-session log return centred on launch day (days -2..+2), standardised by pre-event vol."""
    L = ev[ev.kind == "launch_rockstar"].copy()
    O = ev[ev.kind == "launch_other"].copy()
    D = ev[ev.kind == "delay"].copy()
    return {"rockstar": L, "other": O, "delay": D}


def implied_event(a: pd.DataFrame, base_vol: float) -> dict:
    """Split the Nov 20 expiry's variance into normal days at base_vol and one launch jump."""
    front = a.iloc[0]
    tot = front.iv ** 2 * front.days / 252
    base = base_vol ** 2 * (front.days - 1) / 252
    j = np.sqrt(max(tot - base, 0))
    return {"front_exp": front.exp, "front_days": int(front.days), "front_iv": float(front.iv),
            "total_sd_pct": float(np.sqrt(tot) * 100), "base_vol_used": base_vol, "launch_jump_sd_pct": float(j * 100),
            "straddle_mid_move_pct": float(front.move_mid_pct)}


# ------------------------------------------------------------------------------------------------ 3. 500 CAD simulation
def block_bootstrap(X: np.ndarray, h: int, n: int, block: int = 10) -> np.ndarray:
    T = len(X)
    nb = int(np.ceil(h / block))
    starts = RNG.integers(0, T - block, size=(n, nb))
    idx = (starts[:, :, None] + np.arange(block)[None, None, :]).reshape(n, -1)[:, :h]
    return X[idx]  # n x h x k


def simulate(df: pd.DataFrame, fx: pd.Series, a: pd.DataFrame, model_vol: dict, jump_draws: np.ndarray,
             S: float, usdcad: float, n: int = 40000) -> dict:
    J = pd.concat([df.r, df.spy, np.log(fx).diff().rename("fx")], axis=1).loc["2017":].dropna()
    hist_vol = J.r.std() * np.sqrt(252)
    mu = J.mean().values
    X = (J - J.mean()).values
    rf, erp, beta = 0.037, 0.05, float(np.polyfit(J.spy, J.r, 1)[0])
    drift = np.array([(rf + beta * erp - 0.5 * hist_vol ** 2) / 252, (rf + erp - 0.5 * (J.spy.std() * np.sqrt(252)) ** 2) / 252, 0.0])
    horizons = {"to Nov 20 (launch week)": "2026-11-20", "to Dec 18": "2026-12-18", "to Mar 19 2027 (after Feb earnings)": "2027-03-19",
                "1 year (Nov 2027)": "2027-12-17"}
    stake = 500.0
    out = {"assumptions": {"rf": rf, "erp": erp, "beta_ttwo_spy_2017_2026": beta, "hist_vol_ttwo_2017_2026": hist_vol,
                           "usdcad": usdcad, "spot": S, "paths": n, "block_days": 10,
                           "mu_hist_daily": dict(zip(["ttwo", "spy", "fx"], mu.tolist()))}}
    for name, e in horizons.items():
        row = a[a.exp == e].iloc[0]
        h = int(row.days)
        B = block_bootstrap(X, h, n)
        fxr = B[:, :, 2].sum(1)
        spy = (B[:, :, 1] + drift[1]).sum(1)
        tt = B[:, :, 0]
        # market view: scale TTWO so its horizon vol equals the ATM implied vol for that expiry
        base_sum = tt.sum(1)
        k_imp = row.iv / hist_vol
        for _ in range(3):  # calibrate so the simulated E|S_T - K| equals the straddle mid (risk-neutral fair value)
            ev_abs = np.abs(S * np.exp(base_sum * k_imp + drift[0] * h) - row.K).mean()
            k_imp *= row.strad_mid / ev_abs
        r_imp = (tt * k_imp + drift[0]).sum(1)
        # history view: model vol (GJR-t forecast for the horizon, ex-launch) + one launch jump from past launches
        k_mod = model_vol[e] / hist_vol
        jump = RNG.choice(jump_draws, size=n) if LAUNCH <= pd.Timestamp(e) else 0.0
        r_hist = (tt * k_mod + drift[0]).sum(1) + jump
        res = {"days": h, "iv": float(row.iv), "model_vol_ex_launch": float(model_vol[e]),
               "market_view_vol": float(k_imp * hist_vol), "history_view_vol": float(np.sqrt(((r_hist - r_hist.mean()) ** 2).mean() * 252 / h))}
        for lab, lr in (("TTWO, market-implied risk", r_imp), ("TTWO, history-based risk", r_hist), ("SPY", spy)):
            v = stake * np.exp(lr + fxr)
            res[lab] = summarize(v, stake)
        # straddle (ATM, buy at ask, settle at intrinsic at expiry; the straddle8 pot's ask -> bid rule means any exit
        # before expiry is at the bid)
        for lab, lr in (("straddle, market-implied", r_imp), ("straddle, history-based", r_hist)):
            ST = S * np.exp(lr)
            pay = np.abs(ST - row.K)
            v = stake * pay / row.strad_ask * np.exp(fxr)
            res[lab] = summarize(v, stake)
            res[lab]["contracts_500cad_buys"] = float(stake / usdcad / (row.strad_ask * 100))
            res[lab]["one_contract_cad"] = float(row.strad_ask * 100 * usdcad)
            short = (row.strad_bid - pay) / row.strad_bid  # seller's P/L per dollar of premium (bid), before margin
            res[lab]["short_pl_per_premium_mean"] = float(short.mean())
            res[lab]["short_pl_per_premium_p5"] = float(np.percentile(short, 5))
            res[lab]["short_p_loss"] = float((short < 0).mean())
        out[name] = res
    return out


def summarize(v: np.ndarray, stake: float) -> dict:
    q = np.percentile(v, [5, 25, 50, 75, 95])
    return {"mean": float(v.mean()), "p5": float(q[0]), "p25": float(q[1]), "median": float(q[2]), "p75": float(q[3]),
            "p95": float(q[4]), "p_loss": float((v < stake).mean()), "p_loss_20pct": float((v < 0.8 * stake).mean()),
            "p_gain_20pct": float((v > 1.2 * stake).mean())}


def current_runup(df: pd.DataFrame) -> dict:
    est = df.iloc[-280:-25]
    b = np.polyfit(est.spy, est.r, 1)
    ab = df.r - (b[1] + b[0] * df.spy)
    return {"car_last60": float(ab.iloc[-60:].sum()), "car_last20": float(ab.iloc[-20:].sum()),
            "raw_last60": float(df.r.iloc[-60:].sum()), "raw_ytd": float(df.r.loc["2026"].sum()),
            "spy_ytd": float(df.spy.loc["2026"].sum())}


# ------------------------------------------------------------------------------------------------ main
def main() -> None:
    d, out = Path(sys.argv[1]), Path(sys.argv[2])
    out.mkdir(parents=True, exist_ok=True)
    L = load(d)
    df, fx, earn = L["df"], L["fx"], L["earn"]
    o = chain(d)
    S = float(o.attrs["spot"])
    a = atm(o, S)
    a.to_csv(out / "atm_term_structure.csv", index=False)

    ev = event_study(df, earn)
    ev.to_csv(out / "events.csv", index=False)
    em = earnings_moves(df, earn)

    # models: fit on 2005+ (post dot-com; TTWO was a small cap before)
    dd = df.loc["2005":]
    # remove scheduled-event days from the "normal day" model so the launch can be added explicitly
    normal = dd.drop(index=[t for t in earn if t in dd.index], errors="ignore")
    fc = []
    for _, row in a.iterrows():
        h = int(row.days)
        g, _ = garch_forecast(normal.r, h)
        gj, _ = garch_forecast(normal.r, h, o=1)
        fc.append({"exp": row.exp, "days": h, "implied_atm": row.iv, "har": har_fit_forecast(normal.rv, h), "garch_t": g,
                   "gjr_t": gj, "kalman_logrv": kalman_logrv(normal.rv.iloc[-3000:], h),
                   "realized_22d": float(df.r.iloc[-22:].std() * np.sqrt(252)),
                   "realized_252d": float(df.r.iloc[-252:].std() * np.sqrt(252))})
    F = pd.DataFrame(fc)
    # add scheduled jumps: launch (history) and earnings (~Feb 5 2027, ~May 2027, ~Aug, ~Nov) to the model forecast
    L5 = ev[ev.kind == "launch_rockstar"]
    launch_sd = float(np.sqrt((L5.car_m2_p2 ** 2).mean()))  # 5-session window around launch (abnormal)
    earn_sd = em["sd"]
    earn_dates = pd.to_datetime(["2027-02-04", "2027-05-20", "2027-08-05", "2027-11-04"])
    rows = []
    for _, f in F.iterrows():
        e = pd.Timestamp(f.exp)
        base = np.nanmean([f.har, f.gjr_t, f.garch_t])
        var = base ** 2 * f.days / 252
        nj = int((earn_dates <= e).sum())
        var_e = var + nj * earn_sd ** 2
        var_l = var_e + (launch_sd ** 2 if e >= LAUNCH else 0)
        rows.append({"model_ex_events": base, "model_plus_earnings": np.sqrt(var_e * 252 / f.days),
                     "model_plus_earnings_and_launch_history": np.sqrt(var_l * 252 / f.days)})
    F = pd.concat([F, pd.DataFrame(rows)], axis=1)
    F["implied_minus_model_pts"] = (F.implied_atm - F.model_plus_earnings_and_launch_history) * 100
    F.to_csv(out / "vol_forecasts_vs_implied.csv", index=False)

    base_vol = float(F.iloc[0].model_ex_events)
    ie = implied_event(a, base_vol)
    # chi-square: are past launch-window moves consistent with the launch move the options price now (in vol units)?
    # Compare standardised moves: past 5-day abnormal launch moves / pre-event vol vs today's implied 5-day sd / base vol.
    L5 = ev[ev.kind == "launch_rockstar"].copy()
    L5["z_vs_today_implied"] = L5.car_m2_p2 / (ie["total_sd_pct"] / 100)
    # scale-free version: each past window's move relative to its own pre-event normal vol, vs today's ratio
    ratio_today = (ie["total_sd_pct"] / 100) / (base_vol * np.sqrt(5 / 252))
    L5["ratio_past"] = np.abs(L5.car_m2_p2) / (L5.vol_base * np.sqrt(5 / 252))
    chi = float((L5.z_vs_today_implied ** 2).sum())
    p_small = float(stats.chi2.cdf(chi, df=len(L5)))  # chance of moves this small if implied sd were right
    chi_ratio = float(((L5.car_m2_p2 / (L5.vol_base * np.sqrt(5 / 252) * ratio_today)) ** 2).sum())
    p_small_ratio = float(stats.chi2.cdf(chi_ratio, df=len(L5)))
    # all big launches (Rockstar + other), scale-free
    LA = ev[ev.kind.str.startswith("launch")]
    chi_all = float(((LA.car_m2_p2 / (LA.vol_base * np.sqrt(5 / 252) * ratio_today)) ** 2).sum())
    p_small_all = float(stats.chi2.cdf(chi_all, df=len(LA)))
    # replay: today's Nov 20 straddle (mid as % of spot) against each past launch's -3..+1 session raw move
    front = a.iloc[0]
    prem_mid, prem_ask, prem_bid = front.strad_mid / S, front.strad_ask / S, front.strad_bid / S
    LA2 = ev[ev.kind.str.startswith("launch")].copy()
    LA2["abs_move"] = np.abs(np.exp(LA2.raw_m3_p1) - 1)
    LA2["long_pl_pct_of_ask"] = (LA2.abs_move - prem_ask) / prem_ask * 100
    LA2["vol_scaled_move"] = LA2.abs_move / (LA2.vol_base * np.sqrt(5 / 252)) * base_vol * np.sqrt(5 / 252)
    LA2["long_pl_scaled_pct_of_ask"] = (LA2.vol_scaled_move - prem_ask) / prem_ask * 100
    replay = LA2[["kind", "date", "event", "raw_m3_p1", "abs_move", "long_pl_pct_of_ask", "vol_scaled_move",
                  "long_pl_scaled_pct_of_ask"]]
    replay.to_csv(out / "launch_straddle_replay.csv", index=False)
    # placebo: random non-event 5-day windows, standardised by trailing 60d vol, scaled to today's implied ratio:
    # how often do 6 such windows have a sum of squares as small as the 6 Rockstar launches?
    r5 = df.r.rolling(5).sum()
    tv = df.r.rolling(60).std().shift(5) * np.sqrt(5)
    z = (r5 / tv).loc["2005":].dropna()
    excl = set()
    for t in list(earn) + [pd.Timestamp(x[1]) for x in EVENTS]:
        excl |= set(pd.bdate_range(t - pd.Timedelta(days=3), t + pd.Timedelta(days=9)))
    z = z[~z.index.isin(excl)].values
    obs = float(((L5.raw_m3_p1 / (L5.vol_base / np.sqrt(252) * np.sqrt(5)) / ratio_today) ** 2).sum())
    sims = (RNG.choice(z, size=(100000, len(L5))) ** 2).sum(1)
    p_placebo = float((sims <= obs).mean())
    # same for "is the launch week even calmer than a normal week?" (ratio 1)
    obs1 = float(((L5.raw_m3_p1 / (L5.vol_base / np.sqrt(252) * np.sqrt(5))) ** 2).sum())
    p_calm = float((sims <= obs1).mean())
    jump_draws = L5.car_m2_p2.values

    qlike = oos_qlike(df.loc["2005":], h=5)

    model_vol = dict(zip(F.exp, F.model_ex_events))
    sim = simulate(df, fx, a, model_vol, jump_draws, S, float(fx.iloc[-1]))

    news = [json.loads(x) for x in (d / "news_TTWO.jsonl").read_text().splitlines()] if (d / "news_TTWO.jsonl").exists() else []

    res = {"asof": ASOF.date().isoformat(), "spot": S, "usdcad": float(fx.iloc[-1]),
           "implied_event": ie, "ratio_today_implied_to_normal_5d": float(ratio_today),
           "launch_test": {"n_rockstar": int(len(L5)), "chi2_abs_units": chi, "p_moves_this_small_abs": p_small,
                           "chi2_vol_units": chi_ratio, "p_moves_this_small_vol_units": p_small_ratio,
                           "n_all_launches": int(len(LA)), "p_all_launches_vol_units": p_small_all,
                           "rockstar_windows": L5[["date", "event", "car_m2_p2", "vol_base", "ratio_past"]].to_dict("records"),
                           "launch_sd_hist_5d": launch_sd, "p_placebo_empirical": p_placebo,
                           "p_launch_weeks_calmer_than_normal_weeks": p_calm,
                           "premium_mid_ask_bid_pct": [prem_mid * 100, prem_ask * 100, prem_bid * 100],
                           "replay_mean_long_pl_pct_of_ask_rockstar": float(LA2[LA2.kind == "launch_rockstar"].long_pl_pct_of_ask.mean()),
                           "replay_mean_long_pl_pct_of_ask_all": float(LA2.long_pl_pct_of_ask.mean()),
                           "replay_wins_all": int((LA2.long_pl_pct_of_ask > 0).sum()), "replay_n_all": int(len(LA2))},
           "post_launch_drift": {c: {"mean": float(L5[c].mean()), "t": float(L5[c].mean() / (L5[c].std() / np.sqrt(len(L5)))),
                                     "values": [round(float(v), 3) for v in L5[c]]}
                                 for c in ("car_m60_m1", "car_m20_m1", "car_0_20", "car_0_60", "car_0_250")},
           "current_runup": current_runup(df),
           "earnings_moves": em, "oos_qlike_5d": qlike, "simulation": sim,
           "news_word_score": {r["day"]: r["words"] for r in news}}
    (out / "results.json").write_text(json.dumps(res, indent=1, default=float))
    print(json.dumps({k: res[k] for k in ("implied_event", "ratio_today_implied_to_normal_5d", "launch_test")}, indent=1, default=float))


if __name__ == "__main__":
    main()
