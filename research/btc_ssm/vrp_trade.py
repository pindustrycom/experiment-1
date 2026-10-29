"""Tradeable version of the BTC volatility trade: sell ~30-day Deribit options at the real bid on the 1st of each month,
delta-hedge daily, hold to expiry. Compared with the synthetic variance swap on the same dates, to see how much of the
variance risk premium survives real prices and costs.

    python vrp_trade.py DATA_DIR OUT_DIR

Two books per month, both short options sold at bid (or at mid, to split the cost):
  straddle: the call and put at the strike nearest the forward
  strip:    out-of-the-money puts below / calls above the forward, weighted dK / K^2 (the variance-swap replication),
            strikes from 0.6 F to 1.6 F that have a bid
Deribit options are coin-settled; the books are valued in USD with the premium converted at entry and the payoff paid
at Deribit's delivery price. The delta hedge is rebalanced daily at 00:00 UTC (Black-Scholes delta at the entry IV) in
BTC perpetuals at 5 bp per trade. Each month's P&L is reported in variance-swap strike units, P&L / (vega * sigma / 2)
(directly comparable with (K - RV) / K), and as a return on capital equal to one BTC of notional per unit of straddle.
"""
from __future__ import annotations

import json
import sys

import numpy as np
import pandas as pd
from scipy.stats import norm

import evaluate as ev
import run

HEDGE_BP = 5.0
H = 24


def bs(F, K, T, sig, kind):
    """Black-76 with zero rates (Deribit quotes forwards): price in USD, delta, vega per unit vol."""
    sq = sig * np.sqrt(T)
    d1 = (np.log(F / K) + 0.5 * sq ** 2) / sq
    d2 = d1 - sq
    call = kind == "call"
    price = np.where(call, F * norm.cdf(d1) - K * norm.cdf(d2), K * norm.cdf(-d2) - F * norm.cdf(-d1))
    delta = np.where(call, norm.cdf(d1), norm.cdf(d1) - 1)
    return price, delta, F * norm.pdf(d1) * np.sqrt(T)


def load_chain(folder):
    c = pd.read_csv(f"{folder}/options_1st.csv.gz")
    c["date"] = pd.to_datetime(c.date)
    c["exp"] = pd.to_datetime(c.expiration, unit="us")
    c["T"] = (c.exp - c.date) / pd.Timedelta(days=365)
    for k in ("bid_price", "ask_price", "mark_price", "mark_iv", "bid_iv", "ask_iv", "underlying_price", "strike_price"):
        c[k] = pd.to_numeric(c[k], errors="coerce")
    return c


def delivery_prices(folder, d):
    try:
        dv = pd.read_csv(f"{folder}/delivery.csv")
        return pd.Series(dv.delivery_price.to_numpy(), index=pd.to_datetime(dv.date)).sort_index()
    except FileNotFoundError:
        return pd.Series(dtype=float)


def month_trade(c, day, d, dlv):
    snap = c[c.date == day]
    exps = snap.groupby("exp").T.first()
    exps = exps[(exps * 365 >= 18) & (exps * 365 <= 45)]
    if exps.empty:
        return None
    exp = (exps * 365 - 30).abs().idxmin()
    ch = snap[(snap.exp == exp) & (snap.bid_price > 0) & (snap.ask_price > 0)].copy()
    if len(ch) < 6:
        return None
    S0 = float(ch.underlying_price.median())                     # the expiry's forward (Deribit's underlying)
    T = float(ch["T"].iloc[0])
    idx = float(d.close.asof(day - pd.Timedelta(hours=1)))       # spot at 00:00 UTC for converting coin prices to USD
    for k in ("bid_price", "ask_price"):
        ch[k + "_usd"] = ch[k] * idx
    ch["mid_usd"] = 0.5 * (ch.bid_price_usd + ch.ask_price_usd)
    atm_k = ch.strike_price.iloc[(ch.strike_price - S0).abs().argmin()]
    atm = ch[ch.strike_price == atm_k]
    if set(atm.type) != {"call", "put"}:
        return None
    sig_atm = float(atm.mark_iv.mean()) / 100
    # settlement
    exp_day = exp.normalize()
    ST = float(dlv.get(exp_day, np.nan)) if len(dlv) else np.nan
    if not np.isfinite(ST):
        ST = float(d.close.asof(exp - pd.Timedelta(hours=1)))  # Binance close of the 07:00 bar = 08:00 UTC
    # hedge path: daily 00:00 UTC closes from the entry to expiry (08:00)
    path_t = list(pd.date_range(day, exp_day, freq="D")) + [exp]
    path = np.array([float(d.close.asof(t - pd.Timedelta(hours=1))) for t in path_t])   # close of the bar ending at t
    if not np.isfinite(path).all():
        return None
    # realized variance over the life (hourly squared log returns), annualized
    hr = d.r.loc[day:exp - pd.Timedelta(hours=1)]
    rv_ann = float((hr / 100).pow(2).sum() / T)
    books = {}
    strip = ch[((ch.type == "put") & (ch.strike_price < S0)) | ((ch.type == "call") & (ch.strike_price >= S0))]
    strip = strip[(strip.strike_price >= 0.6 * S0) & (strip.strike_price <= 1.6 * S0)].sort_values("strike_price")
    ks = strip.strike_price.to_numpy()
    if len(ks) >= 4:
        dk = np.gradient(ks)
        strip = strip.assign(w=dk / ks ** 2 * S0 ** 2 / 2)   # scaled so the book's vega is comparable to one straddle
        books["strip"] = strip
    books["straddle"] = atm.assign(w=1.0)
    out = {"date": str(day.date()), "expiry": str(exp), "days": round(T * 365, 1), "F0": S0, "ST": ST, "atm_iv": sig_atm,
           "rv": float(np.sqrt(rv_ann)), "dvol": float(d.dvol.asof(day - pd.Timedelta(hours=1))) / 100}
    for name, b in books.items():
        K, kind, w = b.strike_price.to_numpy(), b.type.to_numpy(), b.w.to_numpy()
        sig = b.mark_iv.to_numpy() / 100
        payoff = np.where(kind == "call", np.maximum(ST - K, 0), np.maximum(K - ST, 0))
        _, _, vega = bs(S0, K, T, sig, kind)
        hedge, hcost, h_prev = 0.0, 0.0, 0.0
        for i in range(len(path_t) - 1):
            tau = max((exp - path_t[i]) / pd.Timedelta(days=365), 1e-6)
            _, dl, _ = bs(path[i], K, tau, sig, kind)
            h = float((w * dl).sum())                    # short options -> hold +h BTC to be flat delta
            hcost += abs(h - h_prev) * path[i] * HEDGE_BP / 1e4
            hedge += h * (path[i + 1] - path[i])
            h_prev = h
        hcost += abs(h_prev) * path[-1] * HEDGE_BP / 1e4
        settle = float((w * payoff).sum())
        nv = float((w * vega).sum())
        for fill in ("bid", "mid"):
            prem = float((w * (b.bid_price_usd if fill == "bid" else b.mid_usd)).sum())
            pnl = prem - settle + hedge - hcost
            out[f"{name}_{fill}"] = pnl / (nv * sig_atm / 2)      # strike units, like (K - RV)/K
            out[f"{name}_{fill}_ret"] = pnl / S0                    # capital: 1 BTC per straddle
        out[f"{name}_spread_cost"] = float((w * (b.mid_usd - b.bid_price_usd)).sum()) / (nv * sig_atm / 2)
        out[f"{name}_hedge_cost"] = hcost / (nv * sig_atm / 2)
        out[f"{name}_n"] = len(b)
    out["synthetic_atm"] = (sig_atm ** 2 - rv_ann) / sig_atm ** 2
    out["synthetic_dvol"] = (out["dvol"] ** 2 - rv_ann) / out["dvol"] ** 2 if np.isfinite(out["dvol"]) else np.nan
    return out


def summarize(t, d):
    res = {}
    cols = ["synthetic_dvol", "synthetic_atm", "straddle_mid", "straddle_bid", "strip_mid", "strip_bid"]
    for c in cols:
        x = t[c].dropna().to_numpy()
        res[c] = {"n": len(x), "mean": float(x.mean()), "median": float(np.median(x)), "t": float(x.mean() / x.std(ddof=1) * np.sqrt(len(x))),
                  "hit": float((x > 0).mean()), "worst": float(x.min()), "best": float(x.max()),
                  "sharpe_ann": float(x.mean() / x.std(ddof=1) * np.sqrt(12))}
    for b in ("straddle", "strip"):
        res[f"{b}_costs"] = {"spread": float(t[f"{b}_spread_cost"].mean()), "hedge": float(t[f"{b}_hedge_cost"].mean())}
    # capital view: one BTC of straddle notional per BTC of capital, monthly
    r = t.set_index(pd.to_datetime(t.date)).straddle_bid_ret
    eq = (1 + r).cumprod()
    res["straddle_capital"] = {"cagr": float(eq.iloc[-1] ** (12 / len(r)) - 1), "sharpe": float(r.mean() / r.std() * np.sqrt(12)),
                               "max_dd": float((eq / eq.cummax() - 1).min()), "worst_month": float(r.min()),
                               "mean_month": float(r.mean())}
    # Donchian over the same holding windows
    don, _ = ev.donchian_daily(d, 25, t.date.min(), t.expiry.max()[:10])
    dm = []
    for _, row in t.iterrows():
        dm.append(float(np.prod(1 + don.loc[row.date:row.expiry[:10]]) - 1))
    t["donchian"] = dm
    res["corr_straddle_vs_donchian"] = float(np.corrcoef(t.straddle_bid_ret, t.donchian)[0, 1])
    res["donchian_same_windows"] = {"mean": float(np.mean(dm)), "sharpe": float(np.mean(dm) / np.std(dm, ddof=1) * np.sqrt(12))}
    # what share of the synthetic premium survives
    res["survival_straddle"] = res["straddle_bid"]["mean"] / res["synthetic_atm"]["mean"]
    res["survival_strip"] = res["strip_bid"]["mean"] / res["synthetic_atm"]["mean"]
    return res


def main(folder, out):
    d, _ = run.prepare(folder)
    c = load_chain(folder)
    dlv = delivery_prices(folder, d)
    rows = [m for day in sorted(c.date.unique()) if (m := month_trade(c, pd.Timestamp(day), d, dlv)) is not None
            and pd.Timestamp(m["expiry"]) < d.index[-1]]
    t = pd.DataFrame(rows)
    res = summarize(t, d)
    t.to_csv(f"{out}/vrp_trades.csv", index=False)
    with open(f"{out}/vrp_results.json", "w") as fh:
        json.dump(res, fh, indent=1)
    print(t[["date", "days", "atm_iv", "rv", "synthetic_atm", "straddle_bid", "strip_bid", "straddle_bid_ret"]].round(3).to_string())
    print(json.dumps(res, indent=1))


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
