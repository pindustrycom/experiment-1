"""Round 6: the volatility trade through the Greeks. Same 66 first-of-month Deribit chains and books as vrp_variants.py.

Part 1, where the money comes from. Each month's short, delta-hedged book is split hour by hour (Black-76 at each
leg's entry IV) into
  theta   the time value the sold options lose (the seller's income)
  gamma   -1/2 Gamma dS^2, the cost of re-hedging after moves (buy high, sell low)
  slip    (hedge held - book delta) dS, what the 0.1 band leaves unhedged
  edge    premium at the bid minus model value at Deribit's mark IV (the spread paid on entry)
plus fees, hedge costs and a small discretisation residual. Theta + gamma is the "implied vs realized" bet of the
market-maker story: it is positive when BTC moves less than the IV that was sold.

Part 2, vega risk and exits. Mid-month the book is marked with every leg's entry IV scaled by DVOL / DVOL at entry
(a proportional shift; Deribit's 30-day index stands in for the wing IVs, which we only see on the 1st). Buying back
costs the model price plus each leg's entry half-spread (relative) plus Deribit's fee. Exit rules are checked once a
day at 00:00 UTC, the book is then flat until the next month:
  iv_jumpX  DVOL has risen X vol points above entry           (vega stop)
  stopL     the trade's running loss reaches L x the premium   (classic option-seller stop)
  tpP       the running gain reaches P x the premium           (take profit, buy back cheap)
  rv_breach trailing 3-day realized vol above 1.5 x entry ATM IV (gamma is beating theta)

    python greeks.py DATA_DIR OUT_DIR GATE_CSV      (GATE_CSV = vrp_variants_trades.csv, for iv2 and rough_f30)

P&L is in variance-swap strike units, as in rounds 3-5; t-stats use months with the gate off counted as zero.
"""
from __future__ import annotations

import json
import sys

import numpy as np
import pandas as pd
from scipy.stats import norm

import evaluate as ev
import run
from vrp_trade import bs, delivery_prices, load_chain
from vrp_variants import DLV_FEE, FEE, FEE_CAP, HEDGE_BP, entry, legs, stats

BOOKS = ["call_strip", "strangle25", "strip"]
BAND = 0.10
RULES = {"hold": None, "iv_jump5": ("iv", 5), "iv_jump10": ("iv", 10), "iv_jump15": ("iv", 15), "stop1": ("stop", 1.0),
         "stop2": ("stop", 2.0), "tp50": ("tp", 0.5), "tp75": ("tp", 0.75), "rv_breach": ("rv", 1.5),
         "tp50_stop2": ("tpstop", (0.5, 2.0))}


def greeks(F, K, tau, sig, kind):
    """Black-76, zero rates: price, delta, gamma, theta per year (< 0), vega per unit vol."""
    sq = sig * np.sqrt(tau)
    d1 = (np.log(F / K) + 0.5 * sq ** 2) / sq
    d2 = d1 - sq
    call = kind == "call"
    price = np.where(call, F * norm.cdf(d1) - K * norm.cdf(d2), K * norm.cdf(-d2) - F * norm.cdf(-d1))
    delta = np.where(call, norm.cdf(d1), norm.cdf(d1) - 1)
    pdf = norm.pdf(d1)
    return price, delta, pdf / (F * sq), -F * pdf * sig / (2 * np.sqrt(tau)), F * pdf * np.sqrt(tau)


def setup(e, b, dvol):
    K, kind, w = b.strike_price.to_numpy(), b.type.to_numpy(), b.w.to_numpy()
    sig = b.mark_iv.to_numpy() / 100
    S0, T, path, times, exp = e["S0"], e["T"], e["path"], e["times"], e["exp"]
    _, _, vega = bs(S0, K, T, sig, kind)
    nv = float((w * vega).sum())
    if nv <= 0:
        return None
    unit = nv / (2 * S0 * norm.pdf(0.5 * e["sig_atm"] * np.sqrt(T)) * np.sqrt(T))
    px = np.where(w > 0, b.bid_price_usd, b.ask_price_usd)
    mid = b.mid_usd.to_numpy()
    tau = np.maximum(((exp - times) / pd.Timedelta(days=365)).to_numpy(), 1e-6)
    dv = dvol.reindex(times - pd.Timedelta(hours=1)).ffill().bfill().to_numpy()
    return {"K": K, "kind": kind, "w": w, "sig": sig, "S0": S0, "path": path, "times": times, "tau": tau, "nv": nv,
            "unit": unit, "prem": float((w * px).sum()), "half": np.abs(np.asarray(b.ask_price_usd) - mid) / np.maximum(mid, 1e-9),
            "fee_in": float((np.abs(w) * np.minimum(FEE * e["idx"], FEE_CAP * px)).sum()),
            "model0": float((w * greeks(S0, K, T, sig, kind)[0]).sum()), "dv": dv, "scale": nv * e["sig_atm"] / 2}


def decompose(e, s):
    """Part 1: hold to expiry, entry IVs, band hedge checked hourly (identical to vrp_variants.book_pnl)."""
    K, kind, w, sig, path, tau = s["K"], s["kind"], s["w"], s["sig"], s["path"], s["tau"]
    _, dl, gm, th, _ = greeks(path[:-1, None], K[None, :], tau[:-1, None], sig[None, :], kind[None, :])
    target = (dl * w[None, :]).sum(1)
    cur, held, traded = 0.0, np.empty(len(target)), 0.0
    for i, h in enumerate(target):
        if i == 0 or abs(h - cur) > BAND * s["unit"]:
            traded += abs(h - cur) * path[i]
            cur = h
        held[i] = cur
    traded += abs(cur) * path[-1]
    dS, dt = np.diff(path), -np.diff(tau)
    theta = float(-(th * w[None, :]).sum(1) @ dt)
    gamma = float(-0.5 * (gm * w[None, :]).sum(1) @ dS ** 2)
    slip = float(((held - target) * dS).sum())
    payoff = np.where(kind == "call", np.maximum(e["ST"] - K, 0), np.maximum(K - e["ST"], 0))
    fees = s["fee_in"] + float((np.abs(w) * np.where(payoff > 0, np.minimum(DLV_FEE * e["ST"], FEE_CAP * payoff), 0)).sum())
    hcost = traded * HEDGE_BP / 1e4
    total = s["prem"] - float((w * payoff).sum()) + float((held * dS).sum()) - hcost - fees
    edge = s["prem"] - s["model0"]
    out = {"theta": theta, "gamma": gamma, "slip": slip, "edge": edge, "fees": -fees, "hedge_cost": -hcost, "total": total}
    out["residual"] = total - sum(out[k] for k in ("theta", "gamma", "slip", "edge", "fees", "hedge_cost"))
    # theta per day per BTC of capital (one straddle-equivalent of vega per BTC), first day, in USD
    out = {k: v / s["scale"] for k, v in out.items()}
    out["theta_usd_day1_per_btc"] = float(-(th[0] * w).sum() / 365 / s["unit"])
    return out


def run_exit(e, s, rule):
    """Part 2: same band-hedged book, marked with DVOL-scaled IVs, closed when the rule fires at a 00:00 UTC check."""
    K, kind, w, sig, path, tau, dv = s["K"], s["kind"], s["w"], s["sig"], s["path"], s["tau"], s["dv"]
    n = len(path) - 1
    _, dl, _, _, _ = greeks(path[:-1, None], K[None, :], tau[:-1, None], sig[None, :], kind[None, :])
    target = (dl * w[None, :]).sum(1)
    r2 = np.r_[0, np.diff(np.log(path))] ** 2
    cur, traded, hpnl = 0.0, 0.0, 0.0
    worst = 0.0
    for i in range(n):
        if i > 0 and i % 24 == 0:
            sh = np.maximum(sig * dv[i] / dv[0], 0.05)
            p = greeks(path[i], K, tau[i], sh, kind)[0]
            close = float((w * p * (1 + np.sign(w) * s["half"])).sum())
            run_pnl = s["prem"] - close + hpnl - traded * HEDGE_BP / 1e4
            worst = min(worst, run_pnl)
            fire = False
            if rule is not None:
                kind_r, x = rule
                if kind_r == "iv":
                    fire = dv[i] - dv[0] >= x
                elif kind_r == "stop":
                    fire = run_pnl <= -x * s["prem"]
                elif kind_r == "tp":
                    fire = run_pnl >= x * s["prem"]
                elif kind_r == "tpstop":
                    fire = run_pnl >= x[0] * s["prem"] or run_pnl <= -x[1] * s["prem"]
                elif kind_r == "rv" and i >= 72:
                    fire = np.sqrt(r2[i - 71:i + 1].sum() * 365 * 24 / 72) > x * e["sig_atm"]
            if fire:
                fees = s["fee_in"] + float((np.abs(w) * np.minimum(FEE * path[i], FEE_CAP * np.abs(p))).sum())
                traded += abs(cur) * path[i]
                pnl = s["prem"] - close + hpnl - traded * HEDGE_BP / 1e4 - fees
                return {"pnl": pnl / s["scale"], "ret": pnl / (s["unit"] * s["S0"]), "exit_day": i // 24,
                        "worst_mtm": worst / s["scale"]}
        h = target[i]
        if i == 0 or abs(h - cur) > BAND * s["unit"]:
            traded += abs(h - cur) * path[i]
            cur = h
        hpnl += cur * (path[i + 1] - path[i])
    payoff = np.where(kind == "call", np.maximum(e["ST"] - K, 0), np.maximum(K - e["ST"], 0))
    fees = s["fee_in"] + float((np.abs(w) * np.where(payoff > 0, np.minimum(DLV_FEE * e["ST"], FEE_CAP * payoff), 0)).sum())
    traded += abs(cur) * path[-1]
    pnl = s["prem"] - float((w * payoff).sum()) + hpnl - traded * HEDGE_BP / 1e4 - fees
    return {"pnl": pnl / s["scale"], "ret": pnl / (s["unit"] * s["S0"]), "exit_day": None, "worst_mtm": min(worst, pnl) / s["scale"]}


def main(folder, out, gate_csv):
    d, _ = run.prepare(folder)
    c = load_chain(folder)
    dlv = delivery_prices(folder, d)
    g = pd.read_csv(gate_csv).drop_duplicates("date").set_index("date")
    gate = (g.iv2 > g.rough_f30).astype(float)
    dec, ex = [], []
    for day in sorted(c.date.unique()):
        e = entry(c, pd.Timestamp(day), d, dlv)
        if e is None:
            continue
        ds = str(e["day"].date())
        dv0 = float(d.dvol.asof(e["day"] - pd.Timedelta(hours=1)))
        rv = np.sqrt(e["rv_ann"])
        for book in BOOKS:
            b = legs(e, book)
            if b is None:
                continue
            s = setup(e, b, d.dvol)
            if s is None:
                continue
            base = {"date": ds, "book": book, "gate": gate.get(ds, 0.0), "iv_atm": e["sig_atm"], "rv": rv, "dvol0": dv0,
                    "dvol_max_jump": float(np.nanmax(s["dv"]) - s["dv"][0])}
            dec.append({**base, **decompose(e, s)})
            for name, rule in RULES.items():
                ex.append({**base, "rule": name, **run_exit(e, s, rule)})
    dec, ex = pd.DataFrame(dec), pd.DataFrame(ex)
    dec.to_csv(f"{out}/greeks_decomp.csv", index=False)
    ex.to_csv(f"{out}/greeks_exits.csv", index=False)

    parts = ["theta", "gamma", "slip", "edge", "fees", "hedge_cost", "residual", "total"]
    summary = {"decomposition": {}, "exits": [], "vega": {}}
    for book in BOOKS:
        x = dec[dec.book == book]
        for tm, sel in (("always", x.gate >= 0), ("gate_on", x.gate > 0), ("gate_off", x.gate == 0)):
            summary["decomposition"][f"{book}_{tm}"] = {"months": int(sel.sum()), **x[sel][parts].mean().round(4).to_dict(),
                                                        "theta_usd_day1_per_btc": round(float(x[sel].theta_usd_day1_per_btc.mean()), 1)}
        # theta + gamma against implied minus realized variance (the market-maker story, month by month)
        tg = x.theta + x.gamma
        vp = (x.iv_atm ** 2 - x.rv ** 2) / x.iv_atm ** 2
        summary["decomposition"][f"{book}_corr_theta_gamma_vs_iv2_rv2"] = round(float(np.corrcoef(tg, vp)[0, 1]), 3)
    rows = []
    for (book, rule), x in ex.groupby(["book", "rule"]):
        x = x.set_index("date")
        for tm in ("always", "gate"):
            pnl = x.pnl * (x.gate if tm == "gate" else 1.0)
            ret = x.ret * (x.gate if tm == "gate" else 1.0)
            on = x.gate > 0 if tm == "gate" else x.gate >= 0
            st = stats(pnl.to_numpy())
            eq = (1 + ret).cumprod()
            rows.append({"book": book, "rule": rule, "timing": tm, **st, "exits": int(x[on].exit_day.notna().sum()),
                         "traded": int(on.sum()), "worst_mtm": float(x[on].worst_mtm.min()),
                         "cagr": float(eq.iloc[-1] ** (12 / len(ret)) - 1), "max_dd": float((eq / eq.cummax() - 1).min()),
                         "mean_2025_26": float(pnl[pnl.index >= "2025-01-01"].mean())})
    r = pd.DataFrame(rows)
    r["p_holm"] = ev.holm(r.p.to_numpy())
    # does each exit rule beat holding, month by month (paired)?
    for i, row in r.iterrows():
        a = ex[(ex.book == row.book) & (ex.rule == row.rule)].set_index("date")
        h = ex[(ex.book == row.book) & (ex.rule == "hold")].set_index("date")
        w_ = a.gate if row.timing == "gate" else 1.0
        diff = (a.pnl - h.pnl) * w_
        r.loc[i, "vs_hold_mean"] = float(diff.mean())
        r.loc[i, "vs_hold_t"] = float(diff.mean() / diff.std(ddof=1) * np.sqrt(len(diff))) if diff.std() > 0 else 0.0
    r = r.sort_values(["book", "timing", "t"], ascending=[True, True, False])
    r.to_csv(f"{out}/greeks_exits_summary.csv", index=False)
    summary["exits"] = r.round(4).to_dict("records")
    h = ex[ex.rule == "hold"]
    for book in BOOKS:
        x = h[h.book == book]
        summary["vega"][book] = {"corr_pnl_vs_dvol_max_jump": round(float(np.corrcoef(x.pnl, x.dvol_max_jump)[0, 1]), 3),
                                 "worst_mtm_always": round(float(x.worst_mtm.min()), 3),
                                 "worst_mtm_gate": round(float(x[x.gate > 0].worst_mtm.min()), 3),
                                 "months_dvol_up10": int((x.dvol_max_jump >= 10).sum()),
                                 "gate_on_months_dvol_up10": int(((x.dvol_max_jump >= 10) & (x.gate > 0)).sum())}
    with open(f"{out}/greeks.json", "w") as fh:
        json.dump(summary, fh, indent=1, default=float)
    pd.set_option("display.width", 250)
    print(json.dumps({k: summary[k] for k in ("decomposition", "vega")}, indent=1))
    cols = ["book", "rule", "timing", "traded", "exits", "mean", "t", "p_holm", "vs_hold_mean", "vs_hold_t", "hit", "worst",
            "worst_mtm", "cagr", "max_dd", "mean_2025_26"]
    print(r[cols].round(3).to_string())


if __name__ == "__main__":
    main(*sys.argv[1:4])
