"""Item 7: the gated vol trade entered every day instead of on the 1st of the month.

    python daily_entries.py DATA_DIR OUT_DIR SV_DIR [CHAINS_DIR]

With CHAINS_DIR (the deribit-chains release, collect_chains.py), every daily snapshot is a trade: the round-4 winner
(OTM call wing 1.0-1.6 F sold at the bid, band-0.1 hedge, sold only when ATM implied variance is above the rough-vol
30-day forecast). Overlapping trades are held as a ladder (each entry 1/30 of the book), so the monthly t-stat uses
non-overlapping 30-day blocks. Without it, the free first-of-month chains are the only real prices, so the script runs
the synthetic daily-entry version on DVOL (round 2's variance swap), gated the same way, to show what daily entries
add: about 30x the trades, and a t-stat that no longer hangs on 28 months.
"""
from __future__ import annotations

import glob
import json
import sys

import numpy as np
import pandas as pd

import evaluate as ev
import run
import vrp_variants as vv
from vrp_trade import delivery_prices

H = 24


def _t(x):
    x = np.asarray(x, float)
    x = x[np.isfinite(x)]
    return float(x.mean() / x.std(ddof=1) * np.sqrt(len(x))), len(x)


def load_daily_chains(folder, coin="BTC"):
    files = sorted(glob.glob(f"{folder}/chains_{coin}_*.csv.gz"))
    if not files:
        return None
    c = pd.concat([pd.read_csv(f) for f in files])
    c["date"] = pd.to_datetime(c.date)
    c["exp"] = pd.to_datetime(c.expiration, unit="us")
    c["T"] = (c.exp - c.date) / pd.Timedelta(days=365)
    for k in ("bid_price", "ask_price", "mark_price", "mark_iv", "underlying_price", "strike_price", "delta"):
        c[k] = pd.to_numeric(c[k], errors="coerce")
    return c


def real(d, chains, sv_dir, folder, book="call_strip", hedge="band10"):
    dlv = delivery_prices(folder, d)
    fr = vv.f30(sv_dir)
    rows = []
    for day in sorted(chains.date.unique()):
        e = vv.entry(chains, pd.Timestamp(day), d, dlv)
        if e is None:
            continue
        b = vv.legs(e, book)
        r = vv.book_pnl(e, b, hedge, "bid") if b is not None else None
        if r:
            on = e["sig_atm"] ** 2 > float(fr.asof(e["day"]))
            rows.append({"date": e["day"], "on": on, "pnl": r["pnl"] * on, "ret": r["ret"] * on})
    t = pd.DataFrame(rows).set_index("date")
    blocks = t.pnl.groupby(np.arange(len(t)) // 30).mean()       # non-overlapping 30-entry blocks
    tb, nb = _t(blocks)
    return {"entries": len(t), "share_on": float(t.on.mean()), "mean": float(t.pnl.mean()), "t_blocks": tb, "blocks": nb}


def synthetic(d, sv_dir):
    """Daily entries of a 30-day variance swap struck at DVOL^2, P&L (K - RV) / K, gated by the rough-vol forecast."""
    fr = vv.f30(sv_dir)
    days = pd.date_range("2021-04-01", d.index[-1] - pd.Timedelta(days=31), freq="D")
    r2 = (d.r / 100) ** 2
    rows = []
    for day in days:
        k = float(d.dvol.asof(day - pd.Timedelta(hours=1))) / 100
        rv = float(r2.loc[day:day + pd.Timedelta(days=30) - pd.Timedelta(hours=1)].sum() * 365 / 30)
        if not np.isfinite(k):
            continue
        on = k ** 2 > float(fr.asof(day))
        rows.append({"date": day, "on": on, "pnl": (k ** 2 - rv) / k ** 2})
    t = pd.DataFrame(rows).set_index("date")
    out = {}
    for name, x in (("always", t.pnl), ("gated", t.pnl * t.on)):
        first = x[x.index.day == 1]
        tf, nf = _t(first)
        blocks = x.groupby(np.arange(len(x)) // 30).mean()
        tb, nb = _t(blocks)
        out[name] = {"mean": float(x.mean()), "first_of_month_t": tf, "first_of_month_n": nf, "daily_blocks_t": tb, "blocks": nb,
                     "daily_blocks_nw1_t": float(ev.nw_t(blocks.to_numpy(), 1))}   # adjacent blocks share some days
    out["share_on"] = float(t.on.mean())
    return out


def main(folder, out, sv_dir, chains=None):
    d, _ = run.prepare(folder)
    res = {"synthetic_daily": synthetic(d, sv_dir)}
    c = load_daily_chains(chains) if chains else None
    if c is not None and c.date.nunique() >= 60:
        res["real_daily"] = real(d, c, sv_dir, folder)
    elif c is not None:
        res["real_daily"] = f"only {c.date.nunique()} days of daily chains so far; run again after about 3 months"
    with open(f"{out}/daily_entries.json", "w") as fh:
        json.dump(res, fh, indent=1)
    print(json.dumps(res, indent=1))


if __name__ == "__main__":
    main(*sys.argv[1:5])
