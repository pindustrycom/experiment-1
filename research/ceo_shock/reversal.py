"""Follow-up: does it pay to buy once a key-person shock is resolved? (paper research only)

Events (reversals.csv), three kinds:
  undone  - the shock itself is reversed: settlement, reinstatement, a founder backing off (the NVIDIA case type)
  filled  - a credible new CEO named after a shock exit
  removed - the tainted person leaves after a scandal
Entry at the day-0 open (the news is out before the open for most of these) or at the day-0 close; hold 20 days,
10 bp round trip, compare with SPY. Same abnormal-return model as analysis.py. Baseline: all PIT S&P 500 member-days
with an abnormal gain of 3%+ (ordinary good-news days), for the "does a big up day keep going" question.

    python reversal.py SP500_DIR OUT_DIR
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

from analysis import COST_RT, H, HOLD, betas, car_table, holm, load, resolve


def main(sp: Path, out: Path):
    close, spy, member = load(sp)
    px = pd.read_parquet(sp / "prices.parquet", columns=["date", "ticker", "open", "close", "adj_close"],
                         filters=[("date", ">=", pd.Timestamp("1995-01-01"))]).drop_duplicates(["date", "ticker"], keep="last")
    px["adj_open"] = px["open"] * px["adj_close"] / px["close"]
    opn = px.pivot(index="date", columns="ticker", values="adj_open").reindex(index=close.index)
    spy_open = opn["SPY"] if "SPY" in opn else None
    r = close.pct_change(fill_method=None)
    m = spy.pct_change(fill_method=None).reindex(r.index)
    ar = (r - betas(r, m).mul(m, axis=0)).where(lambda x: x.abs() < 0.9)

    ev = pd.read_csv(Path(__file__).with_name("reversals.csv"), parse_dates=["date"])
    ev["sym"] = [resolve(t, d, close) for t, d in zip(ev["ticker"], ev["date"])]
    missing = ev[ev["sym"].isna()][["ticker", "date", "event"]]
    ev = ev.dropna(subset=["sym"]).reset_index(drop=True)
    ev["date"] = [close.index[close.index.searchsorted(d)] for d in ev["date"]]
    ct = car_table(ar, list(zip(ev["date"], ev["sym"])))
    ev = pd.concat([ev, ct.drop(columns=["date", "ticker"])], axis=1)
    sp_c = spy.reindex(close.index)
    rows = []
    for _, e in ev.iterrows():
        i = close.index.get_loc(e["date"])
        s, o = close[e["sym"]], opn[e["sym"]]
        x = {}
        if i + HOLD < len(close):
            x["open_entry"] = s.iloc[i + HOLD] / o.iloc[i] - 1 - COST_RT
            x["close_entry"] = s.iloc[i + HOLD] / s.iloc[i] - 1 - COST_RT
            x["spy_from_open"] = sp_c.iloc[i + HOLD] / spy_open.iloc[i] - 1
            x["spy_from_close"] = sp_c.iloc[i + HOLD] / sp_c.iloc[i] - 1
            x["gap_open"] = o.iloc[i] / s.iloc[i - 1] - 1
        rows.append(x)
    ev = pd.concat([ev, pd.DataFrame(rows)], axis=1)
    ev["open_excess"] = ev["open_entry"] - ev["spy_from_open"]
    ev["close_excess"] = ev["close_entry"] - ev["spy_from_close"]
    ev.to_csv(out / "reversals_car.csv", index=False, float_format="%.5f")

    # baseline: ordinary 3%+ abnormal up days
    A = ar.where(member).loc["1996-01-01":]
    st = A.stack()
    st = st[st >= 0.03]
    base = car_table(ar, list(st.index)).dropna(subset=["car20"])
    base["crisis_year"] = base["date"].dt.year.isin([2000, 2001, 2002, 2008, 2009, 2020])
    bsum = base.groupby("crisis_year")[[f"car{h}" for h in H]].mean()

    res = {"missing": missing.astype(str).to_dict("records"), "baseline_up_days": int(len(base)),
           "baseline_mean_car": {str(k): {c: float(v) for c, v in row.items()} for k, row in bsum.iterrows()},
           "groups": {}}
    pv = {}
    for name, g in [("undone", ev[ev["type"] == "undone"]), ("all_resolutions", ev[ev["type"] != "live"])]:
        d = {"n": int(len(g)), "mean_ar0": g["ar0"].mean(), "mean_gap_open": g["gap_open"].mean()}
        for h in H:
            c = g[f"car{h}"].dropna()
            t, p = stats.ttest_1samp(c, 0.0) if len(c) > 2 else (np.nan, np.nan)
            d[f"car{h}"] = {"mean": c.mean(), "median": c.median(), "up": float((c > 0).mean()), "t": t, "p": p}
            pv[f"{name}_car{h}"] = p
        for k in ["open_excess", "close_excess"]:
            c = g[k].dropna()
            t, p = stats.ttest_1samp(c, 0.0)
            d[k] = {"n": int(len(c)), "mean": c.mean(), "median": c.median(), "beat_spy": float((c > 0).mean()),
                    "t": t, "p": p}
            pv[f"{name}_{k}"] = p
        res["groups"][name] = d
    hp = holm({k: v for k, v in pv.items() if np.isfinite(v)})
    res["holm"] = hp
    res["by_type_close_excess"] = {k: {"n": int(len(g)), "mean": g["close_excess"].mean()}
                                   for k, g in ev.groupby("type")}
    (out / "reversals.json").write_text(json.dumps(res, indent=1, default=float))
    cols = ["sym", "date", "type", "gap_open", "ar0", "car5", "car20", "car60", "open_excess", "close_excess"]
    print(ev[cols].to_string(float_format=lambda x: f"{x * 100:+.1f}"))
    print(json.dumps(res, indent=1, default=float))


if __name__ == "__main__":
    main(Path(sys.argv[1]), Path(sys.argv[2]))
