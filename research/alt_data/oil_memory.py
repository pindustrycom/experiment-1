"""Quick data checks for the alternative-data write-up (paper research only).

1. Oil shocks vs tech: days when Brent jumps 5%+ in a day, or 15%+ over 5 days, since 1988.
   What do the Nasdaq 100 (^NDX) and the chip index (^SOX) do the same day and over the next 1, 5, 20, 60
   trading days, compared with an average day? Plus named geopolitical episodes (Kuwait 1990 ... Iran 2026).
2. Memory prices vs Micron: does the US producer price index for semiconductors (FRED) lead Micron's
   stock, or does the stock lead it? Monthly lead-lag correlations.

Run: python research/alt_data/oil_memory.py --out research/alt_data/results
"""
import argparse, io, json, os, urllib.request
import numpy as np
import pandas as pd
import yfinance as yf


def fred(sid):
    url = f"https://fred.stlouisfed.org/graph/fredgraph.csv?id={sid}"
    raw = urllib.request.urlopen(url, timeout=60).read().decode()
    df = pd.read_csv(io.StringIO(raw))
    df.columns = ["date", sid]
    df["date"] = pd.to_datetime(df["date"])
    df[sid] = pd.to_numeric(df[sid], errors="coerce")
    return df.set_index("date")[sid].dropna()


def yclose(t, start="1985-01-01"):
    d = yf.download(t, start=start, progress=False, auto_adjust=True)["Close"]
    return d.squeeze().dropna()


def tstat(x):
    x = np.asarray(x, float); x = x[~np.isnan(x)]
    return float(x.mean() / (x.std(ddof=1) / np.sqrt(len(x)))) if len(x) > 2 else None


def declump(dates, gap=20):
    out, last = [], None
    for d in dates:
        if last is None or (d - last).days > gap * 1.4:
            out.append(d); last = d
    return out


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--out", default="results"); a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    res = {}
    brent = fred("DCOILBRENTEU")
    px = pd.DataFrame({"brent": brent, "NDX": yclose("^NDX"), "SOX": yclose("^SOX"), "SPX": yclose("^GSPC"),
                       "QQQ": yclose("QQQ"), "SMH": yclose("SMH"), "MU": yclose("MU")}).sort_index()
    px = px[px["NDX"].notna()].ffill(limit=3)
    lr = np.log(px).diff()
    res["data_range"] = [str(px.index[0].date()), str(px.index[-1].date())]

    # 1. oil shock days
    fwd = {h: np.log(px).shift(-h) - np.log(px) for h in (1, 5, 20, 60)}
    b1 = lr["brent"]; b5 = np.log(px["brent"]).diff(5)
    shocks = {"brent_1d_up5": b1[b1 > 0.05].index, "brent_5d_up15": b5[b5 > 0.15].index,
              "brent_1d_down5": b1[b1 < -0.05].index}
    table = []
    for name, idx in shocks.items():
        idx = declump(list(idx))
        for asset in ("NDX", "SOX", "SPX"):
            row = {"shock": name, "asset": asset, "n_events": len(idx)}
            same = lr[asset].reindex(idx)
            row["same_day_pct"] = round(100 * same.mean(), 2)
            row["same_day_corr_all_days"] = round(float(lr[["brent", asset]].dropna().corr().iloc[0, 1]), 3)
            for h, f in fwd.items():
                ev = f[asset].reindex(idx).dropna()
                base = f[asset].dropna()
                row[f"fwd{h}_pct"] = round(100 * ev.mean(), 2)
                row[f"fwd{h}_base_pct"] = round(100 * base.mean(), 2)
                row[f"fwd{h}_excess_t"] = round(tstat(ev - base.mean()), 2) if len(ev) > 2 else None
                row[f"fwd{h}_up_share"] = round(float((ev > 0).mean()), 2)
            table.append(row)
    pd.DataFrame(table).to_csv(f"{a.out}/oil_shocks.csv", index=False)
    res["oil_shocks"] = table

    # rolling correlation oil vs Nasdaq by period
    per = {}
    for lo, hi in [("1988", "1999"), ("2000", "2007"), ("2008", "2014"), ("2015", "2019"), ("2020", "2021"),
                   ("2022", "2022"), ("2023", "2025"), ("2026", "2026")]:
        s = lr.loc[lo:hi, ["brent", "NDX", "SOX"]].dropna()
        w = s.resample("W").sum()
        per[f"{lo}-{hi}"] = {"daily_corr_NDX": round(float(s.corr().loc["brent", "NDX"]), 3),
                            "weekly_corr_NDX": round(float(w.corr().loc["brent", "NDX"]), 3),
                            "weekly_corr_SOX": round(float(w.corr().loc["brent", "SOX"]), 3)}
    res["oil_tech_corr_by_period"] = per

    # named episodes: start date -> Brent and NDX/SOX change over next 20 and 60 trading days
    episodes = {"Iraq invades Kuwait": "1990-08-01", "Desert Storm starts": "1991-01-16", "9/11": "2001-09-10",
                "Iraq war starts": "2003-03-19", "Arab Spring / Libya": "2011-02-15", "Abqaiq drone attack": "2019-09-13",
                "Soleimani strike": "2020-01-02", "Russia invades Ukraine": "2022-02-23",
                "Israel strikes Iran": "2025-06-12", "US strikes Iran (Midnight Hammer)": "2025-06-20"}
    try:
        war = json.load(open(os.path.join(os.path.dirname(__file__), "episodes_2026.json")))
        episodes.update(war)
    except FileNotFoundError:
        pass
    ep = []
    for k, d in episodes.items():
        d = pd.Timestamp(d)
        if d < px.index[0]:
            continue
        i = px.index.searchsorted(d); i0 = max(i, 0)
        row = {"episode": k, "date": str(px.index[i0].date())}
        for h in (5, 20, 60):
            j = min(i0 + h, len(px) - 1)
            for c in ("brent", "NDX", "SOX"):
                a0, a1 = px[c].iloc[i0], px[c].iloc[j]
                row[f"{c}_{h}d_pct"] = round(100 * (a1 / a0 - 1), 1) if pd.notna(a0) and pd.notna(a1) else None
        ep.append(row)
    pd.DataFrame(ep).to_csv(f"{a.out}/episodes.csv", index=False)
    res["episodes"] = ep

    # 2026 so far
    y = px.loc["2025-12-31":]
    res["ytd_2026"] = {c: round(100 * (y[c].dropna().iloc[-1] / y[c].dropna().iloc[0] - 1), 1) for c in y}
    res["brent_2026_max"] = [str(y["brent"].idxmax().date()), round(float(y["brent"].max()), 1)]
    res["brent_last"] = round(float(px["brent"].dropna().iloc[-1]), 1)

    # 2. memory: semiconductor PPI vs Micron, monthly lead-lag
    mem = {}
    for sid in ("PCU334413334413", "PCU33441333441312", "WPU117831"):
        try:
            s = fred(sid)
        except Exception as e:  # noqa: BLE001
            mem[sid] = f"fetch failed: {e}"
            continue
        m = pd.DataFrame({"ppi": s.resample("ME").last(), "MU": px["MU"].resample("ME").last(),
                          "SMH": px["SMH"].resample("ME").last()}).dropna()
        r = np.log(m).diff(3).dropna()   # 3-month changes
        r["MU_rel"] = r["MU"] - r["SMH"]
        lags = {}
        for L in range(-6, 7):   # L>0: PPI leads Micron by L months
            lags[L] = {"MU": round(float(r["ppi"].shift(L).corr(r["MU"])), 3),
                       "MU_vs_SMH": round(float(r["ppi"].shift(L).corr(r["MU_rel"])), 3)}
        mem[sid] = {"months": len(m), "from": str(m.index[0].date()), "to": str(m.index[-1].date()),
                    "lead_lag_corr_3m_changes": lags}
        # simple rule: long MU when PPI 3m change > 0 (known one month later), else cash; vs hold
        sig = (r["ppi"] > 0).shift(2)   # one month publication lag + next month
        mret = np.log(m["MU"]).diff().reindex(r.index)
        strat = (mret * sig.astype(float)).dropna()
        mem[sid]["rule_long_when_ppi_rising"] = {"ann_ret_pct": round(100 * 12 * strat.mean(), 1),
                                                 "hold_ann_ret_pct": round(100 * 12 * mret.reindex(strat.index).mean(), 1),
                                                 "share_invested": round(float(sig.reindex(strat.index).mean()), 2)}
    res["memory"] = mem
    json.dump(res, open(f"{a.out}/summary.json", "w"), indent=1, default=str)
    print(json.dumps(res, indent=1, default=str)[:20000])


if __name__ == "__main__":
    main()
