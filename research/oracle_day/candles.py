"""Do the candlestick / chart signals day traders use flag the minute-level up-spikes the oracle feeds on?

Target: the next 5 minutes' return (close of bar t to close of bar t+5). A "spike" is a next-5-minute rise in the top 5%
of that ticker's own moves. Each signal uses only bars up to t, indicators restart each session (09:30-16:00 NY).
For every signal: how often a spike follows (base rate 5%), the average next-5-minute move vs. all bars, and a t-stat on
day-by-day means. Then all signals together in a logistic model, trained on the first days and scored on the last ones.

python candles.py week DATA_DIR        (DATA_DIR/<TICKER>/bars.csv; last week's 26 tickers)
python candles.py history DATA_DIR     (DATA_DIR/<TICKER>/<YYYY-MM>.csv; exp7-data SPY/QQQ 2019-2026, per-year check)
Paper research only.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

H = 5          # minutes ahead
TOP = 0.95     # spike = top 5% of a ticker's next-5-minute returns


def load(files: list[Path]) -> pd.DataFrame:
    d = pd.concat([pd.read_csv(f) for f in files if f.stat().st_size > 10], ignore_index=True)
    d["ts"] = pd.to_datetime(d["date"], utc=True).dt.tz_convert("America/New_York")
    d = d.drop(columns="date").drop_duplicates("ts").set_index("ts").sort_index().between_time("09:30", "15:59")
    d["day"] = d.index.strftime("%Y-%m-%d")
    n = d.groupby("day")["close"].transform("nunique")
    return d[n >= 50]


def rsi(c: pd.Series, n: int = 14) -> pd.Series:
    x = c.diff()
    up, dn = x.clip(lower=0).ewm(alpha=1 / n, adjust=False).mean(), (-x.clip(upper=0)).ewm(alpha=1 / n, adjust=False).mean()
    return 100 - 100 / (1 + up / dn.replace(0, np.nan))


def day_features(g: pd.DataFrame) -> pd.DataFrame:
    o, h, l, c, v = g["open"], g["high"], g["low"], g["close"], g["volume"]
    body, rng = (c - o).abs(), (h - l).replace(0, np.nan)
    upper, lower = h - np.maximum(o, c), np.minimum(o, c) - l
    green, red = c > o, c < o
    avg_rng = rng.rolling(20, min_periods=5).mean()
    down5 = c < c.shift(5)
    f = pd.DataFrame(index=g.index)
    f["hammer"] = (lower >= 2 * body) & (upper <= body) & down5 & (rng > 0.5 * avg_rng)
    f["inverted_hammer"] = (upper >= 2 * body) & (lower <= body) & down5 & (rng > 0.5 * avg_rng)
    f["bullish_engulfing"] = red.shift(1, fill_value=False) & green & (c >= o.shift(1)) & (o <= c.shift(1)) & (body > body.shift(1))
    f["morning_star"] = (red.shift(2, fill_value=False) & (body.shift(2) > avg_rng.shift(2) * 0.6) & (body.shift(1) < 0.3 * body.shift(2))
                         & green & (c > (o.shift(2) + c.shift(2)) / 2))
    f["doji_after_drop"] = (body <= 0.1 * rng) & down5
    f["three_white_soldiers"] = green & green.shift(1, fill_value=False) & green.shift(2, fill_value=False) & (c > c.shift(1)) & (c.shift(1) > c.shift(2))
    f["bullish_marubozu"] = green & (body >= 0.9 * rng) & (rng > 1.5 * avg_rng)
    r = rsi(c)
    f["rsi_below_30"] = r < 30
    f["rsi_cross_up_30"] = (r > 30) & (r.shift(1) <= 30)
    f["rsi_above_70"] = r > 70
    macd = c.ewm(span=12, adjust=False).mean() - c.ewm(span=26, adjust=False).mean()
    sig = macd.ewm(span=9, adjust=False).mean()
    f["macd_bull_cross"] = (macd > sig) & (macd.shift(1) <= sig.shift(1))
    m20, s20 = c.rolling(20).mean(), c.rolling(20).std()
    f["bollinger_lower_touch"] = l <= m20 - 2 * s20
    f["bollinger_upper_break"] = c > m20 + 2 * s20
    bw = (4 * s20 / m20)
    f["squeeze_breakout"] = (bw.shift(1) <= bw.rolling(60, min_periods=30).quantile(0.2).shift(1)) & (c > m20 + 2 * s20)
    e9, e21 = c.ewm(span=9, adjust=False).mean(), c.ewm(span=21, adjust=False).mean()
    f["ema9_cross_21"] = (e9 > e21) & (e9.shift(1) <= e21.shift(1))
    vwap = (c * v).cumsum() / v.cumsum().replace(0, np.nan)
    f["vwap_reclaim"] = (c > vwap) & (c.shift(1) <= vwap.shift(1))
    f["volume_spike_green"] = green & (v > 3 * v.rolling(20, min_periods=5).mean().shift(1))
    i = np.arange(len(g))
    orh = h.iloc[:15].max() if len(g) > 15 else np.nan
    brk = (c > orh) & (i >= 15)
    f["opening_range_breakout"] = brk & ~brk.shift(1, fill_value=False) & (brk.cumsum() == 1)
    f["new_day_high"] = (h > h.cummax().shift(1)) & (i >= 30)
    f["fwd"] = c.shift(-H) / c - 1
    return f


def features(d: pd.DataFrame) -> pd.DataFrame:
    out = []
    for (t, day), g in d.groupby(["ticker", "day"], sort=False):
        f = day_features(g)
        f["ticker"], f["day"] = t, day
        out.append(f)
    f = pd.concat(out).dropna(subset=["fwd"])
    f["spike"] = f["fwd"] >= f.groupby("ticker")["fwd"].transform(lambda x: x.quantile(TOP))
    f["drop"] = f["fwd"] <= f.groupby("ticker")["fwd"].transform(lambda x: x.quantile(1 - TOP))
    return f


def score(f: pd.DataFrame, sigs: list[str]) -> pd.DataFrame:
    rows, base, mu = [], f["spike"].mean(), f["fwd"].mean()
    for s in sigs:
        x = f[f[s]]
        if len(x) < 30:
            continue
        dm = (x.groupby("day")["fwd"].mean() - f.groupby("day")["fwd"].mean()).dropna()
        t = dm.mean() / (dm.std(ddof=1) / np.sqrt(len(dm))) if len(dm) > 2 and dm.std() > 0 else np.nan
        rows.append({"signal": s, "n": len(x), "spike_rate_%": round(100 * x["spike"].mean(), 2), "lift": round(x["spike"].mean() / base, 2),
                     "drop_rate_%": round(100 * x["drop"].mean(), 2), "up_share_of_big_%": round(100 * x["spike"].sum() / max(1, x["spike"].sum() + x["drop"].sum()), 1),
                     "next5m_bp": round(1e4 * x["fwd"].mean(), 2), "vs_all_bp": round(1e4 * (x["fwd"].mean() - mu), 2), "t_days": round(t, 2)})
    return pd.DataFrame(rows).sort_values("lift", ascending=False)


def auc(y: np.ndarray, p: np.ndarray) -> float:
    r = pd.Series(p).rank().to_numpy()
    n1 = y.sum()
    return float((r[y].sum() - n1 * (n1 + 1) / 2) / (n1 * (len(y) - n1)))


def logistic(X: np.ndarray, y: np.ndarray, it: int = 300, lam: float = 1e-3) -> np.ndarray:
    X = np.c_[np.ones(len(X)), X]
    w = np.zeros(X.shape[1])
    for _ in range(it):            # Newton steps
        p = 1 / (1 + np.exp(-X @ w))
        g = X.T @ (p - y) + lam * w
        Hm = (X * (p * (1 - p))[:, None]).T @ X + lam * np.eye(len(w))
        w -= np.linalg.solve(Hm, g)
    return w


def main(mode: str, data: str) -> None:
    root = Path(data)
    if mode == "week":
        d = pd.concat([load([f]).assign(ticker=f.parent.name) for f in sorted(root.glob("*/bars.csv"))])
    else:
        d = pd.concat([load(sorted((root / t).glob("*.csv"))).assign(ticker=t) for t in ["SPY", "QQQ"]])
    f = features(d)
    sigs = [c for c in f.columns if c not in ("fwd", "ticker", "day", "spike", "drop")]
    res = {"mode": mode, "bars": int(len(f)), "days": int(f["day"].nunique()), "tickers": int(f["ticker"].nunique()),
           "base_spike_rate_%": round(100 * f["spike"].mean(), 2), "all_bars_next5m_bp": round(1e4 * f["fwd"].mean(), 2)}
    tab = score(f, sigs)
    res["signals"] = tab.to_dict("records")
    # every signal at once: train on earlier days, test on later ones
    days = sorted(f["day"].unique())
    cut = days[int(len(days) * 0.6)]
    tr, te = f[f["day"] < cut], f[f["day"] >= cut]
    X = lambda z: z[sigs].to_numpy(float)
    w = logistic(X(tr), tr["spike"].to_numpy(float))
    p = 1 / (1 + np.exp(-np.c_[np.ones(len(te)), X(te)] @ w))
    yt = te["spike"].to_numpy(bool)
    res_drop = te["drop"].to_numpy(bool)
    top = p >= np.quantile(p, 0.95)
    res["combo"] = {"train_days": f"{days[0]}..{days[days.index(cut) - 1]}", "test_days": f"{cut}..{days[-1]}",
                    "test_auc": round(auc(yt, p), 3), "test_spike_rate_top5pct_%": round(100 * yt[top].mean(), 2),
                    "test_base_%": round(100 * yt.mean(), 2), "test_drop_rate_top5pct_%": round(100 * res_drop[top].mean(), 2), "test_top5pct_next5m_bp": round(1e4 * te["fwd"].to_numpy()[top].mean(), 2),
                    "test_all_next5m_bp": round(1e4 * te["fwd"].mean(), 2)}
    # a plain "spikes cluster" control: was the last 5 minutes a spike-size move?
    f["prev"] = f.groupby(["ticker", "day"])["fwd"].shift(H)
    f["abs_prev_top"] = f["prev"].abs() >= f.groupby("ticker")["prev"].transform(lambda x: x.abs().quantile(TOP))
    res["control_big_prev_move"] = score(f, ["abs_prev_top"]).to_dict("records")
    if mode == "history":
        f["year"] = f["day"].str[:4]
        res["by_year_lift"] = {s: {y: round(g.loc[g[s], "spike"].mean() / g["spike"].mean(), 2) for y, g in f.groupby("year")} for s in sigs}
        res["by_year_vs_all_bp"] = {s: {y: round(1e4 * (g.loc[g[s], "fwd"].mean() - g["fwd"].mean()), 2) for y, g in f.groupby("year")} for s in sigs}
    pd.set_option("display.width", 200)
    print(json.dumps({k: v for k, v in res.items() if k not in ("signals", "by_year_lift", "by_year_vs_all_bp")}, indent=1))
    print(tab.to_string(index=False))
    Path(f"candles_{mode}.json").write_text(json.dumps(res, indent=1))


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
