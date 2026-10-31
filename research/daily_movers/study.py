"""Daily big-mover strategies for a $20 account: do any of them catch ~8% movers often enough to compound?

Universe: point-in-time S&P 500 (members as they were each day, delisted names included), 2000-01 to 2026-10.
One position a day (or an equal split across the top N), all money reinvested, long only, fractional shares.
Costs: commission-free broker, so cost = half-spread + slippage per side (5 bp base, 15 bp stress; Sharma ch.9).

    python study.py PANEL.pkl BTC_DAILY.csv QQQ_CSV OUT_DIR   (QQQ_CSV: papers-data prices/QQQ.csv, from 1999, seeds the 200-day average)
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

TARGET = 0.08


def pick(score: pd.DataFrame, ok: pd.DataFrame, n: int, largest: bool = True) -> pd.DataFrame:
    """Boolean frame: the n best scores per row among eligible names."""
    s = score.where(ok)
    r = s.rank(axis=1, ascending=not largest, method="first")
    return r <= n


def trades(sel: pd.DataFrame, ret: pd.DataFrame) -> pd.DataFrame:
    """One row per (day, name) held: the gross return of that trade."""
    st = sel.stack()
    st = st[st]
    r = ret.stack().reindex(st.index)
    return r.rename("ret").reset_index().rename(columns={"level_0": "date", "level_1": "ticker"}).dropna()


def book(tr: pd.DataFrame, cost: float, days: pd.DatetimeIndex) -> pd.Series:
    """Daily account return: equal weight across that day's trades, cost on both sides, cash (0%) on days with none."""
    d = (tr.assign(net=(1 + tr["ret"]) * (1 - cost) / (1 + cost) - 1).groupby("date")["net"].mean())
    return d.reindex(days).fillna(0.0)


def stats(daily: pd.Series, tr: pd.DataFrame | None, start: str, cost: float) -> dict:
    d = daily[daily.index >= start]
    eq = (1 + d).cumprod()
    yrs = len(d) / 252
    out = {"from": start, "end_$20": round(20 * eq.iloc[-1], 2), "cagr_%": round(100 * (eq.iloc[-1] ** (1 / yrs) - 1), 1),
           "max_dd_%": round(100 * (eq / eq.cummax() - 1).min(), 1),
           "sharpe": round(d.mean() / d.std() * np.sqrt(252), 2) if d.std() > 0 else 0.0}
    if tr is not None:
        t = tr[tr["date"] >= start]
        net = (1 + t["ret"]) * (1 - cost) / (1 + cost) - 1
        out.update({"trades": int(len(t)), "win_%": round(100 * (net > 0).mean(), 1),
                    "hit8_%": round(100 * (t["ret"] >= TARGET - 1e-9).mean(), 1), "lose8_%": round(100 * (t["ret"] <= -TARGET).mean(), 1),
                    "avg_win_%": round(100 * net[net > 0].mean(), 2), "avg_loss_%": round(100 * net[net <= 0].mean(), 2),
                    "avg_trade_bp": round(1e4 * net.mean(), 1), "t_stat": round(net.mean() / net.std() * np.sqrt(len(net)), 1), "worst_%": round(100 * net.min(), 1),
                    "best_%": round(100 * net.max(), 1)})
    return out


def main(panel: str, btc_csv: str, qqq_csv: str, out: Path) -> None:
    P = pd.read_pickle(panel)
    C, O, H, L, V = P["close"], P["open"], P["high"], P["low"], P["volume"]
    days = C.index
    mem = P["member"]
    bad = P["notrade"] | P["notrade"].shift(-1, fill_value=False)
    # A delisted name keeps its last price (no free exit fill on the day after it vanishes).
    Cf = C.ffill(limit=5)
    r1 = Cf.pct_change(fill_method=None)              # close t-1 -> close t
    gap = O / Cf.shift(1) - 1                          # close t-1 -> open t
    intraday = Cf / O - 1                              # open t -> close t
    nxt = r1.shift(-1)                                 # close t -> close t+1 (decided at close t)
    ok_c = mem & ~bad & C.notna() & (C > 5) & nxt.notna()       # $5 floor: no penny prints
    # Opens outside the day's high-low range are bad prints: skip them.
    ok_o = mem & ~P["notrade"] & O.notna() & (O <= H * 1.0001) & (O >= L * 0.9999) & Cf.shift(1).notna() & intraday.notna() & (Cf.shift(1) > 5)
    vol20 = r1.rolling(20, min_periods=15).std()
    hi20 = Cf.rolling(20).max()
    vrel = V / V.rolling(20, min_periods=15).mean().shift(1)

    # Exit at +8% (limit) else close, with an optional stop; if both touch on the same day assume the stop came first.
    def target_exit(stop: float | None) -> pd.DataFrame:
        hit_up = H / O - 1 >= TARGET
        r = intraday.copy()
        r = r.mask(hit_up, TARGET)
        if stop is not None:
            r = r.mask(L / O - 1 <= -stop, -stop)
        return r

    S: dict[str, tuple[pd.DataFrame, pd.DataFrame, str]] = {}
    # Close-to-next-close (decided at today's close)
    S["Chase yesterday's #1 gainer"] = (pick(r1, ok_c, 1), nxt, "close")
    S["Chase top 5 gainers"] = (pick(r1, ok_c, 5), nxt, "close")
    S["Buy yesterday's #1 loser"] = (pick(r1, ok_c, 1, largest=False), nxt, "close")
    S["Buy 5 biggest losers"] = (pick(r1, ok_c, 5, largest=False), nxt, "close")
    brk = ok_c & (Cf >= hi20) & (vrel >= 2)
    S["Breakout: 20-day high on 2x volume (top 1)"] = (pick(r1, brk, 1), nxt, "close")
    S["Most volatile stock (20-day vol)"] = (pick(vol20, ok_c, 1), nxt, "close")
    # Open-to-close (decided at today's open from the overnight gap)
    S["Gap-and-go: biggest gap up, open to close"] = (pick(gap, ok_o, 1), intraday, "open")
    S["Gap fill: biggest gap down, open to close"] = (pick(gap, ok_o, 1, largest=False), intraday, "open")
    S["Biggest gap up, sell at +8% or close"] = (pick(gap, ok_o, 1), target_exit(None), "open")
    S["Biggest gap up, +8% target, -4% stop"] = (pick(gap, ok_o, 1), target_exit(0.04), "open")
    S["Biggest gap down, +8% target, -4% stop"] = (pick(gap, ok_o, 1, largest=False), target_exit(0.04), "open")
    S["Random member each day (coin flip)"] = (pick(pd.DataFrame(np.random.default_rng(7).random(C.shape), index=days, columns=C.columns), ok_c, 1), nxt, "close")
    # Perfect hindsight (impossible): shows where the 8%-a-day dream lives
    S["Hindsight: tomorrow's #1 gainer (impossible)"] = (pick(nxt, ok_c, 1), nxt, "close")

    starts = ["2000-01-03", "2019-01-02", "2025-01-02"]
    rows, curves = [], {}
    for name, (sel, ret, when) in S.items():
        tr = trades(sel, ret)
        for cost in (0.0005, 0.0015):
            dly = book(tr, cost, days)
            if cost == 0.0005:
                curves[name] = dly
            for s in starts:
                rows.append({"strategy": name, "decided_at": when, "cost_bp_side": cost * 1e4, **stats(dly, tr, s, cost)})

    # Index and leveraged-index references (3x QQQ rebuilt daily: 3x return minus 0.95%/yr fee and 2%/yr financing on 2x)
    etf = P["etf"].ffill()
    qlong = pd.read_csv(qqq_csv, parse_dates=["date"]).set_index("date")["adj_close"]
    etf["QQQ"] = qlong.reindex(qlong.index.union(days)).ffill().reindex(days)
    spy, qqq = etf["SPY"].pct_change().fillna(0), etf["QQQ"].pct_change().fillna(0)
    tq = 3 * qqq - (0.0095 + 2 * 0.02) / 252
    ma200 = qlong.rolling(200).mean().reindex(qlong.index.union(days)).ffill().reindex(days)
    ref = {"SPY hold": spy, "QQQ hold": qqq, "3x QQQ hold (TQQQ-like)": tq,
           "3x QQQ only when QQQ > 200-day average": tq.where((etf["QQQ"] > ma200).shift(1, fill_value=False), 0.0)
           - 0.0005 * (etf["QQQ"] > ma200).astype(int).diff().abs().fillna(0),
           "3x QQQ the day after a QQQ down day": tq.where((qqq < 0).shift(1, fill_value=False), 0.0) - 0.001 * (qqq < 0).shift(1, fill_value=False)}
    for name, dly in ref.items():
        curves[name] = dly
        for s in starts:
            rows.append({"strategy": name, "decided_at": "close", "cost_bp_side": 5.0, **stats(dly, None, s, 0)})

    # Bitcoin, 2021-2026 daily (btc_ssm study's hold column), 10 bp a side
    b = pd.read_csv(btc_csv, index_col=0, parse_dates=True)["btc_hold"].iloc[1:]
    bt = {"BTC hold": (b, None)}
    for thr, nm in [(0.05, "BTC: buy the day after a +5% day"), (-0.05, "BTC: buy the day after a -5% day")]:
        sig = (b >= thr) if thr > 0 else (b <= thr)
        on = sig.shift(1, fill_value=False)
        tr = pd.DataFrame({"date": b.index[on], "ticker": "BTC", "ret": b[on].values})
        bt[nm] = ((tr.set_index("date")["ret"].add(1).mul((1 - 0.001) / (1 + 0.001)).sub(1)).reindex(b.index).fillna(0), tr)
    for name, (dly, tr) in bt.items():
        for s in ["2021-01-02", "2025-01-02"]:
            rows.append({"strategy": name, "decided_at": "close", "cost_bp_side": 10.0, **stats(dly, tr, s, 0.001 if tr is not None else 0)})

    res = pd.DataFrame(rows)
    out.mkdir(parents=True, exist_ok=True)
    res.to_csv(out / "results.csv", index=False)
    pd.DataFrame(curves).to_pickle(out / "daily_returns.pkl")
    eq = pd.DataFrame({k: 20 * (1 + v[v.index >= "2019-01-02"]).cumprod() for k, v in curves.items()})
    eq.to_csv(out / "equity_20usd_2019.csv", float_format="%.4f")

    # Base rates: how often does anything move 8%, and how often does the "chosen" name do it
    m = ok_c
    base = {"member-days": int(m.sum().sum()),
            "P(a member rises >= 8% next day)_%": round(100 * (nxt[m] >= TARGET).sum().sum() / m.sum().sum(), 2),
            "P(a member falls >= 8% next day)_%": round(100 * (nxt[m] <= -TARGET).sum().sum() / m.sum().sum(), 2),
            "days with at least one member up >= 8%_%": round(100 * ((nxt.where(m) >= TARGET).any(axis=1)).mean(), 1),
            "median #1 gainer move_%": round(100 * nxt.where(m).max(axis=1).median(), 1)}
    # Required hit rate for +8% wins vs a loss of size L, compounding (log break-even)
    need = {f"loss {int(l*100)}%": round(100 * -np.log(1 - l) / (np.log(1 + TARGET) - np.log(1 - l)), 1) for l in (0.02, 0.04, 0.08)}
    (out / "base_rates.json").write_text(json.dumps({"base": base, "breakeven_hit_rate_%_for_8%_wins": need,
                                                     "100_wins_of_8%_compounded_x": round(1.08 ** 100, 0)}, indent=2))
    pd.set_option("display.width", 250, "display.max_columns", 30, "display.max_rows", 300)
    print(json.dumps({"base": base, "need": need}, indent=1))
    print(res[res.cost_bp_side.isin([5.0, 10.0])].drop(columns=["decided_at"]).to_string(index=False))
    print(res[res.cost_bp_side == 15.0][["strategy", "from", "end_$20", "cagr_%", "avg_trade_bp"]].to_string(index=False))


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2], sys.argv[3], Path(sys.argv[4]))
