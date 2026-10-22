"""Backtest the testable strategies of "151 Trading Strategies" (Kakushadze & Serur 2018) from 2019 to the last close.

python run.py CACHE OUT   (CACHE: Yahoo CSV cache folder, OUT: where table.md / results.json go)

Every strategy trades at the close its signal is computed on (market-on-close) and, as a check, at the next close.
Costs a side: 2 bp stocks/ETFs, 5 bp leveraged and VIX funds, 1 bp FX, 10 bp bitcoin. Short borrow a year: 0.5%
stocks/ETFs, 5% leveraged/VIX funds and bitcoin. Idle cash earns T-bills (BIL)."""
from __future__ import annotations

import json
import os
import sys

import numpy as np
import pandas as pd

import strategies as S
from data import fred, yahoo
from engine import backtest, stats

START = pd.Timestamp("2019-01-02")

# S&P 100 members as of 2026: picked today, so the stock tests carry survivorship bias (see README).
SP100 = [
    "AAPL", "ABBV", "ABT", "ACN", "ADBE", "AIG", "AMD", "AMGN", "AMT", "AMZN", "AVGO", "AXP", "BA", "BAC", "BK", "BKNG", "BLK", "BMY",
    "BRK-B", "C", "CAT", "CHTR", "CL", "CMCSA", "COF", "COP", "COST", "CRM", "CSCO", "CVS", "CVX", "DE", "DHR", "DIS", "DUK", "EMR", "F",
    "FDX", "GD", "GE", "GILD", "GM", "GOOGL", "GS", "HD", "HON", "IBM", "INTC", "INTU", "ISRG", "JNJ", "JPM", "KO", "LIN", "LLY", "LMT",
    "LOW", "MA", "MCD", "MDLZ", "MDT", "MET", "META", "MMM", "MO", "MRK", "MS", "MSFT", "NEE", "NFLX", "NKE", "NOW", "NVDA", "ORCL", "PEP",
    "PFE", "PG", "PLTR", "PM", "PYPL", "QCOM", "RTX", "SBUX", "SCHW", "SO", "SPG", "T", "TGT", "TMO", "TMUS", "TSLA", "TXN", "UBER", "UNH",
    "UNP", "UPS", "USB", "V", "VZ", "WFC", "WMT", "XOM"
]
SECTORS = ["XLB", "XLE", "XLF", "XLI", "XLK", "XLP", "XLU", "XLV", "XLY", "XLRE", "XLC"]
MULTI = ["SPY", "EFA", "EEM", "TLT", "IEF", "GLD", "DBC", "VNQ"]
COMMOD = ["GLD", "SLV", "USO", "UNG", "DBA", "CPER", "PPLT", "PALL", "CORN", "WEAT", "SOYB", "DBB"]
FUT = ["SPY", "EFA", "EEM", "TLT", "IEF", "GLD", "SLV", "USO", "UNG", "DBA", "CPER", "FXE", "FXY", "FXB", "FXA"]
OTHER = ["QQQ", "IWM", "BIL", "TQQQ", "SQQQ", "VIXY", "VIXM", "^VIX", "^VIX3M", "BTC-USD"]
# Yahoo FX ticker -> (currency, invert so the series is the USD price of the foreign currency)
FX = {"EURUSD=X": ("EUR", False), "GBPUSD=X": ("GBP", False), "AUDUSD=X": ("AUD", False),
      "JPY=X": ("JPY", True), "CAD=X": ("CAD", True), "CHF=X": ("CHF", True)}
RATES = {"USD": "TB3MS", "EUR": "IR3TIB01EZM156N", "GBP": "IR3TIB01GBM156N", "AUD": "IR3TIB01AUM156N",
         "JPY": "IR3TIB01JPM156N", "CAD": "IR3TIB01CAM156N", "CHF": "IR3TIB01CHM156N"}
SIX = ["SPY", "GLD", "EUR", "TLT", "USO", "BTC"]
PRICY = {"TQQQ", "SQQQ", "VIXY", "VIXM"}


def cost_bp(t: str) -> float:
    return 10.0 if t == "BTC" else 1.0 if t in RATES else 5.0 if t in PRICY else 2.0


def borrow(t: str) -> float:
    return 0.05 if t in PRICY or t == "BTC" else 0.0 if t in RATES else 0.005


def to_calendar(d: pd.DataFrame, cal: pd.DatetimeIndex) -> pd.DataFrame:
    """Fold a 7-day (crypto) or other-calendar series onto NYSE days: each day goes to the first NYSE day on/after it."""
    d = d[(d.index >= cal[0]) & (d.index <= cal[-1])]
    pos = cal.searchsorted(d.index)
    g = d.groupby(cal[pos])
    return pd.DataFrame({"open": g.open.first(), "high": g.high.max(), "low": g.low.min(), "close": g.close.last()}).reindex(cal)


def load(cache: str) -> dict:
    tickers = list(dict.fromkeys(SP100 + SECTORS + MULTI + COMMOD + FUT + OTHER))
    raw = {}
    for t in tickers + list(FX):
        d = yahoo(t, cache)
        if d is not None:
            raw[t] = d
    print(f"loaded {len(raw)} of {len(tickers) + len(FX)} series")
    cal = raw["SPY"].index
    ohlc = {}
    for t, d in raw.items():
        if t in FX:
            name, inv = FX[t]
            d = to_calendar(d, cal)
            if inv:
                d = pd.DataFrame({"open": 1 / d.open, "high": 1 / d.low, "low": 1 / d.high, "close": 1 / d.close})
            ohlc[name] = d
        elif t == "BTC-USD":
            ohlc["BTC"] = to_calendar(d, cal)
        else:
            ohlc[t] = d.reindex(cal)
    rates = {}
    for cur, sid in RATES.items():
        s = fred(sid)
        if s is not None:
            rates[cur] = s / 100
            print(f"  rate {cur} {sid}: last {s.index[-1].date()} {s.iloc[-1]:.2f}%")
    return {"ohlc": ohlc, "rates": pd.DataFrame(rates)}


def build(data: dict):
    ohlc = data["ohlc"]
    close = pd.DataFrame({t: d.close for t, d in ohlc.items()})
    cal = close.index
    rf = close["BIL"].pct_change().fillna(0.0).clip(-0.01, 0.01)
    rates = data["rates"]
    # monthly rates, usable from the first day of the following month; missing currencies earn the US rate
    if rates.empty:
        rd = pd.DataFrame(index=cal)
    else:
        rd = rates.copy()
        rd.index = pd.DatetimeIndex(rd.index) + pd.offsets.MonthBegin(1)
        rd = rd.reindex(cal.union(rd.index)).ffill().reindex(cal)
    us = rd["USD"] if "USD" in rd else rf * 252
    ret = close.pct_change(fill_method=None)
    for cur in RATES:
        if cur != "USD" and cur in ret:
            fr = rd.get(cur, us)
            ret[cur] = ret[cur] + fr.fillna(us).fillna(0) / 252   # foreign deposit interest on a long position
    return close, ret, rf, rd


def evaluate(name, chapter, W, ret, rf, spy, note=""):
    W = W.reindex(ret.index)
    W = W.loc[:, [c for c in W.columns if c in ret.columns]]
    out = {"name": name, "chapter": chapter, "note": note}
    res = {}
    for key, kw in (("net", {"cost_bp": cost_bp, "borrow_yr": borrow, "lag": 0}),
                    ("net_next", {"cost_bp": cost_bp, "borrow_yr": borrow, "lag": 1}),
                    ("gross", {"cost_bp": 0.0, "borrow_yr": 0.0, "lag": 0})):
        bt = backtest(W, ret, rf, **kw)
        bt = bt[bt.index >= START]
        res[key] = bt
        out[key] = stats(bt.ret, spy)
    out["turnover_yr"] = float(res["net"].turnover.mean() * 252)
    out["series"] = res["net"].ret
    return out


def hold(t, cal):
    return pd.DataFrame({t: pd.Series(1.0, index=cal)})


def run_all(data: dict) -> tuple[list, dict]:
    close, ret, rf, rd = build(data)
    ohlc = data["ohlc"]
    cal = close.index
    spy = ret["SPY"].fillna(0.0)
    spy_s = spy[spy.index >= START]
    stocks = close[[t for t in SP100 if t in close]]
    sect = close[[t for t in SECTORS if t in close]]
    rows = []

    def add(name, ch, W, note=""):
        print(f"  {name}")
        rows.append(evaluate(name, ch, W, ret, rf, spy_s, note))

    # benchmarks
    add("Hold SPY", "benchmark", hold("SPY", cal))
    add("Hold QQQ", "benchmark", hold("QQQ", cal))
    add("S&P 100 equal weight (same stocks)", "benchmark", S.equal_weight(stocks))
    # 3. Stocks
    add("3.1 Price momentum, long/short decile", "Stocks", S.price_momentum(stocks))
    add("3.1 Price momentum, long top decile", "Stocks", S.price_momentum(stocks, long_only=True))
    add("3.4 Low volatility, long/short decile", "Stocks", S.low_vol(stocks))
    add("3.4 Low volatility, long calmest decile", "Stocks", S.low_vol(stocks, long_only=True))
    add("3.7 Residual momentum, long/short decile", "Stocks", S.residual_momentum(stocks, close.SPY))
    add("3.8 Pairs trading (distance, 20 pairs)", "Stocks", S.pairs(stocks))
    add("3.9 Mean reversion, single cluster (daily)", "Stocks", S.mean_reversion(stocks))
    add("3.10 Mean reversion, sector clusters (daily)", "Stocks", S.mean_reversion(stocks, S.sector_clusters(stocks, sect)))
    add("3.18 k-nearest neighbours on SPY (long/cash)", "Stocks", S.knn(close.SPY))
    add("3.21 Alpha combo (mom + resid mom + low vol)", "Stocks", S.alpha_combo(stocks, close.SPY))
    for fn, nm in ((S.sma_rule, "3.12 Single moving average 200d"), (S.two_ma, "3.13 Two moving averages 50/200"),
                   (S.three_ma, "3.14 Three moving averages 10/50/200")):
        add(f"{nm} on SPY, long/short", "Stocks", S.signal_weights(fn(close.SPY), "SPY"))
        add(f"{nm} on SPY, long/cash", "Stocks", S.signal_weights(fn(close.SPY, short=False), "SPY"))
    add("3.15 Support & resistance pivots on SPY", "Stocks", S.signal_weights(S.pivot(ohlc["SPY"]), "SPY"))
    add("3.16 Channel 20d (buy floor, sell ceiling) on SPY", "Stocks", S.signal_weights(S.channel(ohlc["SPY"]), "SPY"))
    # 4. ETFs
    add("4.1 Sector momentum rotation, top 3", "ETFs", S.sector_rotation(sect))
    add("4.1 Sector rotation + 200d MA filter", "ETFs", S.sector_rotation(sect, mode="ma"))
    add("4.1 Sector rotation, dual momentum", "ETFs", S.sector_rotation(sect, rf_price=close.BIL, mode="dual"))
    add("4.2 Alpha rotation (sectors)", "ETFs", S.alpha_rotation(sect, close.SPY))
    add("4.3 R-squared alpha rotation (sectors)", "ETFs", S.alpha_rotation(sect, close.SPY, r2_filter=True))
    add("4.4 IBS mean reversion on SPY", "ETFs", S.signal_weights(S.ibs(ohlc["SPY"]), "SPY"))
    add("4.4 IBS mean reversion on QQQ", "ETFs", S.signal_weights(S.ibs(ohlc["QQQ"]), "QQQ"))
    add("4.5 Short TQQQ + SQQQ (leveraged decay)", "ETFs", S.leveraged_pair(close))
    add("4.6 Multi-asset trend following (8 ETFs)", "ETFs", S.trend_multi(close[MULTI]))
    # 6. Indexes, 7. Volatility
    add("6.4 SPY volatility targeting 15% + T-bills", "Indexes", S.vol_target(close.SPY))
    add("7.1 VIX basis: short VIXY in contango, long in backwardation", "Volatility", S.vix_basis(close["^VIX"], close["^VIX3M"]))
    add("7.1 VIX basis: short VIXY in contango only", "Volatility", S.vix_basis(close["^VIX"], close["^VIX3M"], both=False))
    add("7.2 Vol carry: short VIXY, long VIXM hedge", "Volatility", S.vol_carry_etns(close))
    add("7.3 Volatility risk premium: short VIXY 20%", "Volatility", S.short_vol(index=cal))
    # 8. FX
    fxc = [c for c in ("EUR", "GBP", "AUD", "JPY", "CAD", "CHF") if c in close]
    fxp = close[fxc]
    hp = pd.DataFrame({c: S.hp_trend(fxp[c]) for c in fxc}) / len(fxc)
    add("8.1 HP-filtered moving averages, 6 currencies", "FX", hp)
    if all(c in rd for c in fxc + ["USD"]):
        rates_m = rd[fxc + ["USD"]]
        add("8.2 Carry trade (G7 currencies)", "FX", S.fx_carry(rates_m, fxp))
        add("8.3 Dollar carry", "FX", S.fx_carry(rates_m, fxp, mode="dollar"))
        add("8.4 Momentum & carry combo", "FX", S.fx_carry(rates_m, fxp, mode="combo"))
    else:
        print("  FX carry skipped: interest rates missing")
    # 9. Commodities, 10. Futures
    com = close[[t for t in COMMOD if t in close]]
    add("9.3 Diversify with commodities: 80% SPY / 20% DBC", "Commodities", S.fixed_mix(close, {"SPY": 0.8, "DBC": 0.2}))
    add("9.4 Commodity value (5-year reversal), 3 vs 3", "Commodities", S.commodity_value(com))
    add("9.5 Commodity skewness premium, 3 vs 3", "Commodities", S.skewness(com))
    fut = close[[t for t in FUT if t in close]]
    add("10.3 Futures contrarian, weekly (15 ETF proxies)", "Futures", S.contrarian(fut))
    add("10.4 Futures trend following (15 ETF proxies)", "Futures", S.trend_multi(fut))
    # 18. Crypto, 19. Global macro
    add("Hold bitcoin", "benchmark", hold("BTC", cal))
    add("18.2 Neural network on bitcoin (long/cash)", "Crypto", S.ann_crypto(close.BTC))
    add("19.5 Pre-FOMC drift: SPY the day before decisions", "Global macro", S.pre_fomc(cal))

    # Same idea, six markets: each technical rule on SPY, gold, euro, long bonds, oil and bitcoin, long/short as in
    # the paper, plus all six together at 1/6 each.
    six = {}
    rules = {"Hold": lambda m: pd.Series(1.0, index=cal),
             "3.12 SMA 200": lambda m: S.sma_rule(close[m]), "3.13 MA 50/200": lambda m: S.two_ma(close[m]),
             "3.14 MA 10/50/200": lambda m: S.three_ma(close[m]), "3.15 Pivots": lambda m: S.pivot(ohlc[m]),
             "3.16 Channel 20d": lambda m: S.channel(ohlc[m]), "10.4 12m momentum": lambda m: S.ts_momentum(close[m]),
             "8.1 HP filter MA": lambda m: S.hp_trend(close[m])}
    for rn, fn in rules.items():
        sigs = {m: fn(m) for m in SIX if m in close}
        six[rn] = {}
        for m, sg in sigs.items():
            r = evaluate(f"{rn} {m}", "six", S.signal_weights(sg, m), ret, rf, spy_s)
            six[rn][m] = {k: r["net"][k] for k in ("cagr", "sharpe", "max_dd")}
        W = pd.DataFrame(sigs) / len(sigs)
        r = evaluate(f"{rn} all six", "six", W, ret, rf, spy_s)
        six[rn]["all six"] = {k: r["net"][k] for k in ("cagr", "sharpe", "max_dd")}
        print(f"  six markets: {rn}")
    return rows, six


def edge(r, spy_sharpe):
    n = r["net"]
    return n["sharpe"] > spy_sharpe and n["alpha_t"] > 2 and n["cagr_2019_22"] > 0 and n["cagr_2023_26"] > 0


def report(rows, six, out: str):
    os.makedirs(out, exist_ok=True)
    spy = next(r for r in rows if r["name"] == "Hold SPY")["net"]
    ranked = sorted(rows, key=lambda r: -r["net"]["sharpe"])
    first, last = rows[0]["series"].index[0].date(), rows[0]["series"].index[-1].date()
    lines = [(f"Period {first} to {last}, net of costs, trades at the signal close (MOC). SPY: {spy['cagr']:+.1%}/yr, "
              f"Sharpe {spy['sharpe']:.2f}, worst drop {spy['max_dd']:.0%}."), "",
             ("| # | strategy | CAGR | Sharpe | worst drop | beta | alpha/yr | alpha t | 2019-22 | 2023-26 | Sharpe next close | "
              "gross CAGR | turnover/yr | edge |"), "|" + "---|" * 14]
    for k, r in enumerate(ranked, 1):
        n = r["net"]
        lines.append(f"| {k} | {r['name']} | {n['cagr']:+.1%} | {n['sharpe']:.2f} | {n['max_dd']:.0%} | {n['beta']:.2f} | "
                     f"{n['alpha_ann']:+.1%} | {n['alpha_t']:.1f} | {n['cagr_2019_22']:+.1%} | {n['cagr_2023_26']:+.1%} | "
                     f"{r['net_next']['sharpe']:.2f} | {r['gross']['cagr']:+.1%} | {r['turnover_yr']:.0f}x | "
                     f"{'yes' if edge(r, spy['sharpe']) else ''} |")
    lines += ["", "edge = net Sharpe above holding SPY, alpha t-stat above 2, and positive in both 2019-22 and 2023-26.", "",
              "## Same idea, six markets (long/short as in the paper; net Sharpe, CAGR in brackets)", ""]
    mk = list(next(iter(six.values())).keys())
    lines += ["| rule | " + " | ".join(mk) + " |", "|" + "---|" * (len(mk) + 1)]
    for rn, v in six.items():
        lines.append(f"| {rn} | " + " | ".join(f"{v[m]['sharpe']:.2f} ({v[m]['cagr']:+.0%})" for m in mk) + " |")
    md = "\n".join(lines)
    with open(f"{out}/table.md", "w") as f:
        f.write(md + "\n")
    js = [{k: v for k, v in r.items() if k != "series"} for r in rows]
    with open(f"{out}/results.json", "w") as f:
        json.dump({"rows": js, "six_markets": six}, f, indent=1, default=float)
    pd.DataFrame({r["name"]: r["series"] for r in rows}).to_csv(f"{out}/daily_returns.csv", float_format="%.6g")
    print(md)


if __name__ == "__main__":
    cache, out = sys.argv[1], sys.argv[2]
    np.seterr(all="ignore")
    rows, six = run_all(load(cache))
    report(rows, six, out)
