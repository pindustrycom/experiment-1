"""Static site page for the BTC volatility study: experiments/btc_vol/index.html (listed in hub/build_hub.CATALOG).

    python site_page.py DATA_DIR

DATA_DIR holds btc_1h.csv (the btc-ssm-data release). The equity curves are the live pots' own rule
(experiments/strategies/strats/crypto.run_rule) on Binance 4-hour bars; every other number is from results/*.json.
"""
from __future__ import annotations

import html
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent.parent
sys.path.insert(0, str(ROOT / "experiments" / "strategies"))
from strats import crypto

e = html.escape
CSS = (ROOT / "experiments" / "trader_skill" / "index.html").read_text().split("<style>")[1].split("</style>")[0]


def bars(folder):
    b = pd.read_csv(f"{folder}/btc_1h.csv", index_col=0)
    b.index = pd.to_datetime(b.index, format="mixed")
    g = b.groupby(b.index.floor("4h"))
    d = pd.DataFrame({"open": g.open.first(), "high": g.high.max(), "low": g.low.min(), "close": g.close.last()})
    return d[g.size() == 4]


def curves(d, start="2019-01-01"):
    plain = crypto.run_rule(d, stop=0.06)["equity"]
    sized = crypto.run_rule(d, stop=0.06, size=True)["equity"]
    c = pd.DataFrame({"BTC breakout, vol-sized": sized, "BTC breakout": plain, "Hold BTC": d.close}).resample("D").last().loc[start:]
    return c / c.iloc[0]


def stats(s):
    r = s.pct_change().dropna()
    yrs = len(r) / 365
    return {"cagr": s.iloc[-1] ** (1 / yrs) - 1, "sharpe": r.mean() / r.std() * np.sqrt(365), "dd": (s / s.cummax() - 1).min(),
            "years": {y: (1 + g).prod() - 1 for y, g in r.groupby(r.index.year)}}


def svg_log(c, colors, w=760, h=300, l=46, rt=640):
    lo, hi = np.log(c.min().min()), np.log(c.max().max())
    x = lambda i: l + (rt - l) * i / (len(c) - 1)
    y = lambda v: 274 - 262 * (np.log(v) - lo) / (hi - lo)
    out = [f'<svg viewBox="0 0 {w} {h}" role="img" aria-label="Growth of 1, log scale">']
    for v in np.exp(np.linspace(lo, hi, 5)):
        out.append(f'<line class="grid" x1="{l}" x2="{rt}" y1="{y(v):.1f}" y2="{y(v):.1f}"/>'
                   f'<text class="ax" x="{l - 6}" y="{y(v) + 4:.1f}" text-anchor="end">{v:.1f}x</text>')
    for yr in range(c.index[0].year + 1, c.index[-1].year + 1):
        i = c.index.searchsorted(pd.Timestamp(f"{yr}-01-01"))
        out.append(f'<text class="ax" x="{x(i):.1f}" y="294" text-anchor="middle">{yr}</text>')
    step = max(1, len(c) // 600)
    for (name, s), col in zip(c.items(), colors):
        pts = " ".join(f"{x(i):.1f},{y(v):.1f}" for i, v in enumerate(s.to_numpy()) if i % step == 0 or i == len(s) - 1)
        out.append(f'<polyline fill="none" stroke="{col}" stroke-width="2" points="{pts}"/>')
        out.append(f'<text class="lab" x="{rt + 6}" y="{y(s.iloc[-1]) + 4:.1f}" fill="{col}">{e(name)} {s.iloc[-1]:.0f}x</text>')
    return "".join(out) + "</svg>"


def pct(v, sign=True):
    cls = "pos" if v > 0 else "neg" if v < 0 else ""
    return f'<span class="{cls}">{v:+.1%}</span>' if sign else f"{v:.1%}"


def page(folder):
    c = curves(bars(folder))
    st = {k: stats(s) for k, s in c.items()}
    years = sorted(st["Hold BTC"]["years"])
    vs, pl = st["BTC breakout, vol-sized"], st["BTC breakout"]
    yt = "".join(f"<tr><td>{e(k)}</td>" + "".join(f"<td>{pct(s['years'][y])}</td>" for y in years)
                 + f"<td>{pct(s['cagr'])}</td><td>{s['sharpe']:.2f}</td><td>{pct(s['dd'])}</td></tr>" for k, s in st.items())
    r5 = json.loads((HERE / "results" / "round5.json").read_text())
    k6 = r5["item6_kelly"]
    chart = svg_log(c.rename(columns={"BTC breakout, vol-sized": "Vol-sized", "BTC breakout": "Breakout", "Hold BTC": "Hold"}),
                    ["var(--c1)", "var(--c2)", "var(--muted)"])
    return f"""<!doctype html><html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1"><meta name="robots" content="noindex,nofollow">
<title>BTC volatility study</title><style>{CSS}</style></head><body><main class="wrap">
<h1>BTC volatility study</h1>
<p class="lede">Can modelling Bitcoin's volatility make money? Five rounds of tests (Kalman + GRU, Heston, rough volatility,
real Deribit option prices), all walk-forward, all paper only.</p>
<div class="verdict"><b>Verdict: volatility forecasts pay through sizing, not direction.</b> No model predicts which way BTC moves.
Scaling the BTC breakout position by recent volatility does help, and it is now a live paper pot (Markets, Short-term,
"BTC breakout, vol-sized"). Selling options when implied vol is above our forecast also earns money, but it is not yet
proven on daily data. A daily Deribit option-chain collector started on 30 October 2026 to settle that.</div>
<div class="stats">
<div class="stat"><span>Vol-sized BTC breakout, per year</span><b class="pos">{vs['cagr']:+.1%}</b><span>2019 to 2026, Sharpe {vs['sharpe']:.2f}, worst drop {vs['dd']:.0%}</span></div>
<div class="stat"><span>Plain BTC breakout, per year</span><b>{pl['cagr']:+.1%}</b><span>Sharpe {pl['sharpe']:.2f}, worst drop {pl['dd']:.0%}</span></div>
<div class="stat"><span>Gated option selling (call wing)</span><b class="pos">t 4.4</b><span>+11.3% of the strike a month, 28 of 66 months</span></div>
</div>
<h2>The live pot: BTC breakout, sized by volatility</h2>
<p>Same signal as the BTC breakout pot (4-hour Donchian channel). While long, the position is 2.5% / BTC's daily volatility over
the past 30 days, up to 2x the pot when BTC is calm and less when it is wild, reset once a day only if the size moves 25% or more.
25 bp a side and the 6% daily stop, like the plain pot. Backtest on Binance 4-hour bars:</p>
<figure><figcaption>Growth of 1 since January 2019 (log scale)</figcaption>{chart}</figure>
<div class="tbl"><table><tr><th>Year</th>{''.join(f'<th>{y}</th>' for y in years)}<th>Per year</th><th>Sharpe</th><th>Worst drop</th></tr>{yt}</table></div>
<p class="small muted">Half-Kelly leverage instead (round 5): {k6['kelly_half|hist30']['cagr']:+.1%} a year from 2021 at the same Sharpe
({k6['kelly_half|hist30']['sharpe']:.2f}) and a deeper worst drop ({k6['kelly_half|hist30']['max_dd']:.0%}), so it adds leverage, not edge.
Sizing with the sharper GRU forecast did worse than plain 30-day volatility.</p>
<h2>Forecasting volatility (rounds 1 and 2)</h2>
<div class="tbl"><table><tr><th>Model</th><th>Next hour (log-lik. vs GARCH)</th><th>Next 30 days</th><th>Direction</th></tr>
<tr><td>Kalman + GRU, Student-t</td><td class="pos">best, -0.169</td><td>worse than GARCH</td><td>none</td></tr>
<tr><td>GRU, Student-t</td><td class="pos">-0.160, only calibrated 1% VaR</td><td>worse than GARCH</td><td>none</td></tr>
<tr><td>Heston + jumps</td><td>-0.144</td><td>-</td><td>none</td></tr>
<tr><td>Rough volatility + jumps</td><td>-0.128</td><td class="pos">best (QLIKE -0.175)</td><td>none</td></tr>
<tr><td>HAR</td><td>-</td><td>best for next day</td><td>none</td></tr></table></div>
<p>Fat tails explain most of the gain. Continuous-time volatility (GBM with moving volatility: Heston, rough vol) improves the
forecast of how far BTC moves, never which way.</p>
<h2>Trading volatility with real Deribit prices (rounds 3 to 5)</h2>
<div class="tbl"><table><tr><th>Version (66 monthly entries, 2021-04 to 2026-09)</th><th>Per month, % of strike</th><th>t</th></tr>
<tr><td>Synthetic variance swap at DVOL</td><td>+9.5%</td><td>1.5</td></tr>
<tr><td>Option strip sold at the bid, daily hedge, with fees</td><td>+2.8%</td><td>0.4</td></tr>
<tr><td>Call wing only, band hedge</td><td>+11.6%</td><td>2.1</td></tr>
<tr><td>Call wing, sold only when implied vol is above the rough-vol forecast</td><td class="pos">+11.3%</td><td class="pos">4.4</td></tr>
<tr><td>Risk reversal (sell call, buy put)</td><td>+3.4%</td><td>0.9</td></tr>
<tr><td>Synthetic swap entered daily, gated</td><td>-</td><td class="pos">4.0</td></tr></table></div>
<p>The forecast's job is timing: it skips months when implied vol is too cheap, which were most of the crash months. Most of the
gain came in 2021 and 2024, and 2025-26 earned only about 1-2% a year, so there is no option pot yet. The real daily-entry
test runs once about three months of daily chains are collected.</p>
<p class="small muted">Code and full write-up: <a href="https://github.com/pindustrycom/experiment-1/tree/main/research/btc_ssm">research/btc_ssm</a>.
Paper only, no money involved.</p>
</main></body></html>
"""


if __name__ == "__main__":
    out = ROOT / "experiments" / "btc_vol"
    out.mkdir(exist_ok=True)
    (out / "index.html").write_text(page(sys.argv[1]))
    print(out / "index.html")
