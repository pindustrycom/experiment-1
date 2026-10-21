"""Daily paper tracking of the reel's rule: SPY, 3-minute bars, 9 EMA crossing the session VWAP, always in after the
first cross (long above VWAP, short below), flat at the close. Run after each close: it replays the day's 1-minute bars
and records the trades the rule would have made. 0.3 bp is charged per side. No orders are placed anywhere.

Writes into the site folder:
  data/daily.csv     one row per session from START: date, ret (fraction, USD, after costs), trades
  data/trades.csv    every trade: day, side, entry/exit time and price, return after costs
  data/summary.json  latest day, position text, and the 2019-2026 backtest by year (if results.json is given)
  index.html         the Exp7 page (the hub adds the 1000 CAD box and the site theme)
"""
from __future__ import annotations

import html
import json
from datetime import UTC, datetime
from pathlib import Path

import pandas as pd

from . import bars, strategy

START = "2026-10-19"
TICKER, MINUTES, VARIANT, COST_BP = "SPY", 3, "flip", 0.3
BACKTEST_ROWS = [("SPY 3m flip cost0.3bp", "The reel's rule, SPY 3-minute"), ("SPY 3m long cost0.3bp", "Long side only"),
                 ("SPY open-to-close hold", "Holding SPY from open to close")]
e = html.escape


def replay(d1: pd.DataFrame, start: str = START) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Trades and day returns from `start` on. Indicators use all earlier bars (the EMA carries across days)."""
    d = strategy.indicators(strategy.resample(d1, MINUTES))
    tr = strategy.trades(d[d.day >= start], VARIANT) if (d.day >= start).any() else strategy.trades(d.iloc[:0], VARIANT)
    days = pd.Index(sorted(d.day[d.day >= start].unique()))
    r = strategy.daily(tr, days, COST_BP)
    daily = pd.DataFrame({"date": days, "ret": r.to_numpy(), "trades": tr.groupby("day").size().reindex(days, fill_value=0).to_numpy()})
    tr = tr.assign(ret=tr.ret - 2 * COST_BP / 1e4)
    return daily, tr


def _ema_side(d1: pd.DataFrame) -> str:
    d = strategy.indicators(strategy.resample(d1, MINUTES))
    last = d.iloc[-1]
    return "above" if last.ema > last.vwap else "below"


def _fmt_t(t) -> str:
    return pd.Timestamp(t).strftime("%H:%M")


def page(daily: pd.DataFrame, tr: pd.DataFrame, summary: dict) -> str:
    last_day = daily.date.iloc[-1] if len(daily) else None
    today = tr[tr.day == last_day] if last_day else tr.iloc[:0]
    rows = "".join(f"<tr><td>{'long' if t.side > 0 else 'short'}</td><td>{_fmt_t(t.entry_time)}</td><td>{t.entry:.2f}</td>"
                   f"<td>{_fmt_t(t.exit_time)}</td><td>{t.exit:.2f}</td><td class=\"{'pos' if t.ret > 0 else 'neg'}\">{100 * t.ret:+.2f}%</td></tr>"
                   for t in today.itertuples())
    today_html = (f"<table><tr><th>Side</th><th>In</th><th>Price</th><th>Out</th><th>Price</th><th>Result</th></tr>{rows}</table>"
                  if rows else '<p class="muted">No trade: the EMA and VWAP did not cross.</p>') if last_day else '<p class="muted">Tracking starts with the first session after ' + e(START) + '.</p>'
    hist = "".join(f"<tr><td>{r.date}</td><td>{r.trades}</td><td class=\"{'pos' if r.ret > 0 else 'neg' if r.ret < 0 else ''}\">{100 * r.ret:+.2f}%</td></tr>"
                   for r in daily.iloc[::-1].itertuples())
    bt = summary.get("backtest") or {}
    years = sorted({y for v in bt.values() for y in v if y != "all"})
    bt_rows = "".join(f"<tr><td>{e(label)}</td>" + "".join(f"<td>{v[y]['annual_pct']:+.1f}</td>" if v.get(y) else "<td></td>" for y in years)
                      + f"<td><b>{v['all']['annual_pct']:+.1f}</b></td><td>{v['all']['max_drawdown_pct']:.0f}</td></tr>"
                      for label, v in bt.items())
    bt_html = (f'<div class="scroll"><table><tr><th>2019 to 2026, % a year</th>{"".join(f"<th>{y}</th>" for y in years)}<th>All</th><th>Worst drop %</th></tr>'
               f"{bt_rows}</table></div>") if bt else ""
    built = datetime.now(UTC).strftime("%Y-%m-%d %H:%M UTC")
    return f"""<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<meta name="robots" content="noindex"><title>Experiment 7 · VWAP + 9 EMA</title><style>
body{{margin:0;font:15px/1.5 system-ui,sans-serif}}a{{text-decoration:none}}.wrap{{max-width:1100px;margin:0 auto;padding:0 16px}}header .wrap{{padding:12px 16px}}
header a{{margin-right:14px;color:var(--muted)}}.brand{{font-weight:650;color:var(--text)!important}}h1{{font-size:24px;margin:0 0 4px}}h2{{font-size:18px;margin:22px 0 8px}}
.muted{{color:var(--muted)}}.panel{{padding:14px 16px;margin:14px 0}}table{{border-collapse:collapse;width:100%;font-variant-numeric:tabular-nums}}
th,td{{text-align:left;padding:6px 8px;border-bottom:1px solid var(--line)}}.scroll{{overflow-x:auto}}.pos{{color:var(--lo)}}.neg{{color:var(--hi)}}
details summary{{cursor:pointer}}footer{{color:var(--muted);font-size:13px;padding:16px 0 30px}}</style></head><body>
<header><div class="wrap"><a class="brand" href="index.html">Experiment 7 · VWAP + 9 EMA day trades</a></div></header>
<main class="wrap"><h1>Experiment 7 · VWAP + 9 EMA cross</h1>
<p class="muted">From an Instagram reel: on a 3-minute SPY chart, go long when the 9-period EMA crosses above the day's VWAP and short when it crosses
below, and be flat at the close. Each evening the day's 1-minute bars are replayed and the trades the rule would have made are recorded, at
0.3 bp per side. Signals are read at a bar's close and filled at the next bar's open.</p>
<div class="panel"><h2 style="margin-top:0">{'Trades on ' + e(last_day) if last_day else 'Trades'}</h2>{today_html}</div>
<div class="panel"><h2 style="margin-top:0">Every session</h2><div class="scroll"><table><tr><th>Date</th><th>Trades</th><th>Day result</th></tr>{hist}</table></div></div>
<div class="panel"><h2 style="margin-top:0">Backtest before tracking</h2>{bt_html}
<p class="muted">Tiingo IEX 1-minute bars, 2019 to October 2026. The 2020 and 2022 numbers looked like 20% a year, but over the whole period the rule made about
what holding SPY from open to close made, and 2025 and 2026 were flat to negative. See research notes in the repository (experiments/exp7_vwap).</p></div>
</main><footer><div class="wrap">Fake money: no orders are placed anywhere. Built {built}.</div></footer></body></html>"""


def run(data: Path, site: Path, results: Path | None = None) -> dict:
    d1 = bars.load(data, TICKER)
    daily, tr = replay(d1)
    (site / "data").mkdir(parents=True, exist_ok=True)
    daily.to_csv(site / "data" / "daily.csv", index=False)
    tr.to_csv(site / "data" / "trades.csv", index=False)
    bt = {}
    if results and results.exists():
        res = json.loads(results.read_text())
        bt = {label: res[k] for k, label in BACKTEST_ROWS if k in res}
    last = daily.iloc[-1] if len(daily) else None
    pos = (f"{last.date}: {int(last.trades)} trade{'s' if last.trades != 1 else ''}, day {100 * last.ret:+.2f}%; flat overnight"
           if last is not None else f"Tracking starts {START}; last bars {d1.index[-1].date()}")
    summary = {"start": START, "last_bars": str(d1.index[-1]), "last_day": None if last is None else last.date, "position": pos,
               "ema_vs_vwap_at_close": _ema_side(d1), "backtest": bt}
    (site / "data" / "summary.json").write_text(json.dumps(summary, indent=1, default=str))
    (site / "index.html").write_text(page(daily, tr, summary))
    return {"sessions": len(daily), "trades": len(tr), "position": pos, "last_bars": summary["last_bars"]}
