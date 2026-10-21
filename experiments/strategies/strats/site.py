"""The strategy pots' page and data files, published under /strategies/ by pages.yml:
  data/daily.csv     one row per session: each pot's return (fraction, after costs); read by experiments/hub/portfolio.py
  data/summary.json  each pot's current positions in words, and the replay results
  index.html         one section per pot, grouped long-term / short-term
"""
from __future__ import annotations

import csv
import html
import json
from pathlib import Path

from . import rules

e = html.escape
CSS = """
*{box-sizing:border-box}body{margin:0;background:#000;color:#f5f5f7;font:15px/1.5 -apple-system,BlinkMacSystemFont,"Segoe UI",system-ui,sans-serif}
.wrap{max-width:1100px;margin:0 auto;padding:0 16px}main.wrap{padding-top:22px;padding-bottom:40px}h1{font-size:28px;margin:0 0 4px}
h2{font-size:22px;margin:30px 0 10px}h3{font-size:18px;margin:0 0 4px}.muted{color:#98989f}.small{font-size:13px}
.panel{border-radius:20px;padding:16px 18px;margin:12px 0}.stats{display:grid;grid-template-columns:repeat(auto-fit,minmax(120px,1fr));gap:8px;margin:10px 0}
.stat{border-radius:14px;padding:8px 10px}.stat .k{font-size:12px;color:#98989f}.stat .v{font-size:18px;font-weight:650}
.pos{color:#30d158}.neg{color:#ff453a}table{border-collapse:collapse;width:100%;font-size:13px}th,td{text-align:right;padding:4px 6px;border-bottom:1px solid rgba(255,255,255,.08)}
th:first-child,td:first-child{text-align:left}.scroll{overflow-x:auto}
"""


def pct(v: float | None, sign: bool = True) -> str:
    if v is None:
        return "–"
    cls = "pos" if v > 0.00005 else "neg" if v < -0.00005 else ""
    return f'<span class="{cls}">{v:+.1%}</span>' if sign else f"{v:.1%}"


def position_text(key: str, row: dict | None, events: dict) -> str:
    if not row:
        return "No positions yet: the first book is formed after the next close."
    w = row["w"]
    if not w:
        return f"In cash after the {row['date']} close: nothing qualifies."
    inv = sum(abs(x) for x in w.values())
    if key == "trend":
        legs = sorted(w.items(), key=lambda kv: -abs(kv[1]))
        return f"From the {row['date']} close: " + ", ".join(f"{'long' if x > 0 else 'short'} {t} {abs(x):.0%}" for t, x in legs)
    names = sorted(w)
    left = events.get(key, {})
    tail = "" if key == "reversal" else f"; next to leave: {min(left, key=left.get)} in {min(left.values())} sessions" if left else ""
    return (f"From the {row['date']} close: {len(names)} stocks, {inv:.0%} invested, the rest in cash: "
            + ", ".join(names[:25]) + (" …" if len(names) > 25 else "") + tail)


def write(out: Path, state: dict, summary_replay: dict | None, info: dict) -> dict:
    (out / "data").mkdir(parents=True, exist_ok=True)
    pots = state.get("pots", {})
    dates = sorted({r["date"] for rows in pots.values() for r in rows})
    by = {k: {r["date"]: r["ret"] for r in rows} for k, rows in pots.items()}
    with (out / "data" / "daily.csv").open("w", newline="") as fh:
        wr = csv.writer(fh)
        wr.writerow(["date", *rules.STRATEGIES])
        for d in dates:
            wr.writerow([d, *(f"{by.get(k, {}).get(d):.8f}" if d in by.get(k, {}) else "" for k in rules.STRATEGIES)])
    summary = {"start": state.get("start"), "last": dates[-1] if dates else None, "info": info,
               "strategies": {k: {"name": n, "category": c, "what": w,
                                  "position": position_text(k, (pots.get(k) or [None])[-1], state.get("events", {}))}
                              for k, (n, c, w) in rules.STRATEGIES.items()},
               "replay": summary_replay}
    (out / "data" / "summary.json").write_text(json.dumps(summary, indent=1))
    (out / "index.html").write_text(page(summary, pots))
    return {"sessions": len(dates), "last": summary["last"]}


def _replay_block(key: str, rp: dict | None) -> str:
    if not rp or key not in rp or not rp[key]:
        if key in ("pead", "insider"):
            return '<p class="muted small">No replay: the free data does not reach back far enough for earnings and insider events, so this pot is tested forward only.</p>'
        return '<p class="muted small">The replay appears after the first run.</p>'
    s, spy = rp[key], rp.get("SPY") or {}
    years = sorted(s.get("years", {}))
    yt = ("<div class=\"scroll\"><table><tr><th>Year</th>" + "".join(f"<th>{y}</th>" for y in years) + "</tr>"
          f"<tr><td>{e(rules.STRATEGIES[key][0])}</td>" + "".join(f"<td>{pct(s['years'][y])}</td>" for y in years) + "</tr>"
          "<tr><td>SPY</td>" + "".join(f"<td>{pct(spy.get('years', {}).get(y))}</td>" for y in years) + "</tr></table></div>")
    note = {"reversal": " Uses today's S&P 500 members, so it looks better than it really was.",
            "best_ideas": " The 23 funds were picked because they are famous today, which flatters it; most of the gain is Scion's GameStop in 2020-21.",
            "conviction": " The funds were picked because they are famous today, which flatters it.",
            "pelosi": " Pelosi was picked because her record is famous today; about 60 buys, mostly big tech, and she trades calls, we copy with shares.",
            }.get(key, "")
    return (f'<p class="small muted">Replay {e(rp["from"])} to {e(rp["to"])}, same rule, 1 bp per side.{note}</p><div class="stats">'
            f'<div class="stat"><div class="k">Per year</div><div class="v">{pct(s["cagr"])}</div></div>'
            f'<div class="stat"><div class="k">SPY per year</div><div class="v">{pct(spy.get("cagr"))}</div></div>'
            f'<div class="stat"><div class="k">Sharpe (SPY)</div><div class="v">{s["sharpe"]:.2f} ({spy.get("sharpe", 0):.2f})</div></div>'
            f'<div class="stat"><div class="k">Worst drop (SPY)</div><div class="v">{pct(s["max_dd"])} ({pct(spy.get("max_dd"))})</div></div>'
            f'<div class="stat"><div class="k">Correlation with SPY</div><div class="v">{rp["corr_spy"][key]:+.2f}</div></div></div>{yt}')


def page(summary: dict, pots: dict) -> str:
    rp = summary.get("replay")
    parts = []
    for cat in (rules.LONG_TERM, rules.SHORT_TERM):
        parts.append(f"<h2>{cat}</h2>")
        for k, s in summary["strategies"].items():
            if s["category"] != cat:
                continue
            rows = pots.get(k) or []
            live = (rows[-1]["index"] - 1) if rows else None
            parts.append(f'<section class="panel" id="{k}"><h3>{e(s["name"])}</h3><p class="muted">{e(s["what"])}</p>'
                         f'<p class="small">Paper pot since {e(rows[0]["date"]) if rows else "the next close"}: {pct(live)} in USD (the Markets page shows it in CAD)</p>'
                         f'<p class="small">{e(s["position"])}</p>{_replay_block(k, rp)}</section>')
    return f"""<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<meta name="robots" content="noindex,nofollow"><title>Strategy pots</title><style>{CSS}</style></head><body>
<main class="wrap"><h1>Strategy pots</h1><p class="muted">{len(summary["strategies"])} rules, each traded every day on paper with its own
1,000 CAD from its first close. Long-term pots hold positions for weeks or months; short-term ones for days. Last close: {e(summary.get("last") or "none yet")}.</p>
{"".join(parts)}<p class="muted small">Fake money: no orders are placed anywhere. Prices from Yahoo, earnings and insider filings from Finnhub,
13Fs from SEC EDGAR, Pelosi's trades from the House Clerk.</p></main></body></html>"""
