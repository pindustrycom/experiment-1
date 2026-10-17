"""Experiment 2's dashboard: today's prediction, how the predictions have done, and what the errors teach.

Static HTML (no scripts), built after every run and published under /exp2/ next to Experiment 1.
"""
from __future__ import annotations

import html
import json
import math
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import pandas as pd

from .live import LEVERAGE, WEIGHTS

e = html.escape
CSS = """
:root{--bg:#f7f7f5;--panel:#fff;--text:#1d1d1b;--muted:#6b6b66;--line:#e3e2dd;--accent:#2f5fd0;--lo:#2e7d4f;--hi:#c0392b;--band:#e8eefc;--grid:#ecebe6}
@media (prefers-color-scheme:dark){:root{--bg:#141413;--panel:#1d1d1b;--text:#ecebe6;--muted:#9c9b94;--line:#33322e;--accent:#7aa2ff;--lo:#6fcf97;--hi:#ff7b6b;--band:#25304a;--grid:#2a2a27}}
*{box-sizing:border-box}body{margin:0;background:var(--bg);color:var(--text);font:15px/1.5 system-ui,-apple-system,"Segoe UI",sans-serif}
a{color:var(--accent);text-decoration:none}a:hover{text-decoration:underline}
header{background:var(--panel);border-bottom:1px solid var(--line)}.wrap{max-width:1100px;margin:0 auto;padding:0 16px}
header .wrap{display:flex;flex-wrap:wrap;gap:6px 18px;align-items:center;padding-top:12px;padding-bottom:12px}
.brand{font-weight:650;font-size:17px;color:var(--text)}nav{display:flex;gap:4px;flex-wrap:wrap}
nav a{padding:4px 10px;border-radius:999px;color:var(--muted)}nav a.on{background:var(--band);color:var(--text);font-weight:600}
main{padding:22px 0 40px}h1{font-size:24px;margin:0 0 4px}h2{font-size:18px;margin:28px 0 10px}.sub{color:var(--muted);margin:0 0 18px;max-width:820px}
.muted{color:var(--muted)}.small{font-size:13px}.pos{color:var(--lo)}.neg{color:var(--hi)}
.panel{background:var(--panel);border:1px solid var(--line);border-radius:12px;padding:16px 18px}
.grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(170px,1fr));gap:10px}
.stat{background:var(--panel);border:1px solid var(--line);border-radius:10px;padding:10px 12px}.stat .k{font-size:12px;color:var(--muted)}.stat .v{font-size:20px;font-weight:650}
.stat .d{font-size:12px;color:var(--muted)}
table{border-collapse:collapse;width:100%;font-size:14px}th,td{padding:6px 8px;border-bottom:1px solid var(--line);text-align:right;white-space:nowrap}
th:first-child,td:first-child{text-align:left}th{color:var(--muted);font-weight:600;font-size:12px}
.scroll{overflow-x:auto}svg{width:100%;height:auto;display:block}svg text{fill:var(--muted);font-size:11px}
ul{margin:6px 0;padding-left:20px}details{margin-top:22px}summary{cursor:pointer;font-size:17px}details .panel{margin-top:8px}li{margin:3px 0}
footer{border-top:1px solid var(--line);color:var(--muted);font-size:13px;padding:16px 0 30px}
"""


def pct(v, d=2, sign=True) -> str:
    if v is None or (isinstance(v, float) and math.isnan(v)):
        return "—"
    s = f"{v:+.{d}f}%" if sign else f"{v:.{d}f}%"
    cls = "pos" if v > 0 else "neg" if v < 0 else ""
    return f'<span class="{cls}">{s}</span>' if sign else s


def stat(k: str, v: str, d: str = "") -> str:
    return f'<div class="stat"><div class="k">{e(k)}</div><div class="v">{v}</div><div class="d">{d}</div></div>'


def perf(r: pd.Series) -> dict:
    """r: daily % returns."""
    r = r.dropna() / 100
    if len(r) < 2:
        return {"days": len(r)}
    eq = np.cumprod(1 + r.to_numpy())
    yrs = len(r) / 252
    return {"days": len(r), "total_pct": 100 * (eq[-1] - 1), "annual_pct": 100 * (eq[-1] ** (1 / yrs) - 1) if yrs > 0.25 else None,
            "sharpe": float(r.mean() / r.std() * math.sqrt(252)) if r.std() > 0 else None,
            "max_dd_pct": float(100 * (1 - (eq / np.maximum.accumulate(eq)).min()))}


def calib(s: pd.DataFrame) -> dict:
    if s.empty:
        return {"n": 0}
    err = s["error_pp"].astype(float)
    t = err.mean() / err.std() * math.sqrt(len(s)) if len(s) > 2 and err.std() > 0 else None
    return {"n": len(s), "in68": float(s["in68"].astype(str).str.lower().eq("true").mean()),
            "in95": float(s["in95"].astype(str).str.lower().eq("true").mean()), "bias_pp": float(err.mean()), "bias_t": t,
            "mae_pp": float(err.abs().mean()), "vol_ratio": float(s["vol_ratio"].astype(float).mean()),
            "direction_right": float(s["up_right"].astype(str).str.lower().eq("true").mean())}


def lessons(c: dict, attr: dict) -> list[str]:
    out = []
    if c.get("n", 0) < 20:
        return ["Too few scored sessions to draw lessons yet; the learning factors stay close to 1 until at least 20 have been scored."]
    if c["in68"] < 0.60:
        out.append(f"Only {100 * c['in68']:.0f}% of outcomes fell inside the 68% band: the bands were too narrow, so the band multiplier is widening them.")
    elif c["in68"] > 0.76:
        out.append(f"{100 * c['in68']:.0f}% of outcomes fell inside the 68% band: the bands were too wide, so the band multiplier is narrowing them.")
    else:
        out.append(f"{100 * c['in68']:.0f}% of outcomes fell inside the 68% band (68% is the target): the bands are about right.")
    if c["bias_t"] is not None and abs(c["bias_t"]) > 2:
        out.append(f"Predictions were systematically {'too low' if c['bias_pp'] > 0 else 'too high'} by {abs(c['bias_pp']):.3f} pp a day (t = {c['bias_t']:.1f}).")
    else:
        out.append("No systematic bias: the average error is within noise.")
    # 0.8 = E|r|/s for a normal return; above that, the volatility forecast was too low
    vr = c["vol_ratio"] / math.sqrt(2 / math.pi)
    if vr > 1.15:
        out.append(f"SPY moved about {100 * (vr - 1):.0f}% more than our volatility forecast: the volatility multiplier is raising it.")
    elif vr < 0.87:
        out.append(f"SPY moved about {100 * (1 - vr):.0f}% less than our volatility forecast: the volatility multiplier is lowering it.")
    else:
        out.append("The SPY volatility forecast was well calibrated.")
    if attr:
        big = max(attr, key=attr.get)
        out.append(f"Most of the error came from the {big} sleeve (average miss {attr[big]:.3f} pp a day).")
    return out


def line_chart(series: dict[str, pd.Series], marker: str | None = None, w: int = 1000, h: int = 260, log_scale: bool = False) -> str:
    """Growth of 1 for each series (index = dates)."""
    pad_l, pad_b, pad_t = 48, 22, 10
    allv = pd.concat(series.values())
    if allv.empty:
        return ""
    x0, x1 = allv.index.min(), allv.index.max()
    tr = (lambda v: np.log(v)) if log_scale else (lambda v: v)
    lo, hi = float(tr(allv.min())), float(tr(allv.max()))
    hi = hi if hi > lo else lo + 1
    fx = lambda d: pad_l + (w - pad_l - 8) * ((d - x0) / (x1 - x0 if x1 > x0 else pd.Timedelta(days=1)))  # noqa: E731
    fy = lambda v: pad_t + (h - pad_t - pad_b) * (1 - (tr(v) - lo) / (hi - lo))  # noqa: E731
    colors = ["var(--accent)", "var(--muted)", "var(--lo)"]
    parts = [f'<svg viewBox="0 0 {w} {h}" role="img" aria-label="Growth of 1">']
    for k in range(5):
        v = math.exp(lo + (hi - lo) * k / 4) if log_scale else lo + (hi - lo) * k / 4
        y = fy(v)
        parts.append(f'<line x1="{pad_l}" x2="{w - 8}" y1="{y:.1f}" y2="{y:.1f}" stroke="var(--grid)"/><text x="4" y="{y + 4:.1f}">{v:.2f}</text>')
    for yr in range(x0.year + 1, x1.year + 1):
        x = fx(pd.Timestamp(yr, 1, 1))
        parts.append(f'<text x="{x - 14:.1f}" y="{h - 4}">{yr}</text>')
    if marker is not None and x0 <= pd.Timestamp(marker) <= x1:
        x = fx(pd.Timestamp(marker))
        parts.append(f'<line x1="{x:.1f}" x2="{x:.1f}" y1="{pad_t}" y2="{h - pad_b}" stroke="var(--hi)" stroke-dasharray="4 3"/>'
                     f'<text x="{x + 4:.1f}" y="{pad_t + 10}">live tracking starts</text>')
    for s, c in zip(series.values(), colors, strict=False):
        pts = " ".join(f"{fx(d):.1f},{fy(v):.1f}" for d, v in s.items())
        parts.append(f'<polyline fill="none" stroke="{c}" stroke-width="1.6" points="{pts}"/>')
    parts.append("</svg>")
    legend = " ".join(f'<span style="color:{c}">■</span> {e(n)}' for n, c in zip(series, colors))
    return "".join(parts) + f'<div class="small muted">{legend}</div>'


def band_chart(s: pd.DataFrame, w: int = 1000, h: int = 220) -> str:
    """Last sessions: the predicted 68% band (bar) and what happened (dot)."""
    if s.empty:
        return ""
    lo = min(s["band_lo"].astype(float).min(), s["real_ret"].astype(float).min())
    hi = max(s["band_hi"].astype(float).max(), s["real_ret"].astype(float).max())
    pad_l, pad_b, pad_t = 48, 22, 8
    n = len(s)
    bw = (w - pad_l - 8) / n
    fy = lambda v: pad_t + (h - pad_t - pad_b) * (1 - (v - lo) / (hi - lo if hi > lo else 1))  # noqa: E731
    p = [f'<svg viewBox="0 0 {w} {h}" role="img" aria-label="Predicted band and outcome">']
    for k in range(5):
        v = lo + (hi - lo) * k / 4
        p.append(f'<line x1="{pad_l}" x2="{w - 8}" y1="{fy(v):.1f}" y2="{fy(v):.1f}" stroke="var(--grid)"/><text x="4" y="{fy(v) + 4:.1f}">{v:+.1f}%</text>')
    for i, (_, r) in enumerate(s.iterrows()):
        x = pad_l + i * bw
        y1, y2 = fy(float(r["band_hi"])), fy(float(r["band_lo"]))
        p.append(f'<rect x="{x + bw * 0.2:.1f}" y="{y1:.1f}" width="{bw * 0.6:.1f}" height="{max(y2 - y1, 1):.1f}" fill="var(--band)"/>')
        if r["status"] == "scored":
            ok = str(r["in68"]).lower() == "true"
            p.append(f'<circle cx="{x + bw / 2:.1f}" cy="{fy(float(r["real_ret"])):.1f}" r="3" fill="{"var(--lo)" if ok else "var(--hi)"}"/>')
        if i % max(1, n // 6) == 0:
            p.append(f'<text x="{x:.1f}" y="{h - 4}">{r["date"][5:]}</text>')
    p.append("</svg>")
    return "".join(p) + '<div class="small muted">Shaded bar: the morning\'s 68% band. Dot: the day\'s result (green inside the band, red outside).</div>'


def build(site: Path, L: pd.DataFrame, plans: dict, df: pd.DataFrame) -> dict:
    site.mkdir(parents=True, exist_ok=True)
    (site / "data").mkdir(exist_ok=True)
    L = L.copy()
    L["d"] = pd.to_datetime(L["date"])
    scored = L[L["status"] == "scored"]
    live = L[L["source"] == "live"]
    live_scored = live[live["status"] == "scored"]
    back = scored[scored["source"] == "backfill"]
    first_live = live["date"].min() if not live.empty else None
    spy = (df["r"] * 100).reindex(scored["d"]).fillna(0.0)
    spy.index = scored.index
    perf_all = {"portfolio": perf(scored["real_ret"].astype(float)), "spy": perf(spy)}
    perf_live = {"portfolio": perf(live_scored["real_ret"].astype(float)),
                 "spy": perf((df["r"] * 100).reindex(live_scored["d"]).reset_index(drop=True))}
    recent = scored.tail(120)
    attr = {}
    for k, name in (("hrp", "stock book"), ("regime", "SPY regime"), ("varswap", "variance swap")):
        if not recent.empty:
            attr[name] = float((recent[f"real_{k}"].astype(float) - recent[f"exp_{k}"].astype(float)).abs().mean())
    c_recent, c_live, c_back = calib(recent), calib(live_scored), calib(back)
    latest = L.iloc[-1].to_dict() if not L.empty else {}
    summary = {"built": datetime.now(UTC).strftime("%Y-%m-%d %H:%M UTC"), "first_live": first_live,
               "latest": {k: (None if isinstance(v, float) and math.isnan(v) else v) for k, v in latest.items() if k != "d"},
               "performance_since_2019": perf_all, "performance_live": perf_live, "calibration_last_120": c_recent,
               "calibration_live": c_live, "calibration_backfill": c_back, "error_by_sleeve_last_120": attr,
               "lessons": lessons(c_recent, attr), "weights": WEIGHTS, "leverage": LEVERAGE,
               "current_plan": {k: v for k, v in plans.get(str(pd.Timestamp(latest.get("date", "2000-01-01")).to_period("M")), {}).items() if k != "weights"}}
    (site / "data" / "summary.json").write_text(json.dumps(summary, indent=1, default=str))
    L.drop(columns=["d"]).to_csv(site / "data" / "predictions.csv", index=False, float_format="%.6g")
    (site / "index.html").write_text(page(summary, L, scored, first_live, df))
    (site / ".nojekyll").write_text("")
    return {"rows": len(L), "live_rows": len(live), "scored": len(scored)}


def page(S: dict, L: pd.DataFrame, scored: pd.DataFrame, first_live: str | None, df: pd.DataFrame) -> str:
    lt = S["latest"]
    plan = S["current_plan"]
    if lt:
        status = lt.get("status")
        head = f"{'Prediction' if lt.get('source') == 'live' else 'Replayed prediction'} for {e(str(lt['date']))}"
        when = f"made {e(str(lt.get('made_at') or ''))[:16].replace('T', ' ')} UTC, from the close of {e(str(lt['last_close']))}" if lt.get("made_at") else \
            f"replayed from the close of {e(str(lt['last_close']))}"
        cards = stat("Expected return today", pct(lt["exp_ret"], 3), f"68% band {lt['band_lo']:+.2f}% to {lt['band_hi']:+.2f}%") + \
            stat("Chance the day ends up", f"{100 * float(lt['p_up']):.0f}%") + \
            stat("SPY volatility forecast", f"{100 * float(lt['sigma']):.2f}%", f"a day; {100 * float(lt['sigma']) * math.sqrt(252):.0f}% a year") + \
            stat("Market regime", "calm" if float(lt["p_hi"]) < 0.5 else "turbulent", f"turbulent probability {100 * float(lt['p_hi']):.0f}%")
        if status == "scored":
            ok = str(lt["in68"]).lower() == "true"
            cards += stat("What happened", pct(float(lt["real_ret"]), 3), f"error {float(lt['error_pp']):+.3f} pp, {'inside' if ok else 'outside'} the band")
        else:
            cards += stat("What happened", "pending" if status == "pending" else e(str(status)), "scored after the close")
        pos = (f"<ul><li><b>SPY:</b> {'held' if str(lt['regime_in']).lower() == 'true' else 'out (cash)'} "
               f"({100 * LEVERAGE * WEIGHTS['regime']:.0f}% of capital when held)</li>"
               f"<li><b>Variance swap:</b> {'short, struck at VIX ' + format(float(lt['K']), '.1f') if str(lt['varswap_on']).lower() == 'true' else 'off this month'}"
               f"{'' if not plan.get('fvar') else f' (our forecast for the month: {math.sqrt(plan[chr(102) + chr(118) + chr(97) + chr(114)]):.1f})'}</li>"
               f"<li><b>Stock book:</b> HRP over S&amp;P 500 members without the 10% most collapse-prone "
               f"({100 * LEVERAGE * WEIGHTS['hrp']:.0f}% of capital, beta {float(lt['hrp_beta']):.2f}), formed {e(str(plan.get('formed', '—')))}</li></ul>")
        today = f'<div class="panel"><b>{head}</b> <span class="muted small">{when}</span><div class="grid" style="margin-top:10px">{cards}</div>{pos}</div>'
    else:
        today = '<div class="panel">No predictions yet.</div>'
    if not first_live:
        today += '<p class="muted small">Live tracking starts with the next trading morning. Until then the history below is a replay of 2019 to today with the same rules.</p>'
    pa, ps = S["performance_since_2019"]["portfolio"], S["performance_since_2019"]["spy"]
    pl, pls = S["performance_live"]["portfolio"], S["performance_live"]["spy"]
    def facts(p: dict) -> str:
        return f"total {p.get('total_pct', 0):+.0f}%, Sharpe {p.get('sharpe') or 0:.2f}, worst drop {p.get('max_dd_pct', 0):.0f}%"
    perf_cards = stat("Since 2019: portfolio", pct(pa.get("annual_pct"), 1) + " a year", facts(pa)) + \
        stat("Since 2019: SPY", pct(ps.get("annual_pct"), 1) + " a year", facts(ps)) + \
        stat("Live: portfolio", pct(pl.get("total_pct"), 2) if pl.get("days", 0) else "—", f"{pl.get('days', 0)} sessions scored") + \
        stat("Live: SPY", pct(pls.get("total_pct"), 2) if pls.get("days", 0) else "—", "same sessions")
    eq = {"Portfolio": (1 + scored["real_ret"].astype(float) / 100).cumprod().set_axis(pd.to_datetime(scored["date"])),
          "SPY": (1 + (df["r"]).reindex(pd.to_datetime(scored["date"])).fillna(0.0)).cumprod()}
    c = S["calibration_last_120"]
    cl = S["calibration_live"]

    def crow(name, x):
        if not x.get("n"):
            return f"<tr><td>{name}</td><td colspan=6 class='muted'>none yet</td></tr>"
        return (f"<tr><td>{name}</td><td>{x['n']}</td><td>{100 * x['in68']:.0f}%</td><td>{100 * x['in95']:.0f}%</td><td>{x['bias_pp']:+.3f}</td>"
                f"<td>{x['mae_pp']:.3f}</td><td>{100 * x['direction_right']:.0f}%</td></tr>")
    ctab = ("<div class='scroll'><table><tr><th>Sessions</th><th>Scored</th><th>Inside 68% band</th><th>Inside 95% band</th><th>Average error (pp)</th>"
            "<th>Average miss (pp)</th><th>Direction right</th></tr>" + crow("Last 120", c) + crow("Live only", cl) +
            crow("Replay 2019 on", S["calibration_backfill"]) + "</table></div>")
    vm = float(lt.get("vol_mult", 1) or 1) if lt else 1.0
    bm = float(lt.get("band_mult", 1) or 1) if lt else 1.0
    learn = "".join(f"<li>{e(x)}</li>" for x in S["lessons"])
    misses = scored.tail(250).assign(az=lambda d: d["z"].astype(float).abs()).nlargest(8, "az")
    mtab = "".join(f"<tr><td>{e(r['date'])}</td><td>{float(r['exp_ret']):+.2f}%</td><td>{float(r['real_ret']):+.2f}%</td><td>{float(r['error_pp']):+.2f}</td>"
                   f"<td>{float(r['z']):+.1f}</td><td>{float(r['spy_ret']):+.2f}%</td><td>{100 * float(r['sigma']):.2f}%</td>"
                   f"<td>{'calm' if float(r['p_hi']) < 0.5 else 'turbulent'}</td><td>{'on' if str(r['varswap_on']).lower() == 'true' else 'off'}</td></tr>"
                   for _, r in misses.iterrows())
    last = L.tail(25).iloc[::-1]
    ltab = "".join(f"<tr><td>{e(r['date'])}</td><td>{e(str(r['source']))}</td><td>{float(r['exp_ret']):+.3f}%</td><td>{float(r['band_lo']):+.2f} to {float(r['band_hi']):+.2f}</td>"
                   + (f"<td>{float(r['real_ret']):+.3f}%</td><td>{float(r['error_pp']):+.3f}</td><td>{'yes' if str(r['in68']).lower() == 'true' else 'no'}</td>"
                      if r["status"] == "scored" else f"<td colspan=3 class='muted'>{e(str(r['status']))}</td>") + "</tr>" for _, r in last.iterrows())
    recent_band = scored.tail(60)
    if not L.empty and L.iloc[-1]["status"] != "scored":
        recent_band = pd.concat([recent_band.tail(59), L.tail(1)])
    attr = "".join(f"<li>{e(k)}: average miss {v:.3f} pp a day</li>" for k, v in S["error_by_sleeve_last_120"].items())
    return f"""<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<meta name="robots" content="noindex"><title>Experiment 2: volatility-timed portfolio</title><style>{CSS}</style></head><body>
<header><div class="wrap"><a class="brand" href="../index.html">Market experiments</a><nav><a href="../exp1/index.html">Experiment 1</a>
<a class="on" href="index.html">Experiment 2</a><a href="../index.html">Paper portfolio</a></nav></div></header>
<main class="wrap"><h1>Experiment 2: volatility-timed portfolio</h1>
<p class="sub">Sells SPY volatility when it is overpriced, holds SPY in calm markets, plus a stock book. Predicts each morning, scores each evening.</p>
{today}
<h2>Track record</h2><div class="grid">{perf_cards}</div>
<div class="panel" style="margin-top:12px">{line_chart(eq, first_live, log_scale=True)}</div>
<h2>Predictions against outcomes</h2><div class="panel">{band_chart(recent_band)}</div>
<details><summary><b>What the errors teach</b> <span class="muted small">(details)</span></summary><div class="panel"><ul>{learn}</ul>
<p class="small muted">Learning in use for the next prediction: volatility multiplier {vm:.2f} (scales our SPY volatility forecast) and band
multiplier {bm:.2f} (scales the band to how wide errors really were). Both come from the last 120 scored sessions, shrunk towards 1.</p>
{ctab}<p class="small">Where the misses come from (last 120 sessions):</p><ul class="small">{attr}</ul>
<p class="small">Biggest misses (last year), with their context:</p><div class="scroll"><table><tr><th>Date</th><th>Predicted</th><th>Result</th><th>Error (pp)</th>
<th>Error in bands</th><th>SPY</th><th>Vol forecast</th><th>Regime</th><th>Var swap</th></tr>{mtab}</table></div></div></details>
<details><summary><b>Prediction log</b> <span class="muted small">(last 25)</span></summary>
<div class="panel"><div class="scroll"><table><tr><th>Date</th><th>Source</th><th>Predicted</th><th>68% band</th><th>Result</th>
<th>Error (pp)</th><th>In band</th></tr>{ltab}</table></div>
<p class="small muted">Full log: <a href="data/predictions.csv">predictions.csv</a>; summary: <a href="data/summary.json">summary.json</a>.
Live predictions are also committed to the <code>exp2-ledger</code> branch before the market opens.</p></div></details>
<details><summary><b>Caveats</b></summary><div class="panel small"><ul>
<li>Mix ({100 * WEIGHTS['hrp']:.1f}% stocks, {100 * WEIGHTS['regime']:.1f}% SPY, {100 * WEIGHTS['varswap']:.1f}% variance swap, ×{LEVERAGE}) fixed on 2005–2018 data;
best of ~50 tries, so expect live to be weaker.</li>
<li>The variance swap is synthetic (VIX² against realised variance) and not tradable by retail investors. A crash with no warning would hit it.</li>
<li>No interest on cash; the stock book uses today's S&amp;P 500 members, which flatters the replay.</li></ul></div></details></main>
<footer><div class="wrap">Paper tracking only: nothing here is investment advice and no orders are placed anywhere. Built {e(S['built'])}.</div></footer></body></html>"""
