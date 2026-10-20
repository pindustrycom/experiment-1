"""Static pages for Experiment 2 (plain HTML, inline SVG, no JavaScript), matching Exp1's look."""
from __future__ import annotations

import html
import json
import math
from datetime import UTC, date, datetime
from pathlib import Path

import pandas as pd

from . import config

e = html.escape
REPO_URL = "https://github.com/pindustrycom/experiment-1"
MODEL_LABEL = {"ssm": "Pruned S4D (this model)", "ekf": "EKF + SSM (filtered latent state)", "zero": "No change", "drift": "Average drift", "momentum": "20-day momentum",
               "ridge": "Ridge regression, same inputs"}
TARGET_LABEL = {"y_sector": f"{config.TARGET} return", "y_relative": f"{config.TARGET} minus {config.BENCHMARK}"}

CSS = """
:root{--bg:#f7f7f5;--panel:#fff;--text:#1d1d1b;--muted:#6b6b66;--line:#e3e2dd;--accent:#2f5fd0;
--lo:#2e7d4f;--md:#b7791f;--hi:#c0392b;--lo-bg:#e6f3ea;--md-bg:#fbf0dc;--hi-bg:#fbe5e2;--a:#2f5fd0;--b:#d0632f;--band:#f1f0ec}
@media (prefers-color-scheme:dark){:root{--bg:#141413;--panel:#1d1d1b;--text:#ecebe6;--muted:#9c9b94;--line:#33322e;
--accent:#7aa2ff;--lo:#6fcf97;--md:#f2c14e;--hi:#ff7b6b;--lo-bg:#1d3326;--md-bg:#3a2f15;--hi-bg:#3d1f1b;--a:#7aa2ff;--b:#ff9a66;--band:#252522}}
*{box-sizing:border-box}body{margin:0;background:var(--bg);color:var(--text);font:15px/1.5 system-ui,-apple-system,"Segoe UI",sans-serif}
a{color:var(--accent);text-decoration:none}a:hover{text-decoration:underline}
header{background:var(--panel);border-bottom:1px solid var(--line)}
.wrap{max-width:1100px;margin:0 auto;padding:0 16px}
header .wrap{display:flex;flex-wrap:wrap;gap:6px 22px;align-items:center;padding-top:12px;padding-bottom:12px}
header .brand{font-weight:650;font-size:17px;color:var(--text)}nav a{color:var(--muted);margin-right:16px}nav a.on{color:var(--text);font-weight:600}
.hubbar{font-size:13px;color:var(--muted);padding-top:8px}
main{padding:22px 0 40px}h1{font-size:24px;margin:0 0 4px}h2{font-size:18px;margin:28px 0 10px}h3{font-size:15px;margin:18px 0 6px}
.sub{color:var(--muted);margin:0 0 18px}.muted{color:var(--muted)}
.banner{border-radius:8px;padding:10px 14px;margin:0 0 18px;border:1px solid var(--line)}.banner.warn{background:var(--md-bg)}
.tiles{display:grid;grid-template-columns:repeat(auto-fit,minmax(170px,1fr));gap:12px;margin-bottom:18px}
.tile,.card{background:var(--panel);border:1px solid var(--line);border-radius:10px;padding:12px 14px}
.tile .k{color:var(--muted);font-size:13px}.tile .v{font-size:22px;font-weight:650}.tile .s{color:var(--muted);font-size:13px}
.card{margin-bottom:14px}.scroll{overflow-x:auto}
table{width:100%;border-collapse:collapse;background:var(--panel)}th,td{padding:8px 10px;border-bottom:1px solid var(--line);text-align:left;vertical-align:top}
th{font-size:12px;color:var(--muted);font-weight:600;text-transform:uppercase;letter-spacing:.03em;white-space:nowrap}
td.n,th.n{text-align:right;font-variant-numeric:tabular-nums;white-space:nowrap}tr.me td{font-weight:600}
.pos{color:var(--lo)}.neg{color:var(--hi)}
.chart{width:100%;height:auto;display:block}.chart .grid{stroke:var(--line);stroke-width:1}.chart .axis{fill:var(--muted);font-size:11px}
.chart polyline{fill:none;stroke-width:1.8}.chart .sa{stroke:var(--a)}.chart .sb{stroke:var(--b)}.chart circle.sa{fill:var(--a)}.chart circle.sb{fill:var(--b)}
.chart .zero{stroke:var(--muted);stroke-width:1}.chart .bandfill{fill:var(--band)}.chart .barp{fill:var(--lo)}.chart .barn{fill:var(--hi)}
.legend span{display:inline-block;margin-right:14px;font-size:13px}.legend i{display:inline-block;width:12px;height:3px;vertical-align:middle;margin-right:5px}
.grid2{display:grid;grid-template-columns:repeat(auto-fit,minmax(320px,1fr));gap:14px}
footer{border-top:1px solid var(--line);color:var(--muted);font-size:13px;padding:16px 0 30px}
"""


def page(title: str, active: str, body: str, built: str) -> str:
    nav = "".join(f'<a href="{href}" class="{"on" if key == active else ""}">{label}</a>'
                  for key, href, label in [("home", "index.html", "Overview"), ("method", "method.html", "Method")])
    return f"""<!doctype html><html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1"><meta name="robots" content="noindex">
<title>{e(title)} · Experiment 2</title><style>{CSS}</style></head><body>
<header><div class="wrap"><a class="brand" href="index.html">Exp 2 · Tech sector state-space model</a><nav>{nav}</nav></div></header>
<main class="wrap">{body}</main>
<footer><div class="wrap">Research experiment that forward-tests a model's predictions for the US tech sector. Paper tracking only:
nothing here is investment advice and no orders are placed anywhere. Built {e(built)}. <a href="{REPO_URL}">Source</a>.</div></footer></body></html>"""


def pct(v, d=2, sign=True):
    if v is None or (isinstance(v, float) and not math.isfinite(v)):
        return "—"
    return f"{v:+.{d}f}%" if sign else f"{v:.{d}f}%"


def num(v, d=2):
    if v is None or (isinstance(v, float) and not math.isfinite(v)):
        return "—"
    return f"{v:.{d}f}"


def signed(v, d=2):
    if v is None or not math.isfinite(v):
        return '<span class="muted">—</span>'
    cls = "pos" if v > 0 else "neg" if v < 0 else "muted"
    return f'<span class="{cls}">{v:+.{d}f}%</span>'


def logret_to_pct(v):
    return None if v is None else 100 * (math.exp(v / 100) - 1)


# ---------------------------------------------------------------- charts

def chart(series, height=220, unit="", bands=None, zero=False, bars=None) -> str:
    """series: [(css class, [(iso date, value|None)])]; bands: [(iso date, lo, hi)]; bars: [(iso date, value)]."""
    dates = sorted({d for _, pts in series for d, v in pts if v is not None} | {d for d, *_ in (bands or [])} | {d for d, _ in (bars or [])})
    if len(dates) < 2:
        return '<p class="muted">Not enough data to draw this chart yet.</p>'
    vals = [v for _, pts in series for _, v in pts if v is not None] + [x for _, lo, hi in (bands or []) for x in (lo, hi)] + \
           [v for _, v in (bars or [])] + ([0.0] if zero or bars else [])
    lo, hi = min(vals), max(vals)
    if hi - lo < 1e-9:
        lo, hi = lo - 1, hi + 1
    pad = (hi - lo) * 0.06
    lo, hi = lo - pad, hi + pad
    W, H, L, R, T, B = 760, height, 48, 10, 10, 26
    t0, t1 = date.fromisoformat(dates[0]).toordinal(), date.fromisoformat(dates[-1]).toordinal()
    span = max(1, t1 - t0)

    def x(d):
        return L + (W - L - R) * (date.fromisoformat(d).toordinal() - t0) / span

    def y(v):
        return T + (H - T - B) * (1 - (v - lo) / (hi - lo))
    out = [f'<svg viewBox="0 0 {W} {H}" class="chart" role="img">']
    if bands:
        up = " ".join(f"{x(d):.1f},{y(h):.1f}" for d, _, h in bands)
        dn = " ".join(f"{x(d):.1f},{y(lw):.1f}" for d, lw, _ in reversed(bands))
        out.append(f'<polygon class="bandfill" points="{up} {dn}"/>')
    for k in range(5):
        v = lo + (hi - lo) * k / 4
        out.append(f'<line class="grid" x1="{L}" x2="{W - R}" y1="{y(v):.1f}" y2="{y(v):.1f}"/>'
                   f'<text class="axis" x="{L - 6}" y="{y(v) + 4:.1f}" text-anchor="end">{v:.{0 if hi - lo > 8 else 1}f}{unit}</text>')
    if zero or bars:
        out.append(f'<line class="zero" x1="{L}" x2="{W - R}" y1="{y(0):.1f}" y2="{y(0):.1f}"/>')
    for k in range(4):
        d = date.fromordinal(t0 + round(span * k / 3)).isoformat()
        anchor = "start" if k == 0 else "end" if k == 3 else "middle"
        out.append(f'<text class="axis" x="{x(d):.1f}" y="{H - 6}" text-anchor="{anchor}">{d[:7] if span > 120 else d[5:]}</text>')
    if bars:
        bw = max(1.0, (W - L - R) / max(1, span) * 0.8)
        for d, v in bars:
            y0, y1 = sorted((y(0), y(v)))
            out.append(f'<rect class="{"barp" if v >= 0 else "barn"}" x="{x(d) - bw / 2:.1f}" y="{y0:.1f}" width="{bw:.1f}" height="{max(0.5, y1 - y0):.1f}"/>')
    for cls, pts in series:
        segs, cur = [], []
        for d, v in pts:
            if v is None:
                if cur:
                    segs.append(cur)
                cur = []
            else:
                cur.append(f"{x(d):.1f},{y(v):.1f}")
        if cur:
            segs.append(cur)
        for sg in segs:
            if len(sg) == 1:
                cx, cy = sg[0].split(",")
                out.append(f'<circle class="{cls}" cx="{cx}" cy="{cy}" r="3"/>')
            else:
                out.append(f'<polyline class="{cls}" points="{" ".join(sg)}"/>')
    out.append("</svg>")
    return "".join(out)


def legend(items) -> str:
    return '<div class="legend">' + "".join(f'<span><i style="background:var(--{c})"></i>{e(t)}</span>' for c, t in items) + "</div>"


# ---------------------------------------------------------------- tables

def score_table(card: dict, note: str) -> str:
    if not card:
        return f'<p class="muted">{e(note)}</p>'
    parts = []
    for tgt in ("y_sector", "y_relative"):
        rows = card.get(tgt, {})
        if not rows:
            continue
        body = ""
        for model in ("ssm", "ekf", "ridge", "momentum", "drift", "zero"):
            m = rows.get(model)
            if not m or not m.get("n"):
                continue
            body += (f'<tr class="{"me" if model in ("ssm", "ekf") else ""}"><td>{e(MODEL_LABEL[model])}</td><td class="n">{m["n"]}</td>'
                     f'<td class="n">{num(m.get("mae"))}</td><td class="n">{num(m.get("skill"), 3)}</td>'
                     f'<td class="n">{pct(100 * m["hit"], 0, False) if m.get("hit") is not None else "—"}</td>'
                     f'<td class="n">{num(m.get("ic"), 3)}</td><td class="n">{num(m.get("brier"), 3)}</td>'
                     f'<td class="n">{pct(100 * m["cover80"], 0, False) if m.get("cover80") is not None else "—"}</td></tr>')
        if not body:
            continue
        parts.append(f"""<h3>{e(TARGET_LABEL[tgt])}, next {config.HORIZON} sessions</h3><div class="scroll"><table><thead><tr><th>Model</th>
<th class="n">Scored</th><th class="n">Mean abs error (pp)</th><th class="n">Skill vs no change</th><th class="n">Direction right</th>
<th class="n">Rank corr.</th><th class="n">Brier</th><th class="n">In 80% range</th></tr></thead><tbody>{body}</tbody></table></div>""")
    return "".join(parts) or f'<p class="muted">{e(note)}</p>'


CRITERION_LABEL = {"resonance": "Resonance index (measured input spectrum)", "flat": "Resonance, flat spectrum (control)",
                   "h2": "H2 energy (white input)", "hinf": "H-infinity gain (worst-case input)", "l1": "Kernel L1 bound",
                   "random": "Random"}


def prune_table(compare: dict, full: float | None, note: str) -> str:
    """Validation loss after pruning without retraining, per criterion and level (lower is better)."""
    if not compare:
        return ""
    levels = sorted({k for v in compare.values() for k in v}, key=float)
    best = {k: min(v[k] for v in compare.values() if k in v) for k in levels}
    rows = "".join(
        f'<tr class="{"me" if c == "resonance" else ""}"><td>{e(CRITERION_LABEL.get(c, c))}</td>'
        + "".join(f'<td class="n">{"<b>" if lv.get(k) == best[k] else ""}{num(lv.get(k), 3)}{"</b>" if lv.get(k) == best[k] else ""}</td>' for k in levels)
        + "</tr>" for c, lv in compare.items())
    head = "".join(f'<th class="n">Keep {100 * float(k):g}%</th>' for k in levels)
    return (f'<h3>Pruning criteria compared</h3><p class="muted">{e(note)} Validation loss (Gaussian NLL, lower is better) after pruning '
            f'without retraining; best per level in bold. Unpruned: {num(full, 3)}.</p>'
            f'<div class="scroll"><table><thead><tr><th>Criterion</th>{head}</tr></thead><tbody>{rows}</tbody></table></div>')


# ---------------------------------------------------------------- pages

def build(out: Path, s: dict) -> None:
    """`s` is the run summary written by the pipeline (see cli.py)."""
    out.mkdir(parents=True, exist_ok=True)
    built = datetime.now(UTC).strftime("%Y-%m-%d %H:%M UTC")
    (out / "index.html").write_text(index_page(s, built))
    (out / "method.html").write_text(method_page(s, built))
    (out / ".nojekyll").write_text("")


def index_page(s: dict, built: str) -> str:
    latest = s.get("latest") or {}
    led = pd.DataFrame(s.get("ledger_rows") or [])
    warn = "".join(f'<div class="banner warn">{e(w)}</div>' for w in s.get("warnings", []))
    tiles = ""
    if latest:
        p = {(r["model"], r["target"]): r for r in latest["rows"]}
        sec, rel = p.get(("ssm", "y_sector")), p.get(("ssm", "y_relative"))
        tiles = f"""<div class="tiles">
<div class="tile"><div class="k">Prediction made after the close of</div><div class="v">{e(latest["as_of"])}</div><div class="s">{config.TARGET} closed at {num(latest.get("close"))}</div></div>
<div class="tile"><div class="k">{config.TARGET}, next {config.HORIZON} sessions</div><div class="v">{signed(logret_to_pct(sec["pred"]))}</div>
<div class="s">80% range {pct(logret_to_pct(sec["lo80"]), 1)} to {pct(logret_to_pct(sec["hi80"]), 1)}</div></div>
<div class="tile"><div class="k">Chance {config.TARGET} is up</div><div class="v">{100 * sec["p_up"]:.0f}%</div><div class="s">from the model's predicted spread</div></div>
<div class="tile"><div class="k">{config.TARGET} vs {config.BENCHMARK}</div><div class="v">{signed(rel["pred"])}</div><div class="s">chance tech beats the market {100 * rel["p_up"]:.0f}%</div></div>
</div>"""
    else:
        tiles = '<div class="banner warn">No prediction yet: the first one is made after the first full run with price data.</div>'

    n_pred = led["as_of"].nunique() if len(led) else 0
    n_scored = led.dropna(subset=["realized"])["as_of"].nunique() if len(led) and "realized" in led else 0
    fwd_note = (f"Collecting: {n_pred} predictions made, {n_scored} scored so far. Each is scored {config.HORIZON} sessions after it was made."
                if n_pred else "No predictions yet.")
    fwd = score_table(s.get("forward_scorecard") or {}, fwd_note)

    # predicted vs realised (forward test)
    chart_fwd = ""
    if len(led):
        ss = led[(led["model"] == "ssm") & (led["target"] == "y_sector")].sort_values("as_of")
        if len(ss) >= 2:
            pred = [(d, logret_to_pct(v)) for d, v in zip(ss["as_of"], ss["pred"])]
            real = [(d, logret_to_pct(v) if pd.notna(v) else None) for d, v in zip(ss["as_of"], ss["realized"])]
            bands = [(d, logret_to_pct(a), logret_to_pct(b)) for d, a, b in zip(ss["as_of"], ss["lo80"], ss["hi80"])]
            chart_fwd = ("<h2>Predicted and actual</h2><p class=\"sub\">Each point is a prediction of the next "
                         f"{config.HORIZON} sessions' {config.TARGET} return, plotted on the day it was made; the shaded band is its 80% range.</p>"
                         f'<div class="card">{chart([("sa", pred), ("sb", real)], bands=bands, unit="%", zero=True)}'
                         f'{legend([("a", "Predicted"), ("b", "Actual")])}</div>')

    px = s.get("price_tail") or []
    news = s.get("news_daily") or []
    price_chart = chart([("sa", [(d, v) for d, v in px])]) if px else '<p class="muted">No prices yet.</p>'
    news_chart = ('<p class="muted">No news scored yet.</p>' if not any(n["avail"] for n in news) else
                  chart([("sa", [(n["date"], n["tone_ewm"] if n["avail"] else None) for n in news])],
                        bars=[(n["date"], n["tone"]) for n in news if n["avail"] and n["count"]], zero=True)
                  + legend([("a", "Smoothed tone (5-day half-life)"), ("lo", "Daily tone")]))
    heads = "".join(f'<tr><td class="n">{h["sentiment"]:+.2f}</td><td class="n">{h["relevance"]:.2f}</td><td>{e(h["ticker"])}</td>'
                    f'<td>{e(h["headline"])}</td></tr>' for h in s.get("top_headlines", []))
    heads_html = (f'<div class="scroll"><table><thead><tr><th class="n">Tone</th><th class="n">Relevance</th><th>Co.</th><th>Headline</th></tr></thead>'
                  f'<tbody>{heads}</tbody></table></div>') if heads else '<p class="muted">No scored headlines for the latest session.</p>'

    m = s.get("model") or {}
    seeds = m.get("seeds") or []
    modes = f'{sum(x["modes_active"] for x in seeds)} of {sum(x["modes_total"] for x in seeds)}' if seeds else "—"
    nll = (f'validation loss {sum(x["val_nll"] for x in seeds) / len(seeds):.3f} pruned vs '
           f'{sum(x["val_nll_full"] for x in seeds) / len(seeds):.3f} unpruned') if seeds else ""
    model_card = f"""<div class="tiles">
<div class="tile"><div class="k">Trained on sessions through</div><div class="v">{e(m.get("train_end", "—"))}</div><div class="s">validated {e(m.get("val_start", "—"))} to {e(m.get("val_end", "—"))}</div></div>
<div class="tile"><div class="k">State modes kept after pruning</div><div class="v">{modes}</div><div class="s">{len(seeds)} ensemble members; {nll}</div></div>
<div class="tile"><div class="k">Inputs in the state vector</div><div class="v">{m.get("n_features", "—")}</div><div class="s">prices, related sectors, breadth, news</div></div>
<div class="tile"><div class="k">News scored this month</div><div class="v">${s.get("claude_month_usd", 0):.2f}</div><div class="s">Claude Haiku, budget ${config.CLAUDE_MONTHLY_BUDGET_USD:.0f}/month</div></div>
</div>"""

    bt = s.get("backtest") or {}
    bt_html = (score_table(bt.get("scorecard_nonoverlap") or {}, "Backtest not run yet.") +
               prune_table(bt.get("prune_compare") or {}, bt.get("val_nll_full"), "Averaged over every retrain in the backtest.") +
               (f'<p class="muted">Walk-forward from {e(bt.get("first") or "—")} to {e(bt.get("last") or "—")}, retrained every '
                f'{bt.get("every")} sessions on data known at the time; scored on every {config.HORIZON}th session so outcomes don\'t overlap.'
                f'{"" if bt.get("complete", True) else " Stopped early to stay within the run time limit; it continues on the next run."}</p>'
                if bt else ""))

    body = f"""<h1>Tech sector state-space model</h1>
<p class="sub">A pruned S4D model reads a daily state of prices, related sectors and news, and predicts the
{config.TARGET} ETF's return over the next {config.HORIZON} trading sessions. Every prediction is logged before its outcome is known
and scored against simple baselines.</p>{warn}{tiles}
<h2>Forward test</h2><p class="sub">Only predictions made live, after each close, count here.</p>{fwd}{chart_fwd}
<div class="grid2"><div><h2>{config.TARGET}, last year</h2><div class="card">{price_chart}</div></div>
<div><h2>Tech news tone</h2><div class="card">{news_chart}</div></div></div>
<h2>Headlines behind the latest prediction</h2><p class="sub">The most relevant stories assigned to the latest session, as scored by Claude Haiku.</p>{heads_html}
<h2>Current model</h2>{model_card}{prune_table(m.get("prune_compare") or {}, m.get("val_nll_full"), "Current model, averaged over its ensemble members.")}
<h2>Backtest</h2><p class="sub">How the same method did on past data. This is not a forward test: it shows the method can be fitted, not that it works going forward.
News only exists for the last year, so most of the backtest runs without it.</p>{bt_html}
<p class="muted"><a href="data/predictions.csv">Prediction ledger (CSV)</a> · <a href="data/backtest.csv">Backtest predictions (CSV)</a> ·
<a href="data/summary.json">Run summary (JSON)</a></p>"""
    return page("Overview", "home", body, built)


def method_page(s: dict, built: str) -> str:
    feats = s.get("features") or []
    groups = {"Daily returns": [f for f in feats if f.startswith("r_")],
              "Sector state": [f for f in feats if not f.startswith(("r_", "news_"))],
              "News": [f for f in feats if f.startswith("news_")]}
    flist = "".join(f"<p><b>{e(g)}</b> ({len(v)}): <span class=\"muted\">{e(', '.join(v))}</span></p>" for g, v in groups.items() if v)
    rel = ", ".join(f"{t} ({n})" for t, n in config.RELATED.items())
    mc = config.DEFAULT.model
    body = f"""<h1>Method</h1>
<div class="card"><h3>What is predicted</h3><p>After each US market close, the model predicts the log return of {config.TARGET}
(the Technology Select Sector SPDR ETF) over the next {config.HORIZON} sessions, and the same return minus {config.BENCHMARK}'s.
Each prediction is a mean and a spread, which give the 80% range and the chance of a rise.</p>
<h3>The state vector</h3><p>One row per session, built only from data available at that session's close:</p>
<p>{config.TARGET} and 12 large tech companies ({", ".join(config.CONSTITUENTS)}); related series {e(rel)}.</p>{flist}
<p>News: Finnhub headlines for the 12 companies. A free screen drops duplicates and market roundups, and caps each session at
{config.MAX_HEADLINES_PER_DAY} stories. Claude Haiku scores each story's relevance to the tech sector (0 to 1) and its sentiment for the
sector's stocks (−1 to 1), {config.HEADLINES_PER_CALL} stories per request. A story counts for the first session whose 16:00 New York close
comes after it was published. Before news collection began the news inputs are 0 and a flag says they are missing.</p>
<h3>The model</h3><p>A linear encoder, then {mc.n_layers} S4D layers ({mc.d_model} channels, {mc.n_modes} complex diagonal modes each,
S4D-Lin initialisation, step sizes {mc.dt_min} to {mc.dt_max} so memories range from days to years), each followed by a GELU, a mixing
layer, a residual connection and layer norm. A linear head gives a mean and a log-variance for each target, trained with Gaussian
negative log-likelihood on random {mc.window}-session windows (the first {mc.burn_in} positions only warm up the state).</p>
<h3>Pruning</h3><p>Modes are ranked by the <b>resonance index</b> (Philco, "From Periodicity to Resonance", 2026): the
expected output power of a mode when driven by the input its layer actually receives,
I = |g|² / L · Σ<sub>j</sub> S(ν<sub>j</sub>) / (1 + r² − 2r·cos(ν<sub>j</sub> − θ)), where g is the mode's gain, r·e<sup>iθ</sup> its discrete
pole and S the channel's input power spectrum, a periodogram over {mc.spectrum_seg}-session segments of the training data.
The network is pruned to the highest-scoring {", ".join(f"{100 * k:g}%" for k in mc.prune_keep)} of modes in each layer and fine-tuned at each level;
the sparsest level whose validation loss is within {mc.prune_tolerance} of the best level is used. As in the paper, every retrain also
records how each criterion does when pruning without retraining: the resonance index, its flat-spectrum control, H2 energy, H-infinity
gain, the kernel L1 bound and random.</p>
<h3>EKF + SSM</h3><p>A second model treats the daily vector as noisy measurements of a {mc.ekf_d_state}-dimensional latent
sector state instead of as an input. The state evolves as x<sub>t</sub> = A x<sub>t−1</sub> + U tanh(V x<sub>t−1</sub> + c) + w, where A holds
damped 2×2 rotations (the real form of S4D's complex diagonal modes); measurements are z<sub>t</sub> = H x<sub>t</sub> + b + v with diagonal
noise. An extended Kalman filter predicts and updates the state every session, linearising the dynamics with their Jacobian; news
features are left out of the update before news collection began. The 5-session targets are read out from the filtered state, with
variance from the state covariance. Dynamics, noise levels, measurement map and readout are fitted together by backpropagating through
the filter (one-step measurement likelihood plus target likelihood), with the same training, validation and retraining schedule.</p>
<h3>Training and validation</h3><p>The model is retrained every {mc.retrain_every} sessions on every session whose outcome was already
known. The last {mc.val_sessions} of those are held out for early stopping and the pruning check, with a {config.HORIZON}-session gap.
{len(mc.seeds)} seeds are averaged; the spread combines their predicted variance and their disagreement.</p>
<h3>Baselines</h3><p>Fitted on the same training rows: no change (predict 0), average drift, an OLS on 20-day momentum, and a ridge
regression on the same inputs (last session only, no memory). If the SSM can't beat these, its memory and news inputs aren't adding anything.</p>
<h3>Scores</h3><p>Mean absolute error in percentage points of log return; skill = 1 − MSE / MSE of predicting no change (above 0 beats it);
direction right = share of predictions with the correct sign; rank correlation (Spearman) between predicted and actual; Brier score of
the chance of a rise (lower is better, 0.25 is a coin flip); share of outcomes inside the 80% range (should be near 80%).</p></div>"""
    return page("Method", "method", body, built)


def write_data(out: Path, ledger: pd.DataFrame, backtest: pd.DataFrame | None, summary: dict) -> None:
    d = out / "data"
    d.mkdir(parents=True, exist_ok=True)
    ledger.to_csv(d / "predictions.csv", index=False, float_format="%.6g")
    (backtest if backtest is not None else pd.DataFrame()).to_csv(d / "backtest.csv", index=False, float_format="%.6g")
    slim = {k: v for k, v in summary.items() if k not in ("ledger_rows", "price_tail", "news_daily")}
    (d / "summary.json").write_text(json.dumps(slim, indent=1, default=str))
