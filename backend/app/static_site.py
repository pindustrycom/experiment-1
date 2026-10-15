"""Static, read-only dashboard for GitHub Pages.

Reads everything through the application's own API (in-process, no network), so the published
numbers are exactly what the full app would show, and renders plain HTML pages with inline SVG
charts. No JavaScript, no cookies, no external requests.

    python -m app.static_site --out _site            # refuses if simulated data is present
    python -m app.static_site --out _site --allow-demo   # local preview only, pages are labelled
"""
from __future__ import annotations

import argparse
import html
import json
import math
import secrets
import shutil
import sys
from datetime import UTC, date, datetime
from pathlib import Path

from fastapi.testclient import TestClient
from sqlalchemy import select

from app.core.db import session_scope
from app.core.security import hash_password
from app.models import User

BUILDER_EMAIL = "static-site-builder@localhost"
HORIZON_LABEL = {30: "30 sessions (~6 weeks)", 90: "90 sessions (~4 months)", 180: "180 sessions (~9 months)",
                 252: "252 sessions (~1 year)", 504: "504 sessions (~2 years)"}
DIRECTION = {"a_advantaged": "{a} favoured", "b_advantaged": "{b} favoured", "neutral": "Neutral", "uncertain": "Uncertain",
             "not_applicable": "n/a"}
REPO_URL = "https://github.com/pindustrycom/experiment-1"

e = html.escape


# ---------------------------------------------------------------- data access

class Api:
    def __init__(self) -> None:
        from app.main import create_app
        pw = secrets.token_urlsafe(32)
        with session_scope() as db:
            u = db.scalar(select(User).where(User.email == BUILDER_EMAIL))
            if u is None:
                db.add(User(email=BUILDER_EMAIL, password_hash=hash_password(pw), role="viewer"))
            else:
                u.password_hash, u.role = hash_password(pw), "viewer"
        self.c = TestClient(create_app(), base_url="https://localhost")
        r = self.c.post("/api/v1/auth/login", json={"email": BUILDER_EMAIL, "password": pw}, headers={"X-Requested-With": "splittability"})
        r.raise_for_status()

    def get(self, path: str, **params):
        r = self.c.get("/api/v1" + path, params={k: v for k, v in params.items() if v is not None})
        r.raise_for_status()
        return r.json()

    def text(self, path: str, **params) -> str:
        r = self.c.get("/api/v1" + path, params=params)
        r.raise_for_status()
        return r.text

    def close(self) -> None:
        # The builder account only exists for the build and has a random, discarded password.
        with session_scope() as db:
            u = db.scalar(select(User).where(User.email == BUILDER_EMAIL))
            if u is not None:
                u.password_hash = hash_password(secrets.token_urlsafe(32))


# ---------------------------------------------------------------- formatting

def num(v, digits=1, dash="—"):
    if v is None or (isinstance(v, float) and not math.isfinite(v)):
        return dash
    return f"{v:.{digits}f}"


def pct(v, digits=0):
    return "—" if v is None else f"{100 * v:.{digits}f}%"


def day(s: str | None) -> str:
    return "—" if not s else s[:10]


def when(s: str | None) -> str:
    if not s:
        return "—"
    try:
        t = datetime.fromisoformat(s.replace("Z", "+00:00"))
        return t.astimezone(UTC).strftime("%Y-%m-%d %H:%M UTC")
    except ValueError:
        return e(s)


def risk_class(v) -> str:
    if v is None:
        return "na"
    return "hi" if v >= 70 else "md" if v >= 40 else "lo"


def score_cell(s: dict | None) -> str:
    if not s or s.get("value") is None:
        status = (s or {}).get("status") or "unavailable"
        return f'<span class="score na" title="{e(status)}">n/a</span>'
    v = s["value"]
    prov = ' <span class="tag">provisional</span>' if s.get("status") == "provisional" else ""
    return f'<span class="score {risk_class(v)}">{v:.1f}</span>{prov}'


def change(v) -> str:
    if v is None:
        return '<span class="muted">—</span>'
    cls = "up" if v > 0.05 else "down" if v < -0.05 else "muted"
    return f'<span class="{cls}">{v:+.1f}</span>'


def direction(d: str | None, a: str, b: str) -> str:
    return e(DIRECTION.get(d or "", d or "—").format(a=a, b=b))


# ---------------------------------------------------------------- charts (inline SVG)

def line_chart(series: list[tuple[str, list[tuple[str, float | None]]]], height=220, y_label="", y_range=None,
               bands: list[tuple[float, float, str]] | None = None) -> str:
    """series: [(css class, [(iso date, value)])]. Dates are spaced by position of the union of dates."""
    dates = sorted({d for _, pts in series for d, v in pts if v is not None})
    if len(dates) < 2:
        return '<p class="muted">Not enough data to draw this chart yet.</p>'
    vals = [v for _, pts in series for _, v in pts if v is not None]
    lo, hi = y_range if y_range else (min(vals), max(vals))
    if hi - lo < 1e-9:
        hi, lo = hi + 1, lo - 1
    pad = 0 if y_range else (hi - lo) * 0.06
    lo, hi = lo - pad, hi + pad
    W, H, L, R, T, B = 760, height, 44, 10, 10, 26
    t0, t1 = datetime.fromisoformat(dates[0]).toordinal(), datetime.fromisoformat(dates[-1]).toordinal()
    span = max(1, t1 - t0)

    def x(d):
        return L + (W - L - R) * (datetime.fromisoformat(d).toordinal() - t0) / span

    def y(v):
        return T + (H - T - B) * (1 - (v - lo) / (hi - lo))
    out = [f'<svg viewBox="0 0 {W} {H}" class="chart" role="img" aria-label="{e(y_label)}">']
    for b0, b1, cls in bands or []:
        out.append(f'<rect class="{cls}" x="{L}" width="{W - L - R}" y="{y(min(b1, hi)):.1f}" height="{max(0, y(max(b0, lo)) - y(min(b1, hi))):.1f}"/>')
    for k in range(5):
        v = lo + (hi - lo) * k / 4
        out.append(f'<line class="grid" x1="{L}" x2="{W - R}" y1="{y(v):.1f}" y2="{y(v):.1f}"/>'
                   f'<text class="axis" x="{L - 6}" y="{y(v) + 4:.1f}" text-anchor="end">{v:.0f}</text>')
    for k in range(4):
        d = date.fromordinal(t0 + round(span * k / 3)).isoformat()
        anchor = "start" if k == 0 else "end" if k == 3 else "middle"
        out.append(f'<text class="axis" x="{x(d):.1f}" y="{H - 6}" text-anchor="{anchor}">{d[:7]}</text>')
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
        for sgm in segs:
            if len(sgm) == 1:
                cx, cy = sgm[0].split(",")
                out.append(f'<circle class="{cls}" cx="{cx}" cy="{cy}" r="2.5"/>')
            else:
                out.append(f'<polyline class="{cls}" points="{" ".join(sgm)}"/>')
    out.append("</svg>")
    return "".join(out)


def bar(v: float | None, cls="") -> str:
    if v is None:
        return '<span class="muted">unavailable</span>'
    return f'<span class="bar {cls}"><span style="width:{max(0, min(1, v)) * 100:.0f}%"></span></span> {v:.2f}'


# ---------------------------------------------------------------- page shell

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
main{padding:22px 0 40px}h1{font-size:24px;margin:0 0 4px}h2{font-size:18px;margin:28px 0 10px}h3{font-size:15px;margin:18px 0 6px}
.sub{color:var(--muted);margin:0 0 18px}.muted{color:var(--muted)}
.banner{border-radius:8px;padding:10px 14px;margin:0 0 18px;border:1px solid var(--line)}
.banner.warn{background:var(--md-bg)}.banner.danger{background:var(--hi-bg);font-weight:600}
.tiles{display:grid;grid-template-columns:repeat(auto-fit,minmax(170px,1fr));gap:12px;margin-bottom:18px}
.tile,.card{background:var(--panel);border:1px solid var(--line);border-radius:10px;padding:12px 14px}
.tile .k{color:var(--muted);font-size:13px}.tile .v{font-size:22px;font-weight:650}
.card{margin-bottom:14px}.scroll{overflow-x:auto}
table{width:100%;border-collapse:collapse;background:var(--panel)}th,td{padding:8px 10px;border-bottom:1px solid var(--line);text-align:left;vertical-align:top}
th{font-size:12px;color:var(--muted);font-weight:600;text-transform:uppercase;letter-spacing:.03em;white-space:nowrap}
td.n,th.n{text-align:right;font-variant-numeric:tabular-nums;white-space:nowrap}
.score{display:inline-block;min-width:46px;text-align:center;border-radius:6px;padding:1px 6px;font-weight:650;font-variant-numeric:tabular-nums}
.score.lo{background:var(--lo-bg);color:var(--lo)}.score.md{background:var(--md-bg);color:var(--md)}.score.hi{background:var(--hi-bg);color:var(--hi)}
.score.na{color:var(--muted);border:1px dashed var(--line)}
.tag{font-size:11px;border:1px solid var(--line);border-radius:4px;padding:0 4px;color:var(--muted)}
.up{color:var(--hi)}.down{color:var(--lo)}
.bar{display:inline-block;width:90px;height:8px;background:var(--band);border-radius:4px;vertical-align:middle;overflow:hidden}
.bar span{display:block;height:100%;background:var(--accent)}
.chart{width:100%;height:auto;display:block}.chart .grid{stroke:var(--line);stroke-width:1}.chart .axis{fill:var(--muted);font-size:11px}
.chart polyline{fill:none;stroke-width:1.8}.chart .sa{stroke:var(--a)}.chart .sb{stroke:var(--b)}.chart circle.sa{fill:var(--a)}.chart circle.sb{fill:var(--b)}
.chart .band-hi{fill:var(--hi-bg)}.chart .band-md{fill:var(--md-bg)}
.legend span{display:inline-block;margin-right:14px;font-size:13px}.legend i{display:inline-block;width:12px;height:3px;vertical-align:middle;margin-right:5px}
.quote{border-left:3px solid var(--line);padding:2px 10px;margin:6px 0;color:var(--muted)}
.grid2{display:grid;grid-template-columns:repeat(auto-fit,minmax(320px,1fr));gap:14px}
footer{border-top:1px solid var(--line);color:var(--muted);font-size:13px;padding:16px 0 30px}
"""


def page(title: str, active: str, body: str, ctx: dict, depth: int = 0) -> str:
    root = "../" * depth
    nav = "".join(f'<a href="{root}{href}" class="{"on" if key == active else ""}">{label}</a>'
                  for key, href, label in [("home", "index.html", "Pairs"), ("predictions", "predictions.html", "Predictions"),
                                           ("performance", "performance.html", "Performance"), ("about", "about.html", "About")])
    demo = ('<div class="banner danger">SIMULATED DATA. This is a local preview built from demo data; nothing on this page is real.</div>'
            if ctx["demo"] else "")
    return f"""<!doctype html><html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1"><meta name="robots" content="noindex">
<title>{e(title)} · Splittability Monitor</title><style>{CSS}</style></head><body>
<header><div class="wrap"><a class="brand" href="{root}index.html">Splittability Monitor</a><nav>{nav}</nav></div></header>
<main class="wrap">{demo}{body}</main>
<footer><div class="wrap">Research dashboard for tracking pair-divergence predictions. Scores are relative risk rankings, not
probabilities, and nothing here is investment advice; no orders are placed anywhere. Built {e(ctx["built"])} from data as of
the last completed US market session. <a href="{REPO_URL}">Source and methodology</a>.</div></footer></body></html>"""


# ---------------------------------------------------------------- pages

def freshness_banner(status: dict) -> str:
    bits = []
    if status.get("last_price_update") is None:
        bits.append("No price data has been loaded yet; the first scheduled run fills it in.")
    elif (status.get("price_age_hours") or 0) > 96:
        bits.append(f"Prices were last updated {when(status['last_price_update'])}, more than four days ago.")
    if not status.get("ai", {}).get("configured"):
        bits.append("News analysis is off (no Claude API key), so SI-v2 news components are unavailable.")
    return "".join(f'<div class="banner warn">{e(b)}</div>' for b in bits)


def index_page(status: dict, pairs: list[dict], ctx: dict) -> str:
    p = status.get("pairs", {})
    tiles = f"""<div class="tiles">
<div class="tile"><div class="k">Pairs monitored</div><div class="v">{p.get("monitored", 0)}</div></div>
<div class="tile"><div class="k">High risk (SI-v2 ≥ 70)</div><div class="v">{p.get("high_risk", 0)}</div></div>
<div class="tile"><div class="k">Average SI-v1 / SI-v2</div><div class="v">{num(p.get("avg_si_v1"))} / {num(p.get("avg_si_v2"))}</div></div>
<div class="tile"><div class="k">Prices updated</div><div class="v" style="font-size:15px">{when(status.get("last_price_update"))}</div></div>
<div class="tile"><div class="k">News updated</div><div class="v" style="font-size:15px">{when(status.get("last_news_update"))}</div></div>
</div>"""
    rows = []
    for x in sorted(pairs, key=lambda q: -((q.get("si_v2") or {}).get("value") or -1)):
        a, b = x["company_a"]["ticker"], x["company_b"]["ticker"]
        rows.append(f"""<tr><td><a href="pairs/{e(x["slug"])}.html"><b>{e(a)} / {e(b)}</b></a><div class="muted">{e(x.get("industry") or "")}</div></td>
<td class="n">{score_cell(x.get("si_v1"))}</td><td class="n">{score_cell(x.get("si_v2"))}</td>
<td class="n">{change((x.get("si_v2") or {}).get("change_7d"))}</td><td>{direction(x.get("direction"), a, b)}</td>
<td class="n">{num(x.get("confidence"), 2)}</td><td class="muted">{day(x.get("last_update"))}</td></tr>""")
    table = f"""<div class="scroll"><table><thead><tr><th>Pair</th><th class="n">SI-v1</th><th class="n">SI-v2</th><th class="n">7-day change</th>
<th>Direction</th><th class="n">Confidence</th><th>Updated</th></tr></thead><tbody>{"".join(rows) or '<tr><td colspan="7" class="muted">No pairs yet.</td></tr>'}</tbody></table></div>"""
    body = f"""<h1>Pairs</h1><p class="sub">Risk that each pair's prices drift apart, on a 0–100 scale (higher is riskier). SI-v1 uses market data only;
SI-v2 adds news, fundamentals and structure. "n/a" means there is not enough data to score honestly, never zero risk.</p>
{freshness_banner(status)}{tiles}{table}"""
    return page("Pairs", "home", body, ctx)


def pair_page(pr: dict, scores: list[dict], detail: dict | None, prices: dict, preds: list[dict], news: list[dict], ctx: dict) -> str:
    a, b = pr["company_a"]["ticker"], pr["company_b"]["ticker"]
    head = f"""<p class="muted"><a href="../index.html">Pairs</a> / {e(a)} / {e(b)}</p>
<h1>{e(pr["company_a"]["name"])} vs {e(pr["company_b"]["name"])}</h1><p class="sub">{e(pr.get("industry") or "")}</p>
<div class="tiles"><div class="tile"><div class="k">SI-v1 (market only)</div><div class="v">{score_cell(pr.get("si_v1"))}</div></div>
<div class="tile"><div class="k">SI-v2 (news-enhanced)</div><div class="v">{score_cell(pr.get("si_v2"))}</div></div>
<div class="tile"><div class="k">Direction</div><div class="v" style="font-size:16px">{direction(pr.get("direction"), a, b)}</div></div>
<div class="tile"><div class="k">Evidence confidence</div><div class="v">{num(pr.get("confidence"), 2)}</div></div></div>
<div class="card">{e(pr.get("explanation") or "No score yet.")}</div>"""

    v1 = [(s["as_of"][:10], s["value"]) for s in scores if s["index_name"] == "si_v1"]
    v2 = [(s["as_of"][:10], s["value"]) for s in scores if s["index_name"] == "si_v2"]
    hist = f"""<h2>Score history</h2><div class="card"><div class="legend"><span><i style="background:var(--a)"></i>SI-v1</span>
<span><i style="background:var(--b)"></i>SI-v2</span></div>{line_chart([("sa", v1), ("sb", v2)], y_label="score", y_range=(0, 100),
                                                                    bands=[(70, 100, "band-hi"), (40, 70, "band-md")])}</div>"""

    if prices.get("available"):
        norm = prices.get("normalized") or []
        pc = f"""<h2>Prices (rebased to 100, total return)</h2><div class="card"><div class="legend"><span><i style="background:var(--a)"></i>{e(a)}</span>
<span><i style="background:var(--b)"></i>{e(b)}</span> <span class="muted">source: {e(prices.get("provider") or "")}, last close {e(prices.get("last_date") or "")}
{"(price return only)" if not prices.get("is_total_return") else ""}</span></div>
{line_chart([("sa", [(r["date"], r["a"]) for r in norm]), ("sb", [(r["date"], r["b"]) for r in norm])], y_label="rebased price")}</div>"""
    else:
        pc = '<h2>Prices</h2><p class="muted">No price data for this pair yet.</p>'

    comps = ""
    if detail and detail.get("components"):
        rows = "".join(f"""<tr><td><b>{e(c["component"])}</b></td><td class="n">{num(c.get("weight"), 2)}</td><td>{bar(c.get("value") if c.get("available") else None)}</td>
<td>{e(c.get("explanation") or "")}</td></tr>""" for c in sorted(detail["components"], key=lambda c: -(c.get("weight") or 0)))
        comps = f"""<h2>What drives SI-v2</h2><div class="scroll"><table><thead><tr><th>Component</th><th class="n">Weight</th><th>Value (0–1)</th><th>Why</th></tr></thead>
<tbody>{rows}</tbody></table></div><p class="muted">N news asymmetry · F fundamentals · M market relationship · P persistence · S structural exposure.</p>"""

    prow = []
    for x in preds[:25]:
        out = x.get("outcome") or {}
        res = ("pending" if x.get("status") == "pending" else
               f'D = {num(out.get("divergence"), 3)} → {"diverged" if out.get("diverged") else "held together"}' if out else e(x.get("status") or ""))
        prow.append(f"""<tr><td>{day(x["prediction_date"])}</td><td class="n">{x["horizon_days"]}</td><td class="n">{num(x.get("si_v1"))}</td><td class="n">{num(x.get("si_v2"))}</td>
<td>{direction(x.get("direction"), a, b)}</td><td>{day(x.get("maturity_date"))}</td><td>{res}</td></tr>""")
    ptab = f"""<h2>Recent predictions</h2><p class="muted">The latest 25; every prediction is on the <a href="../predictions.html">Predictions</a> page and in the CSV.</p><div class="scroll"><table><thead><tr><th>Made</th><th class="n">Horizon</th><th class="n">SI-v1</th><th class="n">SI-v2</th>
<th>Direction</th><th>Matures</th><th>Outcome</th></tr></thead><tbody>{"".join(prow) or '<tr><td colspan="7" class="muted">No predictions yet.</td></tr>'}</tbody></table></div>"""

    items = []
    for n in news[:20]:
        evs = "".join(
            f"""<div><span class="tag">{e(ev.get("event_type") or "")}</span> severity {num(ev.get("severity"), 2)}{" · counted in score" if (ev.get("index_effect") or {}).get("counted") else ""}
<div>{e(ev.get("summary") or "")}</div>{"".join(f'<div class="quote">“{e(q.get("quote") or "")}”{"" if q.get("verified") else " (not found verbatim)"}</div>' for q in ev.get("evidence") or [])}</div>"""
            for ev in n.get("events") or [] if ev.get("is_material"))
        srcs = ", ".join(f'<a href="{e(d["url"])}" rel="noopener nofollow">{e(d.get("publisher") or d.get("provider") or "source")}</a>'
                         for d in (n.get("documents") or [])[:3] if str(d.get("url", "")).startswith(("https://", "http://")))
        items.append(f"""<div class="card"><b>{e(n.get("headline") or "")}</b><div class="muted">{when(n.get("first_published_at"))} · {srcs}</div>{evs}</div>""")
    newsb = "<h2>Material news</h2>" + ("".join(items) or '<p class="muted">No analysed material news for this pair yet.</p>')
    return page(f"{a} / {b}", "home", head + hist + pc + comps + ptab + newsb, ctx, depth=1)


def predictions_page(by_pair: dict[str, tuple[dict, list[dict]]], ctx: dict) -> str:
    latest, resolved = [], []
    for slug, (pr, preds) in sorted(by_pair.items()):
        a, b = pr["company_a"]["ticker"], pr["company_b"]["ticker"]
        seen = set()
        for x in preds:
            if x["horizon_days"] not in seen:
                seen.add(x["horizon_days"])
                latest.append((slug, a, b, x))
            if x.get("outcome"):
                resolved.append((slug, a, b, x))
    lrows = "".join(f"""<tr><td><a href="pairs/{e(s)}.html">{e(a)} / {e(b)}</a></td><td class="n">{x["horizon_days"]}</td><td>{day(x["prediction_date"])}</td>
<td class="n">{num(x.get("si_v1"))}</td><td class="n">{num(x.get("si_v2"))}</td><td>{direction(x.get("direction"), a, b)}</td>
<td class="n">{num((x.get("target_definition") or {}).get("threshold"), 3)}</td><td>{day(x.get("maturity_date"))}</td></tr>""" for s, a, b, x in latest)
    resolved.sort(key=lambda t: t[3]["outcome"].get("evaluated_at") or t[3].get("maturity_date") or "", reverse=True)
    rrows = "".join(f"""<tr><td><a href="pairs/{e(s)}.html">{e(a)} / {e(b)}</a></td><td class="n">{x["horizon_days"]}</td><td>{day(x["prediction_date"])}</td>
<td class="n">{num(x.get("si_v1"))}</td><td class="n">{num(x.get("si_v2"))}</td><td class="n">{num(x["outcome"].get("divergence"), 3)}</td>
<td>{"diverged" if x["outcome"].get("diverged") else "held together"}</td></tr>""" for s, a, b, x in resolved[:200])
    body = f"""<h1>Predictions</h1><p class="sub">Every trading day after the close, one immutable prediction is recorded per pair and horizon.
When a horizon matures, it is scored against what prices actually did: D = |ln(PA(t+H)/PA(t)) − ln(PB(t+H)/PB(t))|, and "diverged" means D reached the threshold.</p>
<p><a href="data/predictions.csv">Download every prediction and outcome (CSV)</a></p>
<h2>Latest prediction per pair and horizon</h2><div class="scroll"><table><thead><tr><th>Pair</th><th class="n">Horizon</th><th>Made</th><th class="n">SI-v1</th>
<th class="n">SI-v2</th><th>Direction</th><th class="n">Threshold</th><th>Matures</th></tr></thead><tbody>{lrows or '<tr><td colspan="8" class="muted">No predictions recorded yet.</td></tr>'}</tbody></table></div>
<h2>Recently resolved</h2><div class="scroll"><table><thead><tr><th>Pair</th><th class="n">Horizon</th><th>Made</th><th class="n">SI-v1</th><th class="n">SI-v2</th>
<th class="n">D</th><th>Result</th></tr></thead><tbody>{rrows or '<tr><td colspan="7" class="muted">No prediction has matured yet. The shortest horizon is 30 trading sessions.</td></tr>'}</tbody></table></div>"""
    return page("Predictions", "predictions", body, ctx)


LIVE_MODELS = {"si_v1": "SI-v1", "si_v2": "SI-v2", "correlation_only": "Correlation only (baseline)", "previous_return_gap": "Previous return gap (baseline)"}
BACKTEST_MODELS = {"si_v1": "SI-v1", "si_v2_partial_default": "SI-v2 without news (default weights)",
                   "si_v2_partial_train_optimised": "SI-v2 without news (weights fitted on 2015–2019)",
                   "correlation_only": "Correlation only (baseline)", "previous_return_gap": "Previous return gap (baseline)"}


def metric_rows(indices: dict, names: dict = LIVE_MODELS) -> str:
    out = []
    for k, label in names.items():
        m = indices.get(k) or {}
        if m.get("status") not in (None, "ok"):
            out.append(f'<tr><td>{label}</td><td colspan="5" class="muted">{e(str(m.get("status")))}{": " + e(m["reason"]) if m.get("reason") else ""}</td></tr>')
            continue
        ci = m.get("roc_auc_ci") or {}
        cis = f' <span class="muted">({num(ci.get("low"), 2)}–{num(ci.get("high"), 2)})</span>' if ci.get("low") is not None else ""
        out.append(f"""<tr><td>{label}</td><td class="n">{m.get("n", "—")}</td><td class="n">{pct(m.get("base_rate"))}</td>
<td class="n">{num(m.get("roc_auc"), 3)}{cis}</td><td class="n">{num(m.get("pr_auc"), 3)}</td><td class="n">{num(m.get("brier"), 3)}</td></tr>""")
    return "".join(out)


METRIC_HEAD = '<thead><tr><th>Model</th><th class="n">n</th><th class="n">Diverged</th><th class="n">ROC-AUC (95% CI)</th><th class="n">PR-AUC</th><th class="n">Brier</th></tr></thead>'


def performance_page(perf: dict, ctx: dict) -> str:
    live = perf.get("live", {})
    hs = []
    for h, v in sorted(live.get("horizons", {}).items(), key=lambda t: int(t[0])):
        thr = (v.get("target") or {}).get("threshold")
        hs.append(f"""<h3>{e(HORIZON_LABEL.get(int(h), h + " sessions"))}, threshold D ≥ {num(thr, 3)}</h3>
<p class="muted">{v.get("resolved", 0)} resolved, {v.get("pending", 0)} still pending.</p>
<div class="scroll"><table>{METRIC_HEAD}<tbody>{metric_rows(v.get("indices") or {})}</tbody></table></div>""")
    bts = perf.get("backtests") or []
    wf = next((x for x in bts if x.get("kind") == "walk_forward"), None)
    rep = next((x for x in bts if x.get("kind") == "reported_historical"), None)
    wfb = ""
    if wf:
        t = (wf.get("metrics") or {}).get("test") or {}
        lim = "".join(f"<li>{e(x)}</li>" for x in wf.get("limitations") or [])
        wfb = f"""<h2>Walk-forward backtest on these pairs</h2><p class="muted">{e(wf.get("label") or "")}. Trained 2015–2019, embargo 2020–2021, tested 2022–2024,
using only data that was available at each date. News was not archived historically, so SI-v2 here excludes news.</p>
<div class="scroll"><table>{METRIC_HEAD}<tbody>{metric_rows(t, BACKTEST_MODELS)}</tbody></table></div><ul class="muted">{lim}</ul>"""
    repb = ""
    if rep:
        m = rep.get("metrics") or {}
        repb = f"""<h2>Original experiment (as reported, not re-run here)</h2><div class="card">100 pairs, 300 out-of-sample observations (2022–2024), two-year divergence ≥ 0.20.
SI-v1 ROC-AUC <b>{num((m.get("si_v1") or {}).get("roc_auc"), 4)}</b>, correlation only {num((m.get("correlation_only") or {}).get("roc_auc"), 4)},
previous return gap {num((m.get("previous_return_gap") or {}).get("roc_auc"), 4)}. Top-quartile divergence rate {pct((m.get("si_v1") or {}).get("top_quartile_divergence_rate"), 1)}
vs bottom quartile {pct((m.get("si_v1") or {}).get("bottom_quartile_divergence_rate"), 1)}.</div>"""
    body = f"""<h1>Performance</h1><p class="sub">How well the scores ranked pairs that later diverged. ROC-AUC of 0.5 is a coin flip; higher is better.
Live results, the backtest and the original experiment are kept separate and are never pooled.</p>
<h2>Live forward test</h2>{"".join(hs) or '<p class="muted">No live predictions have matured yet.</p>'}{wfb}{repb}"""
    return page("Performance", "performance", body, ctx)


def about_page(status: dict, ctx: dict) -> str:
    prov = status.get("providers") or {}
    names = {"tiingo": "Tiingo daily prices", "stooq": "Stooq daily prices (no key)", "sec": "SEC EDGAR filings and fundamentals",
             "finnhub": "Finnhub company news", "alphavantage": "Alpha Vantage news", "anthropic": "Claude news analysis"}
    rows = "".join(f'<tr><td>{e(label)}</td><td>{"on" if prov.get(k) else "<span class=muted>off</span>"}</td></tr>' for k, label in names.items())
    body = f"""<h1>About</h1><p class="sub">A research dashboard that tracks whether pairs of similar companies are likely to drift apart,
and keeps an honest, timestamped record of every prediction so they can be checked later.</p>
<div class="card"><b>Nothing here involves money.</b> The site places no orders and connects to no brokerage. Scores are relative risk
rankings, not probabilities, and not advice.</div>
<h2>How it updates</h2><p>A scheduled GitHub Actions job runs after each US market close. It fetches prices, filings and news, recomputes
the scores, records that day's predictions, checks matured ones, and republishes this site. Past predictions are never edited.</p>
<h2>Data sources</h2><div class="scroll"><table><thead><tr><th>Source</th><th>Status</th></tr></thead><tbody>{rows}</tbody></table></div>
<h2>The two indices</h2><p><b>SI-v1</b> = (25·C + 15·D + 25·R) × 100 / 65, the original market-only index: C is weak one-year correlation,
D is the drop from five-year to one-year correlation, R is the one-year return gap. <b>SI-v2</b> = 35·N + 25·F + 20·M + 10·P + 10·S adds
company-specific news (N, P), fundamentals (F), market relationship (M) and structural differences (S). Missing inputs are shown as unavailable,
never as zero. Full details: <a href="{REPO_URL}/blob/main/docs/METHODOLOGY.md">methodology</a>.</p>"""
    return page("About", "about", body, ctx)


# ---------------------------------------------------------------- build

def build(out: Path, allow_demo: bool) -> dict:
    api = Api()
    try:
        status = api.get("/system/status")
        demo = status.get("mode") == "demo" or bool(status.get("demo_data_present"))
        if demo and not allow_demo:
            raise SystemExit("refusing to publish: the database contains simulated (demo) data")
        ctx = {"demo": demo, "built": datetime.now(UTC).strftime("%Y-%m-%d %H:%M UTC")}
        pairs = api.get("/pairs", page_size=100)["items"]
        if out.exists():
            shutil.rmtree(out)
        (out / "pairs").mkdir(parents=True)
        (out / "data").mkdir()
        by_pair = {}
        for pr in pairs:
            slug = pr["slug"]
            scores = api.get(f"/pairs/{slug}/scores", days=730)
            v2 = [s for s in scores if s["index_name"] == "si_v2"]
            detail = api.get(f"/pairs/{slug}/scores/{v2[-1]['id']}") if v2 else None
            prices = api.get(f"/pairs/{slug}/prices")
            preds = api.get(f"/pairs/{slug}/predictions", page_size=500)["items"]
            news = api.get(f"/pairs/{slug}/news", page_size=50, material_only=True)["items"]
            by_pair[slug] = (pr, preds)
            (out / "pairs" / f"{slug}.html").write_text(pair_page(pr, scores, detail, prices, preds, news, ctx))
        perf = api.get("/performance")
        (out / "index.html").write_text(index_page(status, pairs, ctx))
        (out / "predictions.html").write_text(predictions_page(by_pair, ctx))
        (out / "performance.html").write_text(performance_page(perf, ctx))
        (out / "about.html").write_text(about_page(status, ctx))
        (out / "data" / "predictions.csv").write_text(api.text("/research/export/predictions.csv"))
        (out / "data" / "scores.csv").write_text(api.text("/research/export/scores.csv"))
        (out / "data" / "pairs.json").write_text(json.dumps(pairs, indent=1))
        (out / ".nojekyll").write_text("")
        return {"pages": 4 + len(pairs), "pairs": len(pairs), "demo": demo}
    finally:
        api.close()


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--out", type=Path, default=Path("_site"))
    ap.add_argument("--allow-demo", action="store_true", help="local preview of simulated data; never use for publishing")
    a = ap.parse_args()
    print(json.dumps(build(a.out, a.allow_demo)))
    sys.stdout.flush()


if __name__ == "__main__":
    main()
