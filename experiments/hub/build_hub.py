"""Assemble the GitHub Pages site: the paper portfolio at the root, one folder per experiment.

    python experiments/hub/build_hub.py --site _site --state state/portfolio.json [--update]

The root page is the combined 1000 CAD paper portfolio (portfolio.py); each experiment's page gets its own
1000 CAD portfolio box at the top.

Expects each experiment's pages already in place (_site/exp1/, _site/exp2/); a missing one is shown
as "not published yet" rather than failing the build. It also:
  - adds "Experiment 2" and "Paper portfolio" links to Exp1's page headers,
  - keeps Exp1's old addresses working: every Exp1 page that used to live at the site root gets a
    redirect page there, and its data files are copied back to /data/ (files can't redirect).
Standard library only, so it runs in any job.
"""
from __future__ import annotations

import argparse
import html
import json
import shutil
from datetime import UTC, datetime
from pathlib import Path

import portfolio
from portfolio import CAPITAL

e = html.escape
REPO_URL = "https://github.com/pindustrycom/experiment-1"
HUB_LINK = ('<a href="{root}exp2/index.html" style="color:var(--muted)">Experiment 2</a> '
            '<a href="{root}index.html" style="color:var(--muted)" data-hub>Paper portfolio</a>')

CSS = """
:root{--bg:#f7f7f5;--panel:#fff;--text:#1d1d1b;--muted:#6b6b66;--line:#e3e2dd;--accent:#2f5fd0;--lo:#2e7d4f;--hi:#c0392b;--band:#f1f0ec}
@media (prefers-color-scheme:dark){:root{--bg:#141413;--panel:#1d1d1b;--text:#ecebe6;--muted:#9c9b94;--line:#33322e;--accent:#7aa2ff;--lo:#6fcf97;--hi:#ff7b6b;--band:#252522}}
*{box-sizing:border-box}body{margin:0;background:var(--bg);color:var(--text);font:15px/1.5 system-ui,-apple-system,"Segoe UI",sans-serif}
a{color:var(--accent);text-decoration:none}a:hover{text-decoration:underline}
header{background:var(--panel);border-bottom:1px solid var(--line)}.wrap{max-width:1100px;margin:0 auto;padding:0 16px}
header .wrap{padding-top:12px;padding-bottom:12px}.brand{font-weight:650;font-size:17px;color:var(--text)}
main{padding:22px 0 40px}h1{font-size:24px;margin:0 0 4px}.sub{color:var(--muted);margin:0 0 22px}.muted{color:var(--muted)}
.cards{display:grid;grid-template-columns:repeat(auto-fit,minmax(320px,1fr));gap:16px}
.card{background:var(--panel);border:1px solid var(--line);border-radius:12px;padding:18px 20px;display:flex;flex-direction:column;gap:10px}
.card h2{font-size:19px;margin:0}.card .tag{font-size:12px;color:var(--muted);text-transform:uppercase;letter-spacing:.04em}
.card p{margin:0}.stats{display:grid;grid-template-columns:repeat(2,1fr);gap:8px}.stat{background:var(--band);border-radius:8px;padding:8px 10px}
.stat .k{font-size:12px;color:var(--muted)}.stat .v{font-size:18px;font-weight:650}.pos{color:var(--lo)}.neg{color:var(--hi)}
.go{margin-top:auto;font-weight:600}.card .v{font-size:26px;font-weight:700}.small{font-size:13px}
.panel{background:var(--panel);border:1px solid var(--line);border-radius:12px;padding:14px 16px;margin:14px 0}
.panel svg{width:100%;height:auto;display:block}.panel svg text{fill:var(--muted);font-size:11px}
header .nav a{color:var(--muted);margin-left:14px}main.wrap{padding-top:22px}:root{--a:#8a6fd1;--b:#d0632f}
footer{border-top:1px solid var(--line);color:var(--muted);font-size:13px;padding:16px 0 30px}
"""


MARK, MARK_END = "<!--portfolio-->", "<!--/portfolio-->"
BANNER_CSS = """
.big,.pf{background:var(--panel);border:1px solid var(--line);border-radius:14px;padding:18px 20px;margin:0 0 16px}
.pf .big{border:0;padding:0;margin:0 0 8px;background:none}
.big .k{font-size:13px;color:var(--muted);text-transform:uppercase;letter-spacing:.04em}
.big .v{font-size:52px;font-weight:750;line-height:1.1;font-variant-numeric:tabular-nums}
.big .d{color:var(--muted);font-size:14px}.big .d b{color:var(--text)}
.big .pos,.card .pos{color:var(--lo,#2e7d4f)}.big .neg,.card .neg{color:var(--hi,#c0392b)}.pf .small{font-size:13px}
.pos-line{font-size:14px;margin:6px 0}
@media (max-width:520px){.big .v{font-size:40px}}
"""


def _json(p: Path):
    try:
        return json.loads(p.read_text())
    except (OSError, ValueError):
        return None


def stat(k: str, v: str) -> str:
    return f'<div class="stat"><div class="k">{e(k)}</div><div class="v">{v}</div></div>'


def cad(v: float, sign: bool = False) -> str:
    s = f"{abs(v):,.2f} CAD"
    return (("+" if v >= 0 else "−") + s) if sign else s


def pl_class(v: float) -> str:
    return "pos" if v > 0.005 else "neg" if v < -0.005 else ""


def value_chart(rows: list[dict], keys: list[tuple[str, str]], w: int = 760, h: int = 200) -> str:
    if len(rows) < 3:
        return '<p class="muted small">The chart appears after a few trading days.</p>'
    vals = [r[k] for r in rows for k, _ in keys]
    lo, hi = min(vals + [CAPITAL]), max(vals + [CAPITAL])
    pad = max((hi - lo) * 0.1, 1.0)
    lo, hi = lo - pad, hi + pad
    L, R, T, B = 52, 8, 8, 22
    x = lambda i: L + (w - L - R) * i / (len(rows) - 1)  # noqa: E731
    y = lambda v: T + (h - T - B) * (1 - (v - lo) / (hi - lo))  # noqa: E731
    out = [f'<svg viewBox="0 0 {w} {h}" role="img" aria-label="Portfolio value in CAD">',
           f'<line x1="{L}" x2="{w - R}" y1="{y(CAPITAL):.1f}" y2="{y(CAPITAL):.1f}" stroke="var(--line)" stroke-dasharray="4 3"/>']
    for v in (lo + pad, CAPITAL, hi - pad):
        out.append(f'<text x="{L - 6}" y="{y(v) + 4:.1f}" text-anchor="end">{v:,.0f}</text>')
    for i, anchor in ((0, "start"), (len(rows) - 1, "end")):
        out.append(f'<text x="{x(i):.1f}" y="{h - 5}" text-anchor="{anchor}">{rows[i]["date"]}</text>')
    for k, color in keys:
        pts = " ".join(f"{x(i):.1f},{y(r[k]):.1f}" for i, r in enumerate(rows))
        out.append(f'<polyline points="{pts}" fill="none" stroke="{color}" stroke-width="{2.4 if k == "combined" else 1.4}"/>')
    return "".join(out) + "</svg>"


def big_number(rows: list[dict], key: str, label: str, start: str | None = None) -> str:
    v = rows[-1][key]
    pl = v - CAPITAL
    day = v - rows[-2][key] if len(rows) > 1 else 0.0
    since = start or rows[0]["date"]
    return (f'<div class="big"><div class="k">{e(label)}</div><div class="v {pl_class(pl)}">{cad(pl, True)}</div>'
            f'<div class="d">Worth <b>{cad(v)}</b> ({100 * pl / CAPITAL:+.2f}%) · today <span class="{pl_class(day)}">{cad(day, True)}</span>'
            f' · fake {CAPITAL:,.0f} CAD from {e(since)} · last update {e(rows[-1]["date"])}</div></div>')


def exp1_positions(t: dict) -> str:
    book = t.get("exp1_book") or []
    if not t.get("exp1_book_date"):
        return "No positions yet: the first book is formed after the next nightly run."
    if not book:
        return f"In cash since {e(t['exp1_book_date'])}: no pair has a call today."
    return f"Holding from the {e(t['exp1_book_date'])} close: " + "; ".join(
        f"<b>long {e(b['long'])} / short {e(b['short'])}</b> <span class=\"muted\">({e(b['why'])})</span>" for b in book)


def portfolio_banner(t: dict | None, which: str, root: str) -> str:
    """The big P/L box put at the top of an experiment's page."""
    if not t:
        return ""
    name = {"exp1": "Exp1 paper portfolio", "exp2": "Exp2 paper portfolio"}[which]
    pos = exp1_positions(t) if which == "exp1" else e(t.get("exp2_position") or "No positions yet.")
    return (f'{MARK}<style>{BANNER_CSS}</style><div class="pf">{big_number(t["rows"], which, name + ": profit / loss", t.get("start"))}'
            f'<div class="pos-line">{pos}</div><div class="muted small">Paper money only, traded daily from this experiment\'s signals. '
            f'<a href="{root}index.html">Combined portfolio →</a></div></div>{MARK_END}')


def inject(f: Path, block: str) -> bool:
    if not f.exists() or not block:
        return False
    s = f.read_text()
    if MARK in s:
        s = s[:s.index(MARK)] + s[s.index(MARK_END) + len(MARK_END):]
    if '<main class="wrap">' not in s:
        return False
    f.write_text(s.replace('<main class="wrap">', '<main class="wrap">' + block, 1))
    return True


def sleeve_card(t: dict, key: str, title: str, href: str, what: str, pos: str) -> str:
    rows = t["rows"]
    v = rows[-1][key]
    pl = v - CAPITAL
    return (f'<div class="card"><div class="tag">{e(title)}</div><div class="v {pl_class(pl)}">{cad(pl, True)}</div>'
            f'<div class="muted small">worth {cad(v)} ({100 * pl / CAPITAL:+.2f}%)</div><p>{what}</p><p class="small">{pos}</p>'
            f'<a class="go" href="{href}">Open →</a></div>')


def hub_page(site: Path, t: dict | None = None) -> str:
    built = datetime.now(UTC).strftime("%Y-%m-%d %H:%M UTC")
    if t is None:
        body = '<p class="muted">The portfolio appears after its first run.</p>'
    else:
        rows = t["rows"]
        fx = "" if t.get("fx_known") else " USD/CAD rate unavailable: values exclude currency moves."
        body = f"""{big_number(rows, "combined", "Combined paper portfolio: profit / loss", t.get("start"))}
<div class="panel">{value_chart(rows, [("combined", "var(--accent)"), ("exp1", "var(--a)"), ("exp2", "var(--b)")])}
<div class="small muted"><span style="color:var(--accent)">━</span> combined <span style="color:var(--a)">━</span> Exp1
<span style="color:var(--b)">━</span> Exp2 · dashed line: the starting 1,000 CAD</div></div>
<p class="small muted">Half of the money follows each experiment. Trades every trading day; US positions, valued in CAD at the
Bank of Canada rate. No costs or interest.{fx}</p>
<div class="cards">{sleeve_card(t, "exp1", "Exp1 · pairs", "exp1/index.html", "Long the stock Exp1 favours in each pair, short the other.", exp1_positions(t))}
{sleeve_card(t, "exp2", "Exp2 · volatility-timed", "exp2/index.html", "Sells SPY volatility when it is overpriced, holds SPY in calm markets, plus a stock book.", e(t.get("exp2_position") or ""))}</div>"""
    return f"""<!doctype html><html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1"><meta name="robots" content="noindex">
<title>Paper portfolio</title><style>{CSS}{BANNER_CSS}</style></head><body>
<header><div class="wrap"><a class="brand" href="index.html">Paper portfolio</a> <span class="nav"><a href="exp1/index.html">Experiment 1</a>
<a href="exp2/index.html">Experiment 2</a></span></div></header>
<main class="wrap">{body}</main>
<footer><div class="wrap">Fake money: no orders are placed anywhere and nothing here is investment advice. Built {built}.
<a href="{REPO_URL}">Source</a>.</div></footer></body></html>"""


def redirect_page(target: str) -> str:
    t = e(target)
    return (f'<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="robots" content="noindex">'
            f'<meta http-equiv="refresh" content="0; url={t}"><link rel="canonical" href="{t}"><title>Moved</title></head>'
            f'<body><p>This page moved to <a href="{t}">{t}</a>.</p></body></html>')


def link_exp1_to_hub(site: Path) -> int:
    d, n = site / "exp1", 0
    for f in d.rglob("*.html"):
        depth = len(f.relative_to(d).parts)  # exp1/index.html -> 1, exp1/pairs/x.html -> 2
        s = f.read_text()
        if "data-hub" in s or "</nav></div></header>" not in s:
            continue
        f.write_text(s.replace("</nav></div></header>", "</nav>" + HUB_LINK.format(root="../" * depth) + "</div></header>", 1))
        n += 1
    return n


def keep_old_addresses(site: Path) -> int:
    """Exp1 used to be the whole site. Point its old addresses at /exp1/."""
    d, n = site / "exp1", 0
    for f in d.rglob("*"):
        if not f.is_file():
            continue
        rel = f.relative_to(d)
        if rel.as_posix() in ("index.html", ".nojekyll"):
            continue
        dest = site / rel
        if dest.exists():
            continue
        dest.parent.mkdir(parents=True, exist_ok=True)
        if f.suffix == ".html":
            dest.write_text(redirect_page("../" * (len(rel.parts) - 1) + "exp1/" + rel.as_posix()))
        else:
            shutil.copy2(f, dest)
        n += 1
    return n


def build(site: Path, state: Path | None = None, update: bool = False) -> dict:
    site.mkdir(parents=True, exist_ok=True)
    linked = link_exp1_to_hub(site) if (site / "exp1").is_dir() else 0
    moved = keep_old_addresses(site) if (site / "exp1").is_dir() else 0
    t, rep = None, {}
    if state is not None:
        r = portfolio.run(site, state, update)
        t, rep = r["table"], r["portfolio"]
        inject(site / "exp1" / "index.html", portfolio_banner(t, "exp1", "../"))
        inject(site / "exp2" / "index.html", portfolio_banner(t, "exp2", "../"))
    (site / "index.html").write_text(hub_page(site, t))
    (site / ".nojekyll").write_text("")
    return {"exp1_pages_linked": linked, "old_addresses_kept": moved, "exp2": (site / "exp2" / "index.html").exists(), **rep}


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--site", type=Path, default=Path("_site"))
    ap.add_argument("--state", type=Path, help="paper portfolio state file (portfolio.json); omit to skip the portfolio")
    ap.add_argument("--update", action="store_true", help="move Experiment 1's book forward (the nightly full run only)")
    a = ap.parse_args()
    print(json.dumps(build(a.site, a.state, a.update), default=str))
