"""Assemble the GitHub Pages site: the paper portfolio at the root, one folder per experiment.

    python experiments/hub/build_hub.py --site _site --state state/portfolio.json [--update]

The root page is the combined 1000 CAD paper portfolio (portfolio.py); each experiment's page gets its own
1000 CAD portfolio box at the top.

Expects each experiment's pages already in place (_site/exp1/, _site/exp2/); a missing one is shown
as "not published yet" rather than failing the build. Experiments 4 to 6 are static pages kept in this repository
(experiments/exp4_resonance/, exp5_vision/, exp6_alphabet/) and are copied to _site/exp4/ to _site/exp6/. It also:
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
EXPERIMENTS = Path(__file__).resolve().parent.parent
EXP4 = EXPERIMENTS / "exp4_resonance"
# Static experiment pages: (site folder, source folder, card title, card text). No money involved in any of them.
STATIC = [
    ("exp4", EXP4, "Exp4 · resonance playground",
     "A 3D model of the resonance architecture by Joaquin Philco: retina and V1, time-based somas with resonators on the axon, "
     "concept structures and the rules that ask, allocate and sleep. Show it a digit in a colour, blur it, or open one cell."),
    ("exp5", EXPERIMENTS / "exp5_vision", "Exp5 · vision stack",
     "Can better vision make Joaquin Philco's resonance agent see natural images? Eleven methods on CIFAR and Imagenette: Gabor and "
     "learned V1, somas with an axon resonance machine, a deeper somatic stack, feedback from memory and super labels."),
    ("exp6", EXPERIMENTS / "exp6_alphabet", "Exp6 · alphabet and words",
     "The same resonance machinery learns letters, numbers and words from images, then words name colour patches with no "
     "stored labels. Includes a 3D view: watch the eye read a word, or a patch evoke its word."),
    ("exp6_live", EXPERIMENTS / "exp6_live", "Exp6 live · the model in your browser",
     "The trained exp6 model runs in the page: type a word in any font, draw a character or pick a colour, and see which "
     "letter, word and naming structures resonate, with a 3D view driven by the computed values."),
]
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


# Frosted-glass theme (black page, deep purple glass blocks, always dark), laid over every page of the site (the hub and each experiment's pages) so they match.
GLASS, GLASS_END = "<!--glass-->", "<!--/glass-->"
GLASS_CSS = """
:root,:root:root{color-scheme:dark;--text:#f5f5f7;--muted:#98989f;--line:rgba(255,255,255,.08);--panel:rgba(255,255,255,.055);--band:rgba(255,255,255,.07);
--accent:#a593d4;--a:#7a62b3;--b:#8e8e93;--lo:#30d158;--hi:#ff453a;--md:#ffd60a;--lo-bg:rgba(48,209,88,.14);--md-bg:rgba(255,214,10,.13);--hi-bg:rgba(255,69,58,.15);
--glass-line:rgba(255,255,255,.12);--glass-bg:linear-gradient(150deg,rgba(52,26,92,.38) 0%,rgba(28,14,52,.22) 45%,rgba(255,255,255,.03) 100%);--glass-shadow:0 20px 50px rgba(0,0,0,.55);--glass-hi:rgba(255,255,255,.10);--grid:rgba(255,255,255,.07)}
html{background:#000;min-height:100%}
body{background:transparent!important;min-height:100vh;color:var(--text);font-family:-apple-system,BlinkMacSystemFont,"SF Pro Text","Segoe UI",system-ui,sans-serif;
-webkit-font-smoothing:antialiased;letter-spacing:-.005em}main.wrap{padding-top:22px}
h1,.big .v,.card .v,.brand{letter-spacing:-.02em}
header{position:sticky;top:0;z-index:10;background:rgba(0,0,0,.55)!important;border-bottom:1px solid var(--glass-line)!important;
-webkit-backdrop-filter:blur(30px) saturate(180%);backdrop-filter:blur(30px) saturate(180%)}
.card,.panel,.tile,.big,.pf,.banner,.scroll>table,details>.panel{background:var(--glass-bg)!important;border:1px solid var(--glass-line)!important;
border-radius:20px!important;box-shadow:var(--glass-shadow),inset 0 1px 0 var(--glass-hi);-webkit-backdrop-filter:blur(30px) saturate(180%);backdrop-filter:blur(30px) saturate(180%)}
.scroll>table{border-collapse:separate;border-spacing:0;overflow:hidden}table{background:transparent!important}
.stat{background:var(--band)!important;border:1px solid var(--glass-line)!important;border-radius:14px!important}
.pf .big{background:none!important;border:0!important;box-shadow:none!important;-webkit-backdrop-filter:none;backdrop-filter:none}
th,td{border-bottom-color:var(--line)!important}footer{border-top:1px solid var(--glass-line)!important}
nav a.on{background:rgba(70,40,120,.35)!important;color:var(--text)!important;border-radius:999px}
.tag{border-color:var(--glass-line)!important}.banner.warn,.banner.danger{background:rgba(52,26,92,.4)!important}
svg text{fill:var(--muted)!important}.chart .grid{stroke:var(--grid)}
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


def theme(site: Path) -> int:
    """Lay the glass theme over every experiment page (replacing any earlier copy)."""
    n = 0
    for d in ("exp1", "exp2"):
        for f in (site / d).rglob("*.html") if (site / d).is_dir() else []:
            s = f.read_text()
            if GLASS in s:
                s = s[:s.index(GLASS)] + s[s.index(GLASS_END) + len(GLASS_END):]
            if "</head>" not in s:
                continue
            f.write_text(s.replace("</head>", f"{GLASS}<style>{GLASS_CSS}</style>{GLASS_END}</head>", 1))
            n += 1
    return n


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
<div class="panel">{value_chart(rows, [("combined", "var(--text)"), ("exp1", "var(--a)"), ("exp2", "var(--b)")])}
<div class="small muted"><span style="color:var(--text)">━</span> combined <span style="color:var(--a)">━</span> Exp1
<span style="color:var(--b)">━</span> Exp2 · dashed line: the starting 1,000 CAD</div></div>
<p class="small muted">Half of the money follows each experiment. Trades every trading day; US positions, valued in CAD at the
Bank of Canada rate. No costs or interest.{fx}</p>
<div class="cards">{sleeve_card(t, "exp1", "Exp1 · pairs", "exp1/index.html", "Long the stock Exp1 favours in each pair, short the other.", exp1_positions(t))}
{sleeve_card(t, "exp2", "Exp2 · volatility-timed", "exp2/index.html", "Sells SPY volatility when it is overpriced, holds SPY in calm markets, plus a stock book.", e(t.get("exp2_position") or ""))}</div>"""
    shown = [x for x in STATIC if (site / x[0] / "index.html").exists()]
    exp4_nav = "".join(f'\n<a href="{d}/index.html">Experiment {d[3:].replace("_", " ")}</a>' for d, *_ in shown)
    exp4_card = ('<h2 style="font-size:18px;margin:28px 0 10px">Other experiments</h2><div class="cards">'
                 + "".join(f'<div class="card"><div class="tag">{e(title)}</div><p>{e(text)}</p>'
                           f'<a class="go" href="{d}/index.html">Open →</a></div>' for d, _, title, text in shown)
                 + '</div>') if shown else ""
    return f"""<!doctype html><html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1"><meta name="robots" content="noindex">
<title>Paper portfolio</title><style>{CSS}{BANNER_CSS}{GLASS_CSS}</style></head><body>
<header><div class="wrap"><a class="brand" href="index.html">Paper portfolio</a> <span class="nav"><a href="exp1/index.html">Experiment 1</a>
<a href="exp2/index.html">Experiment 2</a>{exp4_nav}</span></div></header>
<main class="wrap">{body}{exp4_card}</main>
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


def add_static(site: Path, name: str, src: Path) -> bool:
    """Copy one static experiment page (folder with an index.html) into the site."""
    if not (src / "index.html").exists():
        return False
    shutil.copytree(src, site / name, dirs_exist_ok=True)
    return True


def add_exp4(site: Path, src: Path = EXP4) -> bool:
    """Copy Experiment 4's static page into the site."""
    return add_static(site, "exp4", src)


def build(site: Path, state: Path | None = None, update: bool = False) -> dict:
    site.mkdir(parents=True, exist_ok=True)
    for name, src, *_ in STATIC:
        add_static(site, name, src)
    linked = link_exp1_to_hub(site) if (site / "exp1").is_dir() else 0
    moved = keep_old_addresses(site) if (site / "exp1").is_dir() else 0
    t, rep = None, {}
    if state is not None:
        r = portfolio.run(site, state, update)
        t, rep = r["table"], r["portfolio"]
        inject(site / "exp1" / "index.html", portfolio_banner(t, "exp1", "../"))
        inject(site / "exp2" / "index.html", portfolio_banner(t, "exp2", "../"))
    theme(site)
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
