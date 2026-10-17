"""Assemble the GitHub Pages site: an experiments hub at the root, one folder per experiment.

    python experiments/hub/build_hub.py --site _site

Expects each experiment's pages already in place (_site/exp1/, _site/exp2/); a missing one is shown
as "not published yet" rather than failing the build. It also:
  - adds "Experiment 2" and "All experiments" links to Exp1's page headers,
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

e = html.escape
REPO_URL = "https://github.com/pindustrycom/experiment-1"
HUB_LINK = ('<a href="{root}exp2/index.html" style="color:var(--muted)">Experiment 2</a> '
            '<a href="{root}index.html" style="color:var(--muted)">All experiments</a>')

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
.go{margin-top:auto;font-weight:600}
footer{border-top:1px solid var(--line);color:var(--muted);font-size:13px;padding:16px 0 30px}
"""


def _json(p: Path):
    try:
        return json.loads(p.read_text())
    except (OSError, ValueError):
        return None


def stat(k: str, v: str) -> str:
    return f'<div class="stat"><div class="k">{e(k)}</div><div class="v">{v}</div></div>'


def exp1_card(site: Path) -> str:
    d = site / "exp1"
    if not (d / "index.html").exists():
        body = '<p class="muted">Not published yet.</p>'
    else:
        pairs = _json(d / "data" / "pairs.json") or []
        v2 = [(p.get("si_v2") or {}).get("value") for p in pairs]
        v2 = [v for v in v2 if v is not None]
        top = max(pairs, key=lambda p: (p.get("si_v2") or {}).get("value") or -1, default=None)
        top_s = "—"
        if top and (top.get("si_v2") or {}).get("value") is not None:
            top_s = f'{e(top["company_a"]["ticker"])}/{e(top["company_b"]["ticker"])} {top["si_v2"]["value"]:.0f}'
        body = '<div class="stats">' + stat("Pairs monitored", str(len(pairs))) + stat("Highest SI-v2", top_s) + \
            stat("Average SI-v2", f"{sum(v2) / len(v2):.1f}" if v2 else "—") + stat("Scale", "0–100 risk") + "</div>"
    return f"""<div class="card"><div class="tag">Experiment 1</div><h2>Splittability Monitor</h2>
<p>Ranks pairs of similar companies by the risk that their share prices drift apart, from market data (SI-v1) plus news,
fundamentals and structure (SI-v2), and forward-tests those rankings.</p>{body}<a class="go" href="exp1/index.html">Open Experiment 1 →</a></div>"""


def exp2_card(site: Path) -> str:
    d = site / "exp2"
    s = _json(d / "data" / "summary.json") if (d / "index.html").exists() else None
    if s is None or "performance_since_2019" not in s:
        body = '<p class="muted">Not published yet: it appears after its first daily run.</p>'
    else:
        latest = s.get("latest") or {}
        if latest.get("exp_ret") is not None:
            v = latest["exp_ret"]
            pred = f'<span class="{"pos" if v > 0 else "neg"}">{v:+.2f}%</span>'
        else:
            pred = "—"
        perf = (s["performance_since_2019"].get("portfolio") or {})
        spy = (s["performance_since_2019"].get("spy") or {})
        yr = f'{perf["annual_pct"]:+.1f}% a year' if perf.get("annual_pct") is not None else "—"
        cal = s.get("calibration_live") or {}
        if not cal.get("n"):
            cal = s.get("calibration_last_120") or {}
        band = f'{100 * cal["in68"]:.0f}% of {cal["n"]}' if cal.get("n") else "collecting"
        body = '<div class="stats">' + stat(f'Predicted return {latest.get("date", "—")}', pred) + \
            stat("Inside the 68% band", band) + stat("Since 2019 (SPY)", f'{yr} <span class="muted" style="font-size:13px">'
                                                     f'({spy.get("annual_pct", float("nan")):+.1f}%)</span>') + \
            stat("Worst drop since 2019", f'{perf.get("max_dd_pct", float("nan")):.0f}%') + "</div>"
    return f"""<div class="card"><div class="tag">Experiment 2</div><h2>Volatility-timed portfolio</h2>
<p>A paper portfolio that sells volatility insurance on SPY when it is overpriced and markets are calm, plus a regime-timed SPY
position and a stock book without collapse-prone names. Each morning it predicts the day's return; each evening it logs the
error and learns from it.</p>{body}<a class="go" href="exp2/index.html">Open Experiment 2 →</a></div>"""


def hub_page(site: Path) -> str:
    built = datetime.now(UTC).strftime("%Y-%m-%d %H:%M UTC")
    return f"""<!doctype html><html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1"><meta name="robots" content="noindex">
<title>Market experiments</title><style>{CSS}</style></head><body>
<header><div class="wrap"><a class="brand" href="index.html">Market experiments</a></div></header>
<main class="wrap"><h1>Experiments</h1><p class="sub">Research experiments that make predictions about the stock market and track how
they turn out. Each one logs its predictions before the outcome is known. Updated after every US market close.</p>
<div class="cards">{exp1_card(site)}{exp2_card(site)}</div></main>
<footer><div class="wrap">Paper tracking only: nothing here is investment advice and no orders are placed anywhere. Built {built}.
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
        if "All experiments" in s or "</nav></div></header>" not in s:
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


def build(site: Path) -> dict:
    site.mkdir(parents=True, exist_ok=True)
    linked = link_exp1_to_hub(site) if (site / "exp1").is_dir() else 0
    moved = keep_old_addresses(site) if (site / "exp1").is_dir() else 0
    (site / "index.html").write_text(hub_page(site))
    (site / ".nojekyll").write_text("")
    return {"exp1_pages_linked": linked, "old_addresses_kept": moved, "exp2": (site / "exp2" / "index.html").exists()}


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--site", type=Path, default=Path("_site"))
    print(json.dumps(build(ap.parse_args().site)))
