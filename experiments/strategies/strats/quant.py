"""The Quant lab page (/quant/ on the site, in the Markets group): three studies run on the pots' own data each night.

  1. Implied volatility surface  SPY, QQQ and IWM option chains worked back to implied vols and drawn in 3D
                                 (surface.py), with the daily options summary from openbb-ideas.yml
  2. Market regimes              K-Means on SPY's return and volatility (regimes.py), and each replayed pot by regime
  3. Event-driven backtest       the trend and reversal pots run through an event loop with fills at the next open and
                                 real costs (events.py), next to the vectorized replay the pots page shows

    python -m strats quant --data data --options options_daily.csv --out quant_site [--no-network]

Paper research only: nothing here trades.
"""
from __future__ import annotations

import html
import json
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import pandas as pd

from . import copy, crypto, data, events, live, regimes, rules, surface

e = html.escape


def btc_daily(btc: pd.DataFrame, dates: pd.DatetimeIndex) -> pd.Series | None:
    """The BTC breakout rule's equity, marked at each US close, as daily returns on the stock-market calendar."""
    if btc is None or len(btc) <= crypto.TREND:
        return None
    eq = crypto.run_rule(btc)["equity"]
    t = (dates + pd.Timedelta(hours=16)).tz_localize("America/New_York").tz_convert("UTC").tz_localize(None)
    pos = eq.index.searchsorted(t - crypto.BAR, side="right") - 1
    v = pd.Series([eq.iloc[i] if i >= 0 else np.nan for i in pos], index=dates)
    return v.pct_change().where(pos > 0)


def run(data_dir: Path, options_csv: Path | None, out: Path, fetch: bool = True) -> dict:
    rep: dict = {}
    if fetch:
        rep["chains"] = surface.fetch(data_dir)
    members = [m.replace(".", "-") for m in (sorted(pd.read_csv(data_dir / "sp500.csv")["ticker"].astype(str)) if (data_dir / "sp500.csv").exists() else [])]
    cp = copy.load(data_dir) if (data_dir / "copy").exists() else None
    tickers = sorted(set(rules.TREND_ETFS) | set(members) | (copy.tickers(cp) if copy.has_data(cp) else set()))
    bars = data.load_bars(data_dir, tickers)
    px = bars["close"]
    spy = px["SPY"].dropna()
    px = px.loc[px.index.isin(spy.index)]
    opn = bars["open"].reindex(px.index) if not bars["open"].empty else None

    # 1. surfaces
    surf = {}
    for t in surface.TICKERS:
        got = surface.latest(data_dir, t)
        if got is not None:
            s = surface.surface(*got)
            if s:
                surf[t] = s
    hist = surface.options_history(options_csv)
    rv = np.log(spy).diff().iloc[-21:].std() * np.sqrt(252)
    rep["surface"] = {t: {k: s[k] for k in ("date", "atm_30d", "skew_30d", "contracts", "cboe_median_gap", "dropped_expiries")} for t, s in surf.items()}

    # 2. regimes, on every pot that has a replay
    rp = live.replay(px, members, cp=cp)
    b = btc_daily(crypto.load(data_dir), rp.index)
    if b is not None and b.notna().sum() > 100:
        rp = rp.assign(btc_donchian=b)
    reg = regimes.run(spy, rp, live.REPLAY_FROM)
    rep["regimes"] = {"today": reg["today"], "stable_pct": reg["stable_pct"],
                      "sharpe": {k: {n: x["sharpe"] for n, x in v.items()} for k, v in reg["by_regime"].items()}}

    # 3. event-driven against the replay
    ev = {}
    for key, cols in (("trend", rules.TREND_ETFS), ("reversal", members)):
        cols = [c for c in cols if c in px.columns]
        c = px[cols]
        ev |= events.compare(c, opn.reindex(columns=cols) if opn is not None else None, members, live.REPLAY_FROM, rp, keys=(key,))
    rep["events"] = {k: [(r["setting"], r.get("cagr_pct"), r.get("sharpe")) for r in v["rows"]] for k, v in ev.items()}

    names = {k: v[0] for k, v in rules.STRATEGIES.items()} | {"SPY": "SPY (holding the market)"}
    payload = {"surface": surf, "options_history": hist, "realized_21d": round(float(rv), 4), "regimes": reg, "events": ev,
               "names": names, "built": datetime.now(UTC).strftime("%Y-%m-%d %H:%M UTC"), "last_close": str(spy.index[-1].date()),
               "costs": [{"label": s[0], "fill": s[1], "commission_bp": s[2] * 1e4, "slip_stock_bp": s[3][0] * 1e4,
                          "slip_etf_bp": s[3][1] * 1e4, "borrow_pct": s[4] * 100, "cash": s[5]} for s in events.SETTINGS]}
    (out / "data").mkdir(parents=True, exist_ok=True)
    js = json.dumps(payload, default=_num, separators=(",", ":"))
    (out / "data" / "quant.json").write_text(js)
    (out / "index.html").write_text(page(payload, js))
    return rep


def _num(x):
    if isinstance(x, (np.integer,)):
        return int(x)
    if isinstance(x, (np.floating,)):
        return None if not np.isfinite(x) else float(x)
    return str(x)


def pct(v, sign: bool = True, nd: int = 1) -> str:
    if v is None:
        return "–"
    return f"{v:+.{nd}f}%" if sign else f"{v:.{nd}f}%"


def findings(p: dict) -> list[str]:
    """Plain sentences from the numbers, so the page reads the same way whatever the data says."""
    out = []
    reg, names = p["regimes"], p["names"]
    for k, v in reg["by_regime"].items():
        if len(v) < 2 or k == "SPY":
            continue
        best = max(v, key=lambda n: v[n]["sharpe"])
        worst = min(v, key=lambda n: v[n]["sharpe"])
        out.append(f"<b>{e(names.get(k, k))}</b> earns most in <b>{best}</b> markets (Sharpe {v[best]['sharpe']:.2f}, "
                   f"{pct(v[best]['ann_ret_pct'])} a year) and least in <b>{worst}</b> ones (Sharpe {v[worst]['sharpe']:.2f}, "
                   f"{pct(v[worst]['ann_ret_pct'])} a year).")
    return out


def event_findings(p: dict) -> list[str]:
    out = []
    for k, v in p["events"].items():
        rows = {r["setting"]: r for r in v["rows"]}
        vec = rows.get("Vectorized replay (the pots' backtest)", {})
        nxt = rows.get(events.SETTINGS[1][0], {})
        real = rows.get(events.SETTINGS[2][0], {})
        if not vec or not real:
            continue
        out.append(f"<b>{e(p['names'].get(k, k))}</b>: the replay says {pct(vec.get('cagr_pct'))} a year; filled at the next open it is "
                   f"{pct(nxt.get('cagr_pct'))}, and with spreads, slippage and borrow {pct(real.get('cagr_pct'))} "
                   f"(costs about {real.get('cost_pct_per_year', 0):.1f}% of the pot a year).")
    return out


def page(p: dict, js: str) -> str:
    reg = p["regimes"]
    today = reg["today"]
    sv = p["surface"]
    spy = sv.get("SPY")
    vrp = (spy["atm_30d"] - p["realized_21d"]) * 100 if spy and spy.get("atm_30d") is not None else None
    surf_stats = ("<p class=\"muted\">No option chain saved yet: the surface appears after the next nightly run.</p>" if not sv else
                  '<div class="stats" id="sv-stats"></div>')
    order = ["trend", "reversal", "best_ideas", "conviction", "pelosi", "btc_donchian", "SPY"]
    keys = [k for k in order if k in reg["by_regime"]] + [k for k in reg["by_regime"] if k not in order]
    head = "".join(f'<th><span class="sw" style="background:var(--{n.lower()})"></span>{n}<br><span class="muted small">'
                   f'{reg["share_pct"][n]:.0f}% of days</span></th>' for n in regimes.NAMES)
    body = []
    for k in keys:
        cells = []
        for n in regimes.NAMES:
            x = reg["by_regime"][k].get(n)
            if not x:
                cells.append("<td>–</td>")
                continue
            cls = "pos" if x["sharpe"] > 0.05 else "neg" if x["sharpe"] < -0.05 else ""
            cells.append(f'<td><b class="{cls}">{x["sharpe"]:.2f}</b><br><span class="muted small">{pct(x["ann_ret_pct"])} / yr · {x["days"]} d</span></td>')
        body.append(f'<tr><td>{e(p["names"].get(k, k))}</td>{"".join(cells)}</tr>')
    cen = "".join(f'<div class="stat"><div class="k"><span class="sw" style="background:var(--{n.lower()})"></span>{n} centre</div>'
                  f'<div class="v">{pct(c["ret_pct"], nd=0)} <span class="muted small">/ yr at {c["vol_pct"]:.0f}% vol</span></div></div>'
                  for n, c in reg["centres"].items())
    ev_tables = []
    for k, v in p["events"].items():
        rows = "".join(f'<tr><td>{e(r["setting"])}</td><td>{pct(r.get("cagr_pct"))}</td><td>{r.get("sharpe", "–")}</td>'
                       f'<td>{pct(r.get("max_dd_pct"))}</td><td>{r.get("trades", "–")}</td>'
                       f'<td>{"–" if r.get("cost_pct_per_year") is None else f"{r["cost_pct_per_year"]:.2f}%"}</td></tr>' for r in v["rows"])
        chk = v.get("check") or {}
        ev_tables.append(
            f'<div class="panel"><h3>{e(p["names"].get(k, k))}</h3><div class="scroll"><table><tr><th>How it is run</th><th>Per year</th><th>Sharpe</th>'
            f'<th>Worst drop</th><th>Trades</th><th>Costs / yr</th></tr>{rows}</table></div>'
            f'<div class="chart" id="ev-{k}"></div>'
            + (f'<p class="muted small">Check: with the replay\'s own assumptions the event loop\'s daily returns correlate {chk["corr"]:.3f} with the '
               f'replay\'s (mean gap {chk["mean_abs_gap_bp"]:.1f} bp a day). The gap is the replay earning its target weights every day, '
               f'a free daily rebalance; the event loop holds shares that drift.</p>' if chk else "") + "</div>")
    costs = "".join(f'<li>{e(c["label"])}: fills at the {"same close" if c["fill"] == "close" else "next open"}, '
                    f'{c["commission_bp"]:.0f} bp commission, {c["slip_stock_bp"]:.0f} bp spread and slippage on stocks and '
                    f'{c["slip_etf_bp"]:.0f} bp on ETFs, {c["borrow_pct"]:.1f}% a year to borrow shorts'
                    + (f', whole shares only, starting with {c["cash"]:,.0f} USD (about 1,000 CAD)' if c["cash"] else ", fractional shares")
                    + "</li>" for c in p["costs"])
    vrp_txt = "" if vrp is None else (
        f'<p class="small">SPY\'s 30-day implied vol is <b>{spy["atm_30d"] * 100:.1f}%</b>; its realized vol over the last 21 sessions was '
        f'<b>{p["realized_21d"] * 100:.1f}%</b>. The gap ({vrp:+.1f} points) is the volatility risk premium that Exp2 sells.</p>')
    return f"""<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<meta name="robots" content="noindex,nofollow"><meta name="referrer" content="no-referrer"><title>Quant lab</title><style>{CSS}</style></head><body>
<main class="wrap">
<h1>Quant lab</h1>
<p class="muted">Three classic quant studies, run every night on the data the pots already collect. Research only: nothing here trades.
Last close {e(p["last_close"])} · built {e(p["built"])}.</p>
<nav class="tabs" aria-label="Studies"><a href="#surface">1 · Volatility surface</a><a href="#regimes">2 · Market regimes</a><a href="#events">3 · Event-driven test</a></nav>

<section id="surface"><h2>1 · Implied volatility surface</h2>
<p>Options traders trade volatility, not price. Take every option on the chain, work back from its price to the volatility
that Black's formula needs to give that price, and plot it by strike and expiry. If the market believed the textbook model,
the surface would be flat. It is not: puts below the market cost more (the skew) and the level changes with expiry.</p>
<div class="seg" id="sv-pick" role="group" aria-label="Underlying"></div>
{surf_stats}
<div class="panel surf"><canvas id="sv-3d" role="img" aria-label="Implied volatility by strike and expiry, 3D; drag to turn it"></canvas>
<div class="muted small">Drag to turn it. Across: strike as % of the forward price (left = puts below the market). Depth: days to expiry.
Height and colour: implied volatility. <span id="sv-scale"></span></div></div>
<div class="grid2"><div class="panel"><h3>Smile by expiry</h3><div class="chart" id="sv-smile"></div></div>
<div class="panel"><h3>Term structure at the money</h3><div class="chart" id="sv-term"></div></div></div>
{vrp_txt}
<div class="panel"><h3>Daily options summary (since the collection began)</h3><div class="scroll" id="sv-hist"></div>
<p class="muted small">From the nightly options job (Cboe chains via OpenBB, {p["options_history"].get("names", 0)} names on the latest day).
Skew = 25-delta put vol minus 25-delta call vol at 30 days; put/call = open interest.</p></div>
<p class="muted small">How: forward and discount from put-call parity on near-the-money strikes; mid prices of out-of-the-money options only;
quotes with no bid or a spread wider than the mid are dropped. Our vols differ from Cboe's own by a median of
<span id="sv-gap"></span> vol points.</p>
</section>

<section id="regimes"><h2>2 · Market regimes</h2>
<p>Markets do not behave the same way all the time. K-Means splits every session since {e(reg["fit_from"])} by SPY's past
month of return and volatility into three groups; the data names them by how volatile they are. Then each pot's replay is
scored in each regime, using the regime known at the previous close.</p>
<div class="big"><div class="k">Today ({e(today["date"])})</div><div class="v"><span class="sw lg" style="background:var(--{today["regime"].lower()})"></span>{e(today["regime"])}</div>
<div class="muted">SPY's last month: {pct(today["ret_pct"])} a year at {today["vol_pct"]:.0f}% volatility.</div></div>
<div class="stats">{cen}</div>
<div class="grid2"><div class="panel"><h3>Every session, by cluster</h3><div class="chart" id="rg-scatter"></div></div>
<div class="panel"><h3>SPY since {e(live.REPLAY_FROM[:4])}, shaded by regime</h3><div class="chart" id="rg-time"></div></div></div>
<div class="panel"><h3>Sharpe ratio of each pot's replay, by regime</h3><div class="scroll"><table class="rg"><tr><th>Pot</th>{head}</tr>{"".join(body)}</table></div>
<ul class="small">{"".join(f"<li>{x}</li>" for x in findings(p))}</ul>
<p class="muted small">Replays from {e(live.REPLAY_FROM)} after 1 bp costs (the pots page's numbers). Earnings drift and insider buying have no
history to replay, so they are not here yet. The clusters are fitted on the whole history, so they describe the past; refitted each year on
the data up to then, {reg["stable_pct"]:.0f}% of sessions get the same label.</p></div>
</section>

<section id="events"><h2>3 · Event-driven backtest</h2>
<p>The pots' replay is vectorized: it loops over a table of prices and assumes each book is bought at the close its signal came from.
A real system handles one event at a time: a price arrives, the strategy makes a signal, an order goes out and fills later, with costs.
Here the trend and reversal pots go through such a loop, with the same rule functions as the live pots.</p>
<div class="flow" aria-hidden="true"><span>Open: queued orders fill</span>→<span>Close: price arrives</span>→<span>Strategy: signal from data so far</span>→<span>Portfolio: share orders</span>→<span>Next open: fill + costs</span></div>
<ul class="small">{"".join(f"<li>{x}</li>" for x in event_findings(p))}</ul>
{"".join(ev_tables)}
<details class="small"><summary>The cost settings</summary><ul>{costs}</ul></details>
</section>
</main>
<div id="tip" role="tooltip"></div>
<script>const Q={js};</script><script>{SCRIPT}</script></body></html>"""


CSS = """
:root{--calm:#199e70;--trending:#c98500;--crisis:#d55181;--s1:#3987e5;--s2:#d95926;--s3:#199e70;--s4:#c98500;--s5:#d55181}
*{box-sizing:border-box}body{margin:0;background:#000;color:#f5f5f7;font:15px/1.5 -apple-system,BlinkMacSystemFont,"Segoe UI",system-ui,sans-serif}
.wrap{max-width:1100px;margin:0 auto;padding:0 16px}main.wrap{padding-top:22px;padding-bottom:40px}h1{font-size:28px;margin:0 0 4px}
h2{font-size:24px;margin:36px 0 8px}h3{font-size:16px;margin:0 0 6px}.muted{color:#98989f}.small{font-size:13px}section{scroll-margin-top:60px}
.panel{border-radius:20px;padding:16px 18px;margin:12px 0}.stats{display:grid;grid-template-columns:repeat(auto-fit,minmax(170px,1fr));gap:8px;margin:12px 0}
.stat{border-radius:14px;padding:8px 10px}.stat .k{font-size:12px;color:#98989f}.stat .v{font-size:18px;font-weight:650}
.pos{color:#30d158}.neg{color:#ff453a}table{border-collapse:collapse;width:100%;font-size:13px}th,td{text-align:right;padding:6px 8px;border-bottom:1px solid rgba(255,255,255,.08);vertical-align:top}
th:first-child,td:first-child{text-align:left}.scroll{overflow-x:auto}th{font-weight:500;color:#98989f}
.tabs{display:flex;gap:8px;flex-wrap:wrap;margin:14px 0 0}.tabs a{color:#f5f5f7;padding:6px 12px;border:1px solid rgba(255,255,255,.14);border-radius:999px;font-size:13px;text-decoration:none}
.tabs a:hover{background:rgba(165,147,212,.2)}a{color:#a593d4}
.seg{display:flex;gap:6px;margin:8px 0}.seg button{font:inherit;font-size:13px;color:#f5f5f7;background:transparent;border:1px solid rgba(255,255,255,.18);border-radius:999px;padding:4px 12px;cursor:pointer}
.seg button[aria-pressed=true]{background:#a593d4;color:#000;border-color:#a593d4}
.surf canvas{width:100%;height:440px;display:block;cursor:grab;touch-action:none}.surf canvas:active{cursor:grabbing}
.grid2{display:grid;grid-template-columns:repeat(auto-fit,minmax(340px,1fr));gap:0 14px}.chart svg{width:100%;height:auto;display:block;overflow:visible}
.chart text{fill:#98989f;font-size:11px}.legend{display:flex;flex-wrap:wrap;gap:4px 14px;font-size:12px;color:#c7c7cc;margin-top:6px}
.sw{display:inline-block;width:10px;height:10px;border-radius:3px;margin-right:6px;vertical-align:baseline}.sw.lg{width:22px;height:22px;border-radius:6px;vertical-align:-2px;margin-right:10px}
.big{border-radius:20px;padding:16px 18px;margin:12px 0}.big .k{font-size:12px;color:#98989f;text-transform:uppercase;letter-spacing:.06em}.big .v{font-size:40px;font-weight:750;letter-spacing:-.02em}
table.rg td{white-space:nowrap}.flow{display:flex;flex-wrap:wrap;gap:6px;align-items:center;font-size:12px;color:#98989f;margin:10px 0}
.flow span{border:1px solid rgba(255,255,255,.16);border-radius:10px;padding:5px 9px;color:#f5f5f7}
#tip{position:fixed;pointer-events:none;background:rgba(20,16,30,.95);border:1px solid rgba(255,255,255,.16);border-radius:10px;padding:6px 9px;font-size:12px;
color:#f5f5f7;display:none;z-index:99;max-width:260px}details summary{cursor:pointer;color:#a593d4}
@media (max-width:640px){.surf canvas{height:320px}.big .v{font-size:32px}}
"""

SCRIPT = r"""
(()=>{
const tip=document.getElementById("tip"),NS="http://www.w3.org/2000/svg";
const showTip=(ev,html)=>{tip.innerHTML=html;tip.style.display="block";const w=tip.offsetWidth;tip.style.left=Math.min(ev.clientX+14,innerWidth-w-8)+"px";tip.style.top=(ev.clientY+14)+"px"};
const hideTip=()=>{tip.style.display="none"};
const el=(t,a={},p)=>{const n=document.createElementNS(NS,t);for(const k in a)n.setAttribute(k,a[k]);if(p)p.appendChild(n);return n};
const css=v=>getComputedStyle(document.documentElement).getPropertyValue(v).trim();
const pc=(v,d=1)=>v==null?"–":(v*100).toFixed(d)+"%";
// A line chart with a crosshair tooltip. series: [{name,color,pts:[[x,y]]}]; x numbers (or dates as ms when opt.date)
function lines(id,series,opt={}){const box=document.getElementById(id);if(!box)return;box.innerHTML="";
 const W=opt.w||520,H=opt.h||230,L=46,R=10,T=10,B=24,all=series.flatMap(s=>s.pts).filter(p=>p[1]!=null);if(!all.length){box.textContent="No data yet.";return}
 let x0=Math.min(...all.map(p=>p[0])),x1=Math.max(...all.map(p=>p[0])),y0=Math.min(...all.map(p=>p[1])),y1=Math.max(...all.map(p=>p[1]));
 if(opt.y0!=null)y0=Math.min(y0,opt.y0);const pad=(y1-y0)*.08||.01;y0-=pad;y1+=pad;if(x1===x0)x1=x0+1;
 const X=x=>L+(W-L-R)*(x-x0)/(x1-x0),Y=y=>T+(H-T-B)*(1-(y-y0)/(y1-y0));
 const s=el("svg",{viewBox:`0 0 ${W} ${H}`,role:"img","aria-label":opt.label||""},box);
 if(opt.bands)opt.bands(s,X,Y,T,H-B);
 for(let i=0;i<=4;i++){const v=y0+(y1-y0)*i/4;el("line",{x1:L,x2:W-R,y1:Y(v),y2:Y(v),stroke:"rgba(255,255,255,.07)"},s);
  el("text",{x:L-6,y:Y(v)+4,"text-anchor":"end"},s).textContent=opt.yf?opt.yf(v):v.toFixed(2)}
 for(const[x,a]of[[x0,"start"],[x1,"end"]])el("text",{x:X(x),y:H-6,"text-anchor":a},s).textContent=opt.xf?opt.xf(x):x;
 if(opt.ref!=null)el("line",{x1:L,x2:W-R,y1:Y(opt.ref),y2:Y(opt.ref),stroke:"rgba(255,255,255,.3)","stroke-dasharray":"4 3"},s);
 for(const sr of series){const p=sr.pts.filter(q=>q[1]!=null);if(!p.length)continue;
  el("path",{d:p.map((q,i)=>(i?"L":"M")+X(q[0]).toFixed(1)+","+Y(q[1]).toFixed(1)).join(""),fill:"none",stroke:sr.color,"stroke-width":2,"stroke-linejoin":"round","stroke-dasharray":sr.dash||""},s);
  if(opt.dots)p.forEach(q=>el("circle",{cx:X(q[0]),cy:Y(q[1]),r:3,fill:sr.color,stroke:"#000","stroke-width":1},s))}
 const cross=el("line",{y1:T,y2:H-B,stroke:"rgba(255,255,255,.35)",visibility:"hidden"},s),hit=el("rect",{x:L,y:T,width:W-L-R,height:H-T-B,fill:"transparent"},s);
 hit.addEventListener("pointermove",ev=>{const r=s.getBoundingClientRect(),xv=x0+((ev.clientX-r.left)*W/r.width-L)/(W-L-R)*(x1-x0);
  let rows=[],xs=null;for(const sr of series){let b=null;for(const q of sr.pts)if(q[1]!=null&&(b==null||Math.abs(q[0]-xv)<Math.abs(b[0]-xv)))b=q;
   if(b){xs=xs??b[0];rows.push(`<span class="sw" style="background:${sr.color}"></span>${sr.name}: <b>${opt.yf?opt.yf(b[1]):b[1]}</b>`)}}
  if(xs==null)return;cross.setAttribute("x1",X(xs));cross.setAttribute("x2",X(xs));cross.setAttribute("visibility","visible");
  showTip(ev,`<div class="muted">${opt.xf?opt.xf(xs):xs}</div>`+rows.join("<br>"))});
 hit.addEventListener("pointerleave",()=>{cross.setAttribute("visibility","hidden");hideTip()});
 if(series.length>1){const lg=document.createElement("div");lg.className="legend";
  lg.innerHTML=series.map(sr=>`<span><span class="sw" style="background:${sr.color}"></span>${sr.name}</span>`).join("");box.appendChild(lg)}}
const ms=d=>Date.parse(d),day=x=>new Date(x).toISOString().slice(0,10),mon=x=>new Date(x).toISOString().slice(0,7);

// ---------- 1. surface
const S=Q.surface,tickers=Object.keys(S);let cur=tickers[0];
const pick=document.getElementById("sv-pick"),cv=document.getElementById("sv-3d");
let yaw=-0.75,pitch=0.5;
// sequential ramp: deep violet (low vol) to pale lavender (high vol)
const ramp=t=>{const a=[42,24,88],b=[236,228,255],k=Math.max(0,Math.min(1,t));return`rgb(${a.map((v,i)=>Math.round(v+(b[i]-v)*k)).join(",")})`};
function draw3d(){if(!cv||!S[cur])return;const s=S[cur],g=s.grid,ex=s.expiries.filter(e=>e.iv.filter(v=>v!=null).length>=g.length-2),d=Math.min(devicePixelRatio||1,2),W=cv.clientWidth*d,H=cv.clientHeight*d;
 cv.width=W;cv.height=H;const x=cv.getContext("2d");x.clearRect(0,0,W,H);
 const vals=ex.flatMap(e=>e.iv).filter(v=>v!=null),lo=Math.min(...vals),hi=Math.max(...vals),dmax=Math.max(...ex.map(e=>e.dte));
 document.getElementById("sv-scale").textContent=`Colour runs from ${pc(lo)} (dark) to ${pc(hi)} (light).`;
 // world: u = moneyness in [-1,1], w = sqrt(dte) in [-1,1], z = vol in [0,1]
 const U=m=>(m-g[0])/(g[g.length-1]-g[0])*2-1,Wd=t=>Math.sqrt(t/dmax)*2-1,Z=v=>(v-lo)/((hi-lo)||1);
 const cy=Math.cos(yaw),sy=Math.sin(yaw),cp=Math.cos(pitch),sp=Math.sin(pitch),sc=Math.min(W*.3,H*.42);
 const P=(u,w,z)=>{const xr=u*cy-w*sy,yr=u*sy+w*cy,zz=(z-.35)*1.3;return[W/2+xr*sc,H*.5-(zz*cp-yr*sp)*sc,yr*cp+zz*sp]};
 // floor grid and axes
 x.lineWidth=d;x.strokeStyle="rgba(255,255,255,.12)";x.fillStyle="#98989f";x.font=`${11*d}px system-ui`;
 for(let i=0;i<=4;i++){const a=-1+i/2;let p=P(a,-1,0),q=P(a,1,0);x.beginPath();x.moveTo(p[0],p[1]);x.lineTo(q[0],q[1]);x.stroke();
  p=P(-1,a,0);q=P(1,a,0);x.beginPath();x.moveTo(p[0],p[1]);x.lineTo(q[0],q[1]);x.stroke()}
 for(const m of[g[0],1,g[g.length-1]]){const p=P(U(m),-1.15,0);x.textAlign="center";x.fillText(Math.round(m*100)+"%",p[0],p[1]+12*d)}
 for(const t of[ex[0].dte,Math.round(dmax/4),dmax]){const p=P(1.12,Wd(t),0);x.textAlign="left";x.fillText(t+"d",p[0],p[1])}
 const zp=P(-1,1,1);x.textAlign="right";x.fillText(pc(hi,0),zp[0]-4*d,zp[1]);const zq=P(-1,1,0);x.beginPath();x.moveTo(zq[0],zq[1]);x.lineTo(zp[0],zp[1]);x.stroke();
 // quads, far to near
 const quads=[];for(let j=0;j+1<ex.length;j++)for(let i=0;i+1<g.length;i++){const c=[[i,j],[i+1,j],[i+1,j+1],[i,j+1]].map(([a,b])=>[a,b,ex[b].iv[a]]);
  if(c.some(q=>q[2]==null))continue;const pts=c.map(([a,b,v])=>P(U(g[a]),Wd(ex[b].dte),Z(v)));
  quads.push({pts,z:pts.reduce((s,p)=>s+p[2],0)/4,v:c.reduce((s,q)=>s+q[2],0)/4})}
 quads.sort((a,b)=>b.z-a.z);
 for(const q of quads){x.beginPath();q.pts.forEach((p,i)=>i?x.lineTo(p[0],p[1]):x.moveTo(p[0],p[1]));x.closePath();
  x.fillStyle=ramp(Z(q.v)*.9+.05);x.fill();x.strokeStyle="rgba(0,0,0,.35)";x.lineWidth=d*.8;x.stroke()}}
let drag=null;if(cv){cv.addEventListener("pointerdown",ev=>{drag=[ev.clientX,ev.clientY,yaw,pitch];cv.setPointerCapture(ev.pointerId)});
 cv.addEventListener("pointermove",ev=>{if(!drag)return;yaw=drag[2]+(ev.clientX-drag[0])*.01;pitch=Math.max(.05,Math.min(1.4,drag[3]+(ev.clientY-drag[1])*.01));draw3d()});
 cv.addEventListener("pointerup",()=>drag=null);addEventListener("resize",draw3d)}
const SM=["--s1","--s2","--s3","--s4","--s5"];
function show(t){cur=t;const s=S[t];if(!s)return;[...pick.children].forEach(b=>b.setAttribute("aria-pressed",b.textContent===t));
 document.getElementById("sv-stats").innerHTML=[["Spot",s.spot.toFixed(2)],["At-the-money vol, 30 days",pc(s.atm_30d)],
  ["Skew, 90% − 110% strike",s.skew_30d==null?"–":(s.skew_30d*100).toFixed(1)+" pts"],["Options used",s.contracts.toLocaleString()],["Chain of",s.date]]
  .map(([k,v])=>`<div class="stat"><div class="k">${k}</div><div class="v">${v}</div></div>`).join("");
 document.getElementById("sv-gap").textContent=s.cboe_median_gap==null?"–":(s.cboe_median_gap*100).toFixed(2);
 draw3d();
 lines("sv-smile",Object.entries(s.smiles).map(([d,p],i)=>({name:d+" days",color:css(SM[i%5]),pts:p})),{xf:v=>Math.round(v*100)+"%",yf:v=>pc(v),label:"Implied vol by strike"});
 lines("sv-term",[{name:"ATM vol",color:css("--s1"),pts:s.term.map(r=>[r.dte,r.atm])}],{xf:v=>Math.round(v)+" days",yf:v=>pc(v),dots:true,label:"At-the-money vol by days to expiry"})}
if(pick&&tickers.length){pick.innerHTML=tickers.map(t=>`<button type="button">${t}</button>`).join("");[...pick.children].forEach(b=>b.onclick=()=>show(b.textContent));show(cur)}
const H=Q.options_history,hb=document.getElementById("sv-hist");
if(hb){const rows=["SPY","QQQ","IWM"].flatMap(t=>(H[t]||[]).slice(-5).map(r=>({t,...r}))).sort((a,b)=>b.date.localeCompare(a.date)||a.t.localeCompare(b.t));
 hb.innerHTML=rows.length?"<table><tr><th>Day</th><th>Ticker</th><th>Spot</th><th>ATM vol 30d</th><th>ATM vol 90d</th><th>Skew 30d</th><th>Put/call</th></tr>"+
 rows.map(r=>`<tr><td>${r.date}</td><td>${r.t}</td><td>${r.spot??"–"}</td><td>${pc(r.iv30_atm)}</td><td>${pc(r.iv90_atm)}</td><td>${r.skew30==null?"–":(r.skew30*100).toFixed(1)+" pts"}</td><td>${r.pc_oi??"–"}</td></tr>`).join("")+"</table>"
 :"<p class=muted>No days collected yet.</p>"}

// ---------- 2. regimes
const R=Q.regimes,col=n=>css("--"+n.toLowerCase());
(()=>{const box=document.getElementById("rg-scatter");if(!box)return;const W=520,H=260,L=46,Rr=10,T=10,B=28,pts=R.scatter;
 const xs=pts.map(p=>p[0]),ys=pts.map(p=>p[1]),x0=Math.max(Math.min(...xs),-3),x1=Math.min(Math.max(...xs),3),y0=0,y1=Math.min(Math.max(...ys),.9);
 const X=v=>L+(W-L-Rr)*(Math.max(x0,Math.min(x1,v))-x0)/(x1-x0),Y=v=>T+(H-T-B)*(1-(Math.min(v,y1)-y0)/(y1-y0));
 const s=el("svg",{viewBox:`0 0 ${W} ${H}`,role:"img","aria-label":"Each session's past-month return against volatility, coloured by regime"},box);
 for(let i=0;i<=4;i++){const v=y0+(y1-y0)*i/4;el("line",{x1:L,x2:W-Rr,y1:Y(v),y2:Y(v),stroke:"rgba(255,255,255,.07)"},s);el("text",{x:L-6,y:Y(v)+4,"text-anchor":"end"},s).textContent=pc(v,0)}
 el("line",{x1:X(0),x2:X(0),y1:T,y2:H-B,stroke:"rgba(255,255,255,.2)"},s);
 for(const v of[x0,0,x1])el("text",{x:X(v),y:H-12,"text-anchor":"middle"},s).textContent=pc(v,0);
 el("text",{x:(W+L)/2,y:H,"text-anchor":"middle"},s).textContent="past month's return, a year's pace →";
 el("text",{x:L,y:T-1},s).textContent="volatility";
 for(const p of pts){const c=el("circle",{cx:X(p[0]),cy:Y(p[1]),r:2.6,fill:col(p[2]),"fill-opacity":.75},s)}
 s.addEventListener("pointermove",ev=>{const r=s.getBoundingClientRect(),mx=(ev.clientX-r.left)*W/r.width,my=(ev.clientY-r.top)*H/r.height;let b=null,bd=1e9;
  for(const p of pts){const dd=(X(p[0])-mx)**2+(Y(p[1])-my)**2;if(dd<bd){bd=dd;b=p}}if(b&&bd<200)showTip(ev,`<span class="sw" style="background:${col(b[2])}"></span>${b[2]}<br>return ${pc(b[0])} a year · vol ${pc(b[1])}`);else hideTip()});
 s.addEventListener("pointerleave",hideTip);
 const lg=document.createElement("div");lg.className="legend";lg.innerHTML=["Calm","Trending","Crisis"].map(n=>`<span><span class="sw" style="background:${col(n)}"></span>${n}</span>`).join("");box.appendChild(lg)})();
const tl=R.timeline;
lines("rg-time",[{name:"SPY",color:"#f5f5f7",pts:tl.map(r=>[ms(r.date),r.spy])}],{xf:day,yf:v=>v.toFixed(0),label:"SPY price shaded by regime",
 bands:(s,X,Y,t,b)=>{let i=0;while(i<tl.length){let j=i;while(j+1<tl.length&&tl[j+1].regime===tl[i].regime)j++;
  const a=X(ms(tl[i].date)),z=X(ms(tl[Math.min(j+1,tl.length-1)].date));el("rect",{x:a,y:t,width:Math.max(z-a,.5),height:b-t,fill:col(tl[i].regime),"fill-opacity":.28},s);i=j+1}}});
(()=>{const box=document.getElementById("rg-time");if(!box)return;const lg=document.createElement("div");lg.className="legend";
 lg.innerHTML=["Calm","Trending","Crisis"].map(n=>`<span><span class="sw" style="background:${col(n)}"></span>${n}</span>`).join("");box.appendChild(lg)})();

// ---------- 3. event-driven
const EC=["--s1","--s2","--s3","--s4","--s5"];
for(const[k,v]of Object.entries(Q.events)){lines("ev-"+k,Object.entries(v.equity).map(([name,m],i)=>({name,color:css(EC[i%5]),dash:i?"":"5 3",pts:Object.entries(m).map(([p,x])=>[ms(p+"-01"),x])})),
 {w:1000,h:280,xf:mon,yf:v=>"×"+v.toFixed(2),ref:1,label:"Growth of 1 under each setting"})}
})();
"""
