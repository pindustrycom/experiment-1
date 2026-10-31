"""Assemble the GitHub Pages site: an overview of every experiment at the root, one folder per experiment.

    python experiments/hub/build_hub.py --site _site --state state/portfolio.json [--update]

The root page is one column per group of experiments (GROUPS below): the market experiments (traded on paper, headlined
by the best strategy pot's money from portfolio.py, then the Top 3 pot: the three best strategies together) and the resonance research pages. Clicking a column opens that group
over the whole page (root/index.html#markets, ...). Each market experiment's page gets
its own 1000 CAD portfolio box at the top.

Expects each market experiment's pages already in place (_site/exp1/, _site/exp2/, _site/exp7/); a missing one is
shown as "not published yet" rather than failing the build. The resonance experiments are static pages kept in this
repository (experiments/exp4_resonance/, ...) and are copied in. It also:
  - puts the same navigation bar at the top of every page, so each page links to all the others (CATALOG below is
    the one list it is built from),
  - inlines the shared scripts the static pages load from ../hub/ (the published site encrypts every file that
    is not a page, so a page cannot load a script file),
  - keeps Exp1's old addresses working: every Exp1 page that used to live at the site root gets a
    redirect page there, and its data files are copied back to /data/ (files can't redirect).
Standard library only, so it runs in any job.
"""
from __future__ import annotations

import argparse
import html
import json
import re
import shutil
from datetime import UTC, datetime
from pathlib import Path

import portfolio
from portfolio import CAPITAL

e = html.escape
REPO_URL = "https://github.com/pindustrycom/experiment-1"
EXPERIMENTS = Path(__file__).resolve().parent.parent
EXP4 = EXPERIMENTS / "exp4_resonance"
HUB = Path(__file__).resolve().parent

MARKETS, RESONANCE = "Markets", "Resonance"
# Every experiment on the site, in menu order: (site folder, group, menu label, card title, card text, source folder of a
# static page, or None when another workflow builds the pages). The navigation bar and the overview page both come from it.
CATALOG = [
    ("exp1", MARKETS, "Exp1 pairs", "Exp1 · pairs", "Long the stock Exp1 favours in each pair, short the other.", None),
    ("exp2", MARKETS, "Exp2 volatility", "Exp2 · volatility-timed",
     "Sells SPY volatility when it is overpriced, holds SPY in calm markets, plus a stock book.", None),
    ("exp7", MARKETS, "Exp7 VWAP", "Exp7 · VWAP + 9 EMA day trades",
     "Day-trades SPY on a 3-minute chart: long when the 9 EMA is above the day's VWAP, short below, flat overnight.", None),
    ("strategies", MARKETS, "Strategy pots", "Strategy pots · trend, earnings, insiders, reversal, BTC",
     "Four rules with their own pots: trend following across ETFs, buying after earnings beats, buying after insiders "
     "buy, and buying the week's biggest losers.", None),
    ("quant", MARKETS, "Quant lab", "Quant lab · vol surface, regimes, event-driven test",
     "Three quant studies run each night on the pots' own data: SPY, QQQ and IWM implied volatility surfaces worked back "
     "from option prices, market regimes found by K-Means and how every pot does in each, and the pots re-run through an "
     "event-driven backtest with next-open fills and real costs. Research only, no pot.", None),
    ("btc_vol", MARKETS, "BTC volatility", "BTC volatility study · Kalman, GRU, Heston, rough vol, Deribit options",
     "Can modelling Bitcoin's volatility make money? Five rounds of walk-forward tests: forecasts help size the BTC "
     "breakout pot (now a live vol-sized pot) and time option selling, never predict direction.", EXPERIMENTS / "btc_vol"),
    ("trader_skill", MARKETS, "Copy-trading test", "trader-skill test · copying funds and Congress",
     "A backtest of the trader-skill Claude skills: copying 23 famous funds' 13F filings and every stock buy the House "
     "discloses, against SPY. Research only, no pot.", EXPERIMENTS / "trader_skill"),
    ("exp4", RESONANCE, "Exp4 playground", "Exp4 · resonance playground",
     "A 3D model of the resonance architecture by Joaquin Philco: retina and V1, time-based somas with resonators on the axon, "
     "concept structures and the rules that ask, allocate and sleep. Show it a digit in a colour, blur it, or open one cell.", EXP4),
    ("exp5", RESONANCE, "Exp5 vision", "Exp5 · vision stack",
     "Can better vision make Joaquin Philco's resonance agent see natural images? Eleven methods on CIFAR and Imagenette: Gabor and "
     "learned V1, somas with an axon resonance machine, a deeper somatic stack, feedback from memory and super labels.",
     EXPERIMENTS / "exp5_vision"),
    ("exp6", RESONANCE, "Exp6 alphabet", "Exp6 · alphabet and words",
     "The same resonance machinery learns letters, numbers and words from images, then words name colour patches with no "
     "stored labels. Includes a 3D view: watch the eye read a word, or a patch evoke its word.", EXPERIMENTS / "exp6_alphabet"),
    ("exp6_live", RESONANCE, "Exp6 live", "Exp6 live · the model in your browser",
     "The trained exp6 model runs in the page: type a word in any font, draw a character or pick a colour, and see which "
     "letter, word and naming structures resonate, with a 3D view driven by the computed values.", EXPERIMENTS / "exp6_live"),
    ("exp8", RESONANCE, "Exp8 neuron", "Exp8 · resonant neuron and volume",
     "Joaquin Philco's neuron: a sigmoid soma, one axon per connection with myelin resonators, and oligodendrocytes that set "
     "each weight. One neuron wired to two others runs live in 3D, then a 16 x 16 x 16 volume trained by his resonance-index sleep.",
     EXPERIMENTS / "exp8_neuron"),
    ("exp9", RESONANCE, "Exp9 reading", "Exp9 · reading from vision",
     "Joaquin Philco's memory cells taught like a child: letters by Montessori lessons and negative bonds, words read by "
     "eye movements, words bonded to colours and digits, then two-word phrases with a learned word order. Step through it in 3D.",
     EXPERIMENTS / "exp9_reading"),
    ("exp10", RESONANCE, "Exp10 words", "Exp10 · words playground",
     "The best reading model so far, frozen: build any COLOUR NUMBER phrase and watch the eye trace each letter in 3D, teach "
     "the model your own word bonds and see how it reads with them, and show it colours it has never seen. By Joaquin Philco.",
     EXPERIMENTS / "exp10_words"),
    ("exp11", RESONANCE, "Exp11 fear", "Exp11 · fear channel",
     "Joaquin Philco's emotion gate and his fear channel beyond the dendrite bandwidth: a visual search that learns what hurts, "
     "steers the eye by fear, and keeps fear on a separate electrical-synapse-like channel. A real run replayed in 3D: extinction, "
     "fear returning after a rest, and fast relearning.", EXPERIMENTS / "exp11_fear"),
    ("resonance_writeup", RESONANCE, "Write-up", "Write-up · resonant sensors, with controls",
     "A short paper on what the resonance experiments showed: as a sensor front end resonance beats fair low-pass controls on "
     "spoken digits, spoken words and eye movements; the learning rules built after it did not beat simple controls. By Joaquin Philco.",
     EXPERIMENTS / "resonance_writeup"),
]
# The overview page is one column per group, side by side; clicking a column opens that group over the whole page. A new
# project adds its group here and its experiments to CATALOG: (group, page anchor, colour, moving background drawn behind
# the column: "price" or "rings", one-line description).
GROUPS = [
    (MARKETS, "markets", "#e3ac5b", "price", "Strategies traded every day with fake money: 1,000 CAD each, no orders placed anywhere."),
    (RESONANCE, "resonance", "#a593d4", "rings", "Joaquin Philco's resonance architecture: models and results you can run in the browser."),
]
# Every paper pot on the Markets dashboard, split by how long it holds positions: (key in the portfolio table, label,
# category, page). The strategy pots take their names and descriptions from strategies/data/summary.json.
LONG, SHORT = "Long-term", "Short-term"
POTS = [("exp2", "Exp2 volatility", LONG, "exp2/index.html"), ("trend", "Trend following", LONG, "strategies/index.html#trend"),
        ("pead", "Earnings drift", LONG, "strategies/index.html#pead"), ("insider", "Insider buying", LONG, "strategies/index.html#insider"),
        ("best_ideas", "Best ideas", LONG, "strategies/index.html#best_ideas"),
        ("conviction", "Fund conviction", LONG, "strategies/index.html#conviction"),
        ("pelosi", "Pelosi copy", LONG, "strategies/index.html#pelosi"),
        ("exp1", "Exp1 pairs", SHORT, "exp1/index.html"), ("exp7", "Exp7 VWAP", SHORT, "exp7/index.html"),
        ("reversal", "Weekly reversal", SHORT, "strategies/index.html#reversal"),
        ("reversal_stop", "Weekly reversal, 5% stop", SHORT, "strategies/index.html#reversal_stop"),
        ("btc_donchian", "BTC breakout", SHORT, "strategies/index.html#btc_donchian"),
        ("btc_donchian_vol", "BTC breakout, vol-sized", SHORT, "strategies/index.html#btc_donchian_vol"),
        ("straddle8", "8% day straddles", SHORT, "strategies/index.html#straddle8")]
# (category, anchor, colour, moving background, description)
HORIZONS = [(LONG, "long", "#8fc1ad", "price", "Positions kept for weeks or months: monthly books, year-long trends, copied fund and Pelosi filings, and events held for a season."),
            (SHORT, "short", "#d9a15b", "candles", "Positions kept for a day or a week: daily pair bets, intraday SPY trades and weekly bounces.")]
POT_COLORS = ["#f5f5f7", "#a593d4", "#8fc1ad", "#d9a15b", "#7aa2ff", "#ff9f8a", "#c7c7cc", "#e6c86e", "#6fc7d9", "#d98fb8"]
# Static experiment pages: (site folder, source folder, card title, card text). No money involved in any of them.
STATIC = [(d, src, title, text) for d, _, _, title, text, src in CATALOG if src is not None]

NAV, NAV_END = "<!--labnav-->", "<!--/labnav-->"
NAV_CSS = """
:root{--site-nav-h:44px}
#labnav{box-sizing:border-box;height:var(--site-nav-h);margin:0;padding:0 16px;display:flex;flex-wrap:nowrap;align-items:center;gap:22px;overflow-x:auto;
overflow-y:hidden;white-space:nowrap;scrollbar-width:none;background:#000;border-bottom:1px solid rgba(255,255,255,.1);position:relative;z-index:50;
font:500 13px/1 -apple-system,BlinkMacSystemFont,"Segoe UI",system-ui,sans-serif;letter-spacing:0;text-transform:none;color-scheme:dark}
#labnav::-webkit-scrollbar{display:none}
#labnav a{color:#98989f;text-decoration:none;padding:15px 0 13px;border:0;border-bottom:2px solid transparent;border-radius:0;background:none;margin:0;font-weight:500}
#labnav a:hover{color:#f5f5f7;text-decoration:none}
#labnav a[aria-current=page]{color:#f5f5f7;border-bottom-color:#a593d4}
#labnav a:focus-visible{outline:2px solid #a593d4;outline-offset:2px;border-radius:2px}
#labnav .ln-home{color:#f5f5f7;font-weight:650;font-size:14px}
#labnav .ln-grp{display:flex;flex-direction:row;align-items:center;gap:16px;margin:0;padding:0}
#labnav .ln-g{color:#6e6e75;font-size:12px;padding-left:16px;border-left:1px solid rgba(255,255,255,.1)}
@media print{#labnav{display:none}}
"""
# On a narrow screen the bar scrolls sideways: start it scrolled to the current page.
NAV_SCRIPT = ('<script>(()=>{const n=document.getElementById("labnav"),a=n&&n.querySelector("[aria-current]");'
              'if(a&&a.offsetLeft+a.offsetWidth>n.clientWidth)n.scrollLeft=a.offsetLeft-16})()</script>')
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
HUB_CSS = """
section.group{margin:0 0 34px}section.group h2{font-size:20px;margin:0 0 2px;letter-spacing:-.01em}section.group .intro{margin:0 0 14px}
.card .go{font-weight:600}.card .tag{text-transform:none;letter-spacing:-.01em;font-size:16px;font-weight:650;color:var(--text)}
"""
# The overview page: the group columns, and a group opened over the whole page (the column whose anchor is in the address).
SPLIT_CSS = """
.sr{position:absolute;width:1px;height:1px;overflow:hidden;clip-path:inset(50%);white-space:nowrap}
.split{display:flex;min-height:calc(100vh - var(--site-nav-h));min-height:calc(100svh - var(--site-nav-h))}
.split>.half{flex:1 1 0;min-width:0;position:relative;overflow:hidden;border-right:1px solid var(--glass-line);scroll-margin-top:100vh;
transition:flex-grow .6s cubic-bezier(.2,.8,.2,1)}
.split>.half:last-child{border-right:0}.split:not(:has(.half:target))>.half:hover{flex-grow:1.35}
.half>canvas{position:absolute;inset:0;width:100%;height:100%;display:block;pointer-events:none}
.door{position:absolute;inset:0;display:flex;flex-direction:column;justify-content:flex-end;gap:10px;padding:32px clamp(20px,4vw,52px) 40px;
color:var(--text);background:linear-gradient(0deg,rgba(0,0,0,.8),rgba(0,0,0,0) 65%)}
.door:hover{text-decoration:none}.door:focus-visible{outline:2px solid var(--c);outline-offset:-6px}
.mono{font-family:ui-monospace,"SF Mono",Menlo,Consolas,monospace;font-variant-numeric:tabular-nums}
.door .k{font-size:12px;letter-spacing:.1em;text-transform:uppercase;color:var(--muted)}
.door h2{font-size:clamp(44px,6.5vw,84px);line-height:.95;letter-spacing:-.045em;font-weight:800;margin:0;color:var(--c);text-wrap:balance}
.door p{margin:0;max-width:40ch;color:var(--muted)}.door .v{font-size:26px;font-weight:500}
.door .exps{display:flex;flex-wrap:wrap;gap:4px 16px;font-size:13px;color:var(--text)}.door .cta{font-weight:600;color:var(--c)}
.open{display:none}
.split>.half:is(:target,:has(.half:target)){flex:1 0 100%;overflow:visible;border:0}
.split:has(>.half:is(:target,:has(.half:target)))>.half:not(:target):not(:has(.half:target)){display:none}
.half:is(:target,:has(.half:target))>canvas{position:fixed;opacity:.3}.half:has(.half:target)>canvas{display:none}
.half:is(:target,:has(.half:target))>.door{display:none}
.half:is(:target,:has(.half:target))>.open{display:block;position:relative;padding:22px 0 40px;animation:open .5s cubic-bezier(.2,.8,.2,1) both}
.half:has(.half:target)>.open>.wrap>:not(.split){display:none}.half:has(.half:target)>.open{padding:0}
.open .back{display:inline-block;font-size:13px;color:var(--muted);margin:0 0 14px}.open .back:hover{color:var(--text);text-decoration:none}
.open h2{font-size:clamp(34px,5vw,52px);letter-spacing:-.04em;font-weight:800;margin:0 0 4px;color:var(--c)}
.split.sub{min-height:440px;margin:26px 0 8px;border:1px solid var(--glass-line);border-radius:20px;overflow:hidden}
.split.sub:has(>.half:target){min-height:0;margin:0;border:0;border-radius:0;overflow:visible}
.sub .door h2{font-size:clamp(36px,5vw,64px)}.open .open .wrap{padding:0}
.mk-pots td,.mk-pots th{padding:7px 8px;text-align:right;white-space:nowrap}.mk-pots td:first-child,.mk-pots th:first-child,.mk-pots td:nth-child(2),.mk-pots th:nth-child(2),.mk-pots .cat{text-align:left}
.mk-pots{width:100%;border-collapse:collapse;font-size:14px}.mk-pots th{font-size:12px;color:var(--muted);font-weight:500}
.mk-pots tr+tr td{border-top:1px solid var(--line)}.mk-pots tr.lead td{font-weight:650}.mk-pots td:first-child{color:var(--muted)}
.big.best .name{font-size:clamp(20px,3vw,26px);font-weight:650;margin:2px 0 6px}.big.best .name a{color:var(--text)}.big.minor .v{font-size:28px}.big.best .d+.d{margin-top:4px}.door .lead{font-size:14px;color:var(--text)}.big.top3 .v{font-size:40px}.big.top3 .d+.d{margin-top:6px}.big.top3 .d a{color:var(--text)}.tag3{font-size:11px;font-weight:600;color:#a593d4;border:1px solid rgba(165,147,212,.5);border-radius:6px;padding:1px 6px;margin-left:6px;vertical-align:1px}.mk-pots .cat{font-size:12px;color:var(--muted)}.tw{overflow-x:auto}
@keyframes open{from{opacity:0;transform:translateY(16px)}}
@media (max-width:640px){.split{flex-direction:column}.split>.half{border-right:0;border-bottom:1px solid var(--glass-line);min-height:300px}
.split.sub>.half{min-height:400px}.mk-pots .cat{display:none}}
@media (prefers-reduced-motion:reduce){.split>.half{transition:none}.half .open{animation:none!important}}
"""
# Draws each column's moving background: only while the column is on screen and the tab is visible; one still frame when
# the reader asks for reduced motion.
SPLIT_SCRIPT = """<script>(()=>{
const still=matchMedia("(prefers-reduced-motion: reduce)").matches,cs=[...document.querySelectorAll("canvas[data-bg]")],seen=new Set();
let s=7,v=0;const walk=Array.from({length:700},()=>(v+=(s=(s*9301+49297)%233280)/233280-.485));
function price(x,W,H,d,c,t){const off=Math.floor(t/90)%500,vals=walk.slice(off,off+180),lo=Math.min(...vals),hi=Math.max(...vals);
x.strokeStyle="rgba(255,255,255,.06)";x.lineWidth=d;for(let i=1;i<8;i++){x.beginPath();x.moveTo(0,H*i/8);x.lineTo(W,H*i/8);x.stroke()}
const pt=i=>[i/(vals.length-1)*W,H*.08+(1-(vals[i]-lo)/(hi-lo||1))*H*.34];x.beginPath();vals.forEach((_,i)=>x.lineTo(...pt(i)));
x.strokeStyle=c;x.lineWidth=2*d;x.stroke();const end=pt(vals.length-1);x.lineTo(W,H);x.lineTo(0,H);x.closePath();
const g=x.createLinearGradient(0,0,0,H);g.addColorStop(0,c+"55");g.addColorStop(1,c+"00");x.fillStyle=g;x.fill();
x.beginPath();x.arc(end[0]-4*d,end[1],4*d,0,7);x.fillStyle=c;x.fill()}
function rings(x,W,H,d,c,t){const cx=W*.6,cy=H*.38,ph=t/1400,R=Math.hypot(W,H)*.75;x.strokeStyle=c;x.lineWidth=1.2*d;
for(let k=0;k<16;k++){x.globalAlpha=.5*(1-(k+ph%1)/16);x.beginPath();x.arc(cx,cy,(k+ph%1)/16*R,0,7);x.stroke()}
x.globalAlpha=.9;x.fillStyle=c;for(let k=0;k<5;k++){const a=k*1.3+ph*.4;x.beginPath();x.arc(cx+Math.cos(a)*W*.25,cy+Math.sin(a)*H*.2,3*d,0,7);x.fill()}
x.globalAlpha=1}
function candles(x,W,H,d,c,t){const n=42,cw=W/n,off=Math.floor(t/140),fr=(t%140)/140;x.lineWidth=d;let o=0;
for(let i=0;i<n+1;i++){const k=i+off,h=Math.sin(k*12.9898)*43758.5453,u=h-Math.floor(h),h2=Math.sin(k*78.233)*12345.678,u2=h2-Math.floor(h2),
up=u>.47,body=(.02+.06*u2)*H,cx=(i-fr)*cw,cy=H*.32+Math.sin(k*.21)*H*.08+(u-.5)*H*.05;
x.globalAlpha=.75;x.strokeStyle=x.fillStyle=up?c:"rgba(255,255,255,.35)";x.beginPath();x.moveTo(cx+cw/2,cy-body*.9);x.lineTo(cx+cw/2,cy+body*1.9);x.stroke();
x.fillRect(cx+cw*.2,cy,cw*.6,body)}x.globalAlpha=1}
let pending=0;const go=()=>{if(!pending)pending=requestAnimationFrame(t=>{pending=0;frame(t)})};
function frame(t){for(const c of cs){if(!still&&!seen.has(c))continue;const r=c.getBoundingClientRect(),d=Math.min(devicePixelRatio||1,2),
W=Math.round(r.width*d),H=Math.round(r.height*d);if(!W||!H)continue;if(c.width!==W||c.height!==H){c.width=W;c.height=H}
const x=c.getContext("2d");x.clearRect(0,0,W,H);({price,rings,candles}[c.dataset.bg]||price)(x,W,H,d,c.dataset.color,still?0:t)}if(!still)go()}
const io=new IntersectionObserver(es=>{es.forEach(x=>x.isIntersecting?seen.add(x.target):seen.delete(x.target));go()});cs.forEach(c=>io.observe(c));
addEventListener("visibilitychange",go);addEventListener("resize",go);addEventListener("hashchange",go);go()})()</script>"""
MARK, MARK_END = "<!--portfolio-->", "<!--/portfolio-->"
BANNER_CSS = """
.big,.pf{background:var(--panel);border:1px solid var(--line);border-radius:14px;padding:18px 20px;margin:0 0 16px}
.pf .big{border:0;padding:0;margin:0 0 8px;background:none}
.big .k{font-size:13px;color:var(--muted);text-transform:uppercase;letter-spacing:.04em}
.big .v{font-size:52px;font-weight:750;line-height:1.1;font-variant-numeric:tabular-nums}
.big .d{color:var(--muted);font-size:14px}.big .d b{color:var(--text)}.big .d b.pos{color:var(--lo,#2e7d4f)}.big .d b.neg{color:var(--hi,#c0392b)}
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


def total_return(v: float) -> str:
    """A pot's total return since it started, as a % of its CAPITAL."""
    return f"{100 * (v - CAPITAL) / CAPITAL:+.2f}%"


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
            f'<div class="d">Total return <b class="{pl_class(pl)}">{total_return(v)}</b> · worth <b>{cad(v)}</b> · today <span class="{pl_class(day)}">{cad(day, True)}</span>'
            f' · fake {CAPITAL:,.0f} CAD from {e(since)} · last update {e(rows[-1]["date"])}</div></div>')


def exp1_positions(t: dict) -> str:
    book = t.get("exp1_book") or []
    if not t.get("exp1_book_date"):
        return "No positions yet: the first book is formed after the next nightly run."
    if not book:
        return f"In cash since {e(t['exp1_book_date'])}: no pair has a call today."
    return f"Holding from the {e(t['exp1_book_date'])} close: " + "; ".join(
        (f"<b>{e(b['slug'])} in cash</b> <span class=\"muted\">({e(b['why'])})</span>" if b.get("stopped") else
         f"<b>long {e(b['long'])} / short {e(b['short'])}</b> <span class=\"muted\">({e(b['why'])})</span>") for b in book)


def portfolio_banner(t: dict | None, which: str, root: str) -> str:
    """The big P/L box put at the top of an experiment's page."""
    if not t:
        return ""
    name = {"exp1": "Exp1 paper portfolio", "exp2": "Exp2 paper portfolio", "exp7": "Exp7 paper portfolio"}[which]
    pos = exp1_positions(t) if which == "exp1" else e(t.get(f"{which}_position") or "No positions yet.")
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
    for d in ("exp1", "exp2", "exp7", "strategies", "quant"):
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
    since, _ = pot_since(rows, key)
    return (f'<div class="card"><div class="tag">{e(title)}</div><div class="v {pl_class(pl)}">{cad(pl, True)}</div>'
            f'<div class="small">Total return <b class="{pl_class(pl)}">{total_return(v)}</b>'
            f'<span class="muted"> since {e(since or rows[0]["date"])} · worth {cad(v)}</span></div><p>{what}</p><p class="small">{pos}</p>'
            f'<a class="go" href="{href}">Open →</a></div>')


def nav_bar(current: str, root: str, site: Path | None = None) -> str:
    """The bar at the top of every page. current is the page's site folder ("" for the overview); root leads back to the site root.
    With site given, experiments whose pages are not in it are left out."""
    def link(d: str, label: str) -> str:
        cur = ' aria-current="page"' if d == current else ""
        return f'<a href="{root}{d}/index.html"{cur}>{e(label)}</a>'
    groups = []
    for g, anchor, *_ in GROUPS:
        links = "".join(link(d, label) for d, grp, label, *_ in CATALOG
                        if grp == g and (site is None or (site / d / "index.html").exists()))
        if links:
            groups.append(f'<span class="ln-grp"><a class="ln-g" href="{root}index.html#{anchor}">{g}</a>{links}</span>')
    home = ' aria-current="page"' if current == "" else ""
    return (f'{NAV}<style>{NAV_CSS}</style><nav id="labnav" aria-label="Experiments">'
            f'<a class="ln-home" href="{root}index.html"{home}>All experiments</a>{"".join(groups)}</nav>{NAV_SCRIPT}{NAV_END}')


BODY = re.compile(r"<body\b[^>]*>", re.IGNORECASE)


def add_nav(site: Path) -> int:
    """Put the navigation bar at the top of every page in the site (replacing any earlier copy). Redirect pages are left alone."""
    n = 0
    for f in site.rglob("*.html"):
        s = f.read_text(encoding="utf-8")
        if 'http-equiv="refresh"' in s:
            continue
        if NAV in s:
            s = s[:s.index(NAV)] + s[s.index(NAV_END) + len(NAV_END):]
        m = BODY.search(s)
        if not m:
            continue
        rel = f.relative_to(site).parts
        current = rel[0] if len(rel) > 1 else ""
        f.write_text(s[:m.end()] + nav_bar(current, "../" * (len(rel) - 1), site) + s[m.end():], encoding="utf-8")
        n += 1
    return n


def half(group: str, anchor: str, color: str, bg: str, intro: str, labels: list[str], stat: str, body: str,
         unit: str = "experiment", back: tuple[str, str] = ("#", "All groups")) -> str:
    """One group's column on the overview page, and the full page it opens into. The Markets page holds two more of these
    (long-term and short-term pots), which open over the whole page the same way."""
    count = f'{len(labels)} {unit}{"s" * (len(labels) != 1)}'
    return (f'<section class="half" id="{anchor}" style="--c:{color}"><canvas data-bg="{bg}" data-color="{color}" aria-hidden="true"></canvas>'
            f'<a class="door" href="#{anchor}"><span class="k mono">{count}</span><h2>{e(group)}</h2><p>{e(intro)}</p>{stat}'
            f'<span class="exps">{"".join(f"<span>{e(x)}</span>" for x in labels)}</span><span class="cta">Open {e(group)} →</span></a>'
            f'<div class="open"><div class="wrap"><a class="back" href="{back[0]}">← {e(back[1])}</a><h2>{e(group)}</h2>'
            f'<p class="muted small intro">{e(intro)}</p>{body}</div></div></section>')


def pots_in(t: dict) -> list[tuple[str, str, str, str, str, str]]:
    """The pots the table has, as (key, label, category, page, what, position)."""
    info = {d: text for d, _, _, _, text, _ in CATALOG}
    strat = t.get("strategies") or {}
    last = t["rows"][-1]
    out = []
    for k, label, cat, href in POTS:
        if k not in last or (k == "exp7" and not t.get("exp7")):
            continue
        if k in strat:
            what, pos = e(strat[k].get("what", "")), e(strat[k].get("position", ""))
        elif k == "exp1":
            what, pos = e(info["exp1"]), exp1_positions(t)
        elif k in info:
            what, pos = e(info[k]), e(t.get(f"{k}_position") or "")
        else:
            continue
        out.append((k, label, cat, href, what, pos))
    return out


def total_number(rows: list[dict], keys: list[str], label: str, since: str | None, minor: bool = False) -> str:
    """The big P/L number for a set of pots, each started with CAPITAL (minor: shown smaller, under a headline)."""
    cap = CAPITAL * len(keys)
    v = sum(rows[-1][k] for k in keys)
    day = v - sum(rows[-2][k] for k in keys) if len(rows) > 1 else 0.0
    pl = v - cap
    return (f'<div class="big{" minor" * minor}"><div class="k">{e(label)}</div><div class="v {pl_class(pl)}">{cad(pl, True)}</div>'
            f'<div class="d">Total return <b class="{pl_class(pl)}">{100 * pl / cap:+.2f}%</b> · worth <b>{cad(v)}</b> of {len(keys)} × {CAPITAL:,.0f} CAD · today '
            f'<span class="{pl_class(day)}">{cad(day, True)}</span> · since {e(since or rows[0]["date"])} · last update {e(rows[-1]["date"])}</div></div>')


def pots_chart(rows: list[dict], pots: list[tuple]) -> str:
    keys = [(k, POT_COLORS[i % len(POT_COLORS)]) for i, (k, *_) in enumerate(pots)]
    legend = " ".join(f'<span style="color:{c}">━</span> {e(p[1])}' for (k, c), p in zip(keys, pots))
    return (f'<div class="panel">{value_chart(rows, keys)}<div class="small muted">{legend} · dashed line: the starting '
            f'{CAPITAL:,.0f} CAD</div></div>')


def ranked(rows: list[dict], pots: list[tuple]) -> list[tuple]:
    """The pots from the best % return on their 1,000 CAD to the worst (every pot starts with the same amount)."""
    return sorted(pots, key=lambda p: -rows[-1][p[0]])


def pot_since(rows: list[dict], k: str) -> tuple[str | None, int]:
    """(the close a pot started from, trading days it has moved since), or (None, 0) before its first trade."""
    for i, r in enumerate(rows):
        if abs(r[k] - CAPITAL) > 1e-9:
            return rows[i - 1]["date"], len(rows) - i
    return None, 0


def best_pot(rows: list[dict], pots: list[tuple]) -> str:
    """The headline of the Markets page: the money in the pot with the best % return so far, and the runner-up.
    It always names the top pot, even when every pot is below its start."""
    order = ranked(rows, pots)
    if not order:
        return ""
    k, label, cat, href, *_ = order[0]
    v = rows[-1][k]
    pl = v - CAPITAL
    day = v - rows[-2][k] if len(rows) > 1 else 0.0
    since, days = pot_since(rows, k)
    nxt = ""
    if len(order) > 1:
        k2, label2, _, href2, *_ = order[1]
        pl2 = rows[-1][k2] - CAPITAL
        nxt = f' · runner-up <a href="{href2}">{e(label2)}</a> <span class="{pl_class(pl2)}">{total_return(rows[-1][k2])}</span>'
    notes = []
    if days < 60:
        notes.append(f'Only {days} trading day{"s" * (days != 1)} so far: a lead this young is mostly luck.')
    return (f'<div class="big best"><div class="k">Best strategy so far · {e(cat)}</div>'
            f'<div class="v {pl_class(pl)}">{cad(pl, True)}</div>'
            f'<div class="name"><a href="{href}">{e(label)}</a> <span class="{pl_class(pl)}">{total_return(v)}</span></div>'
            f'<div class="d">Total return <b class="{pl_class(pl)}">{total_return(v)}</b> · worth <b>{cad(v)}</b> from {CAPITAL:,.0f} CAD on {e(since or rows[0]["date"])} · today '
            f'<span class="{pl_class(day)}">{cad(day, True)}</span>{nxt} · last update {e(rows[-1]["date"])}</div>'
            + "".join(f'<div class="d small">{n}</div>' for n in notes) + "</div>")


def top3_box(rows: list[dict], pots: list[tuple], keys: list[str]) -> str:
    """The concentrated pot: 1,000 CAD split evenly across the three best strategies so far, and each of their own pots."""
    if not keys or "top3" not in rows[-1]:
        return ""
    info = {p[0]: p for p in pots}
    v = rows[-1]["top3"]
    pl, day = v - CAPITAL, v - rows[-2]["top3"] if len(rows) > 1 else 0.0
    names = [info[k][1] if k in info else k for k in keys]
    each = []
    for k in keys:
        if k not in info:
            continue
        _, label, _, href, *_ = info[k]
        p = rows[-1][k] - CAPITAL
        each.append(f'<a href="{href}">{e(label)}</a> <span class="mono {pl_class(p)}">{total_return(rows[-1][k])} ({cad(rows[-1][k])})</span>')
    return (f'<div class="big top3"><div class="k">Top 3 strategies together · concentrated pot</div>'
            f'<div class="v {pl_class(pl)}">{cad(pl, True)}</div>'
            f'<div class="d">Total return <b class="{pl_class(pl)}">{total_return(v)}</b> · worth <b>{cad(v)}</b> from {CAPITAL:,.0f} CAD · today '
            f'<span class="{pl_class(day)}">{cad(day, True)}</span> · last update {e(rows[-1]["date"])}</div>'
            f'<div class="d">Equal money in {e(", ".join(names[:-1]))} and {e(names[-1])}, evened out every Friday. '
            f'Each one on its own 1,000 CAD: {" · ".join(each)}</div>'
            f'<div class="d small">Picked on 2020–2025 replays: together +28.8% a year vs SPY +15.0%; on Jan–Oct 2026, '
            f'kept apart as a test, +27.4% a year vs SPY +17.7%. Past replays, not a promise.</div></div>')


def pots_table(rows: list[dict], pots: list[tuple], top: list[str] | None = None) -> str:
    out = ['<div class="panel"><div class="tw"><table class="mk-pots"><tr><th>#</th><th>Pot</th><th>Total return</th><th class="cat">Holds for</th>'
           '<th>Worth</th><th>Profit / loss</th><th>Today</th></tr>']
    for n, (k, label, cat, href, *_) in enumerate(ranked(rows, pots), 1):
        v = rows[-1][k]
        pl, day = v - CAPITAL, v - rows[-2][k] if len(rows) > 1 else 0.0
        tag = ' <span class="tag3">Top 3</span>' if k in (top or []) else ""
        out.append(f'<tr{" class=lead" * (n == 1 and pl_class(pl) == "pos")}><td class="mono">{n}</td><td><a href="{href}">{e(label)}</a>{tag}</td>'
                   f'<td class="mono {pl_class(pl)}">{total_return(v)}</td>'
                   f'<td class="cat">{e(cat)}</td><td class="mono">{cad(v)}</td>'
                   f'<td class="mono {pl_class(pl)}">{cad(pl, True)}</td>'
                   f'<td class="mono {pl_class(day)}">{cad(day, True)}</td></tr>')
    return "".join(out) + "</table></div></div>"


def markets_body(t: dict) -> tuple[str, str]:
    """The Markets page: every pot with its own 1,000 CAD, then the long-term and short-term columns. Returns (body, door stat)."""
    rows, pots = t["rows"], pots_in(t)
    keys = [p[0] for p in pots]
    fx = "" if t.get("fx_known") else " USD/CAD rate unavailable: values exclude currency moves."
    top = t.get("top3") or []
    head = (best_pot(rows, pots) + top3_box(rows, pots, top) + pots_table(rows, pots, top)
            + pots_chart(rows, pots)
            + f'<p class="small muted">Every pot started with a fake {CAPITAL:,.0f} CAD and trades every trading day from its own rule; US positions, '
            f'valued in CAD at the Bank of Canada rate. All pots count from the same start as Exp1; a strategy added later is replayed from that day with the data known each day.{fx}</p>')
    halves = []
    for cat, anchor, color, bg, intro in HORIZONS:
        mine = [p for p in pots if p[2] == cat]
        if not mine:
            continue
        ks = [p[0] for p in mine]
        pl = sum(rows[-1][k] for k in ks) - CAPITAL * len(ks)
        stat = f'<span class="v mono {pl_class(pl)}">{cad(pl, True)}</span>'
        body = (total_number(rows, ks, f"{cat} pots: profit / loss", t.get("start")) + pots_chart(rows, mine)
                + '<div class="cards">' + "".join(sleeve_card(t, k, label, href, what, pos) for k, label, _, href, what, pos in mine) + "</div>")
        halves.append(half(cat, anchor, color, bg, intro, [p[1] for p in mine], stat, body, unit="pot", back=("#markets", "Markets")))
    total = sum(rows[-1][k] for k in keys) - CAPITAL * len(keys)
    stat = f'<span class="lead">All pots together: <span class="mono {pl_class(total)}">{cad(total, True)}</span></span>'
    t3 = rows[-1].get("top3", CAPITAL) - CAPITAL
    for k, label, *_ in ranked(rows, pots)[:1]:
        pl = rows[-1][k] - CAPITAL
        stat = (f'<span class="k mono">Best strategy</span><span class="v mono {pl_class(pl)}">{cad(pl, True)}</span>'
                f'<span class="lead"><b>{e(label)}</b> <span class="mono {pl_class(pl)}">{total_return(rows[-1][k])}</span>'
                + (f' · top 3 together <span class="mono {pl_class(t3)}">{cad(t3, True)}</span>' if top else
                   f' · all pots together <span class="mono {pl_class(total)}">{cad(total, True)}</span>') + '</span>')
    return head + f'<div class="split sub">{"".join(halves)}</div>', stat


def plain_card(d: str, title: str, text: str, published: bool) -> str:
    go = f'<a class="go" href="{d}/index.html">Open {e(title.split(" · ")[0])}</a>' if published else '<p class="muted small">Not published yet.</p>'
    return f'<div class="card"><div class="tag">{e(title)}</div><p>{e(text)}</p>{go}</div>'


def hub_page(site: Path, t: dict | None = None) -> str:
    built = datetime.now(UTC).strftime("%Y-%m-%d %H:%M UTC")
    have = lambda d: (site / d / "index.html").exists()  # noqa: E731
    info = {d: (title, text) for d, _, _, title, text, _ in CATALOG}
    if t is None:
        money = '<p class="muted">The paper portfolio appears after its first nightly run.</p>'
        cards = "".join(plain_card(d, *info[d], have(d)) for d in ("exp1", "exp2", "exp7"))
        body, stat = {MARKETS: money + f'<div class="cards">{cards}</div>'}, {MARKETS: ""}
    else:
        mb, ms = markets_body(t)
        body, stat = {MARKETS: mb}, {MARKETS: ms}
    halves = []
    for g, anchor, color, bg, intro in GROUPS:
        shown = [(d, label) for d, grp, label, *_ in CATALOG if grp == g and have(d)]
        if g not in body:
            if not shown:
                continue
            body[g] = '<div class="cards">' + "".join(plain_card(d, *info[d], True) for d, _ in shown) + "</div>"
        halves.append(half(g, anchor, color, bg, intro, [label for _, label in shown], stat.get(g, ""), body[g]))
    return f"""<!doctype html><html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1"><meta name="robots" content="noindex,nofollow">
<meta name="referrer" content="no-referrer"><title>All experiments</title><style>{CSS}{BANNER_CSS}{GLASS_CSS}{HUB_CSS}{SPLIT_CSS}</style></head><body>
<main><h1 class="sr">All experiments</h1><div class="split">{"".join(halves)}</div></main>
<footer><div class="wrap">Fake money: no orders are placed anywhere and nothing here is investment advice. Built {built}.</div></footer>
{SPLIT_SCRIPT}</body></html>"""


def redirect_page(target: str) -> str:
    t = e(target)
    return (f'<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="robots" content="noindex">'
            f'<meta http-equiv="refresh" content="0; url={t}"><link rel="canonical" href="{t}"><title>Moved</title></head>'
            f'<body><p>This page moved to <a href="{t}">{t}</a>.</p></body></html>')


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


SHARED = re.compile(r'<script src="\.\./hub/([\w-]+\.js)"></script>')


def inline_shared(page: str) -> str:
    """Replace <script src="../hub/x.js"></script> with the script itself: the published site cannot serve script files."""
    return SHARED.sub(lambda m: "<script>\n" + (HUB / m.group(1)).read_text(encoding="utf-8").replace("</script", "<\\/script") + "</script>", page)


def add_static(site: Path, name: str, src: Path) -> bool:
    """Copy one static experiment page (folder with an index.html) into the site."""
    if not (src / "index.html").exists():
        return False
    shutil.copytree(src, site / name, dirs_exist_ok=True)
    for f in (site / name).rglob("*.html"):
        f.write_text(inline_shared(f.read_text(encoding="utf-8")), encoding="utf-8")
    return True


def add_exp4(site: Path, src: Path = EXP4) -> bool:
    """Copy Experiment 4's static page into the site."""
    return add_static(site, "exp4", src)


def build(site: Path, state: Path | None = None, update: bool = False) -> dict:
    site.mkdir(parents=True, exist_ok=True)
    for name, src, *_ in STATIC:
        add_static(site, name, src)
    moved = keep_old_addresses(site) if (site / "exp1").is_dir() else 0
    t, rep = None, {}
    if state is not None:
        r = portfolio.run(site, state, update)
        t, rep = r["table"], r["portfolio"]
        inject(site / "exp1" / "index.html", portfolio_banner(t, "exp1", "../"))
        inject(site / "exp2" / "index.html", portfolio_banner(t, "exp2", "../"))
        inject(site / "exp7" / "index.html", portfolio_banner(t, "exp7", "../"))
    theme(site)
    (site / "index.html").write_text(hub_page(site, t))
    navs = add_nav(site)
    (site / ".nojekyll").write_text("")
    (site / "robots.txt").write_text("User-agent: *\nDisallow: /\n")
    return {"pages_with_nav": navs, "old_addresses_kept": moved, "exp2": (site / "exp2" / "index.html").exists(), **rep}


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--site", type=Path, default=Path("_site"))
    ap.add_argument("--state", type=Path, help="paper portfolio state file (portfolio.json); omit to skip the portfolio")
    ap.add_argument("--update", action="store_true", help="move Experiment 1's book forward (the nightly full run only)")
    a = ap.parse_args()
    print(json.dumps(build(a.site, a.state, a.update), default=str))
