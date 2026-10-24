"""Exp1 proxy: close-only daily rebalance vs close + midday rebalance, 2019-01..2026-10, with 10-min bars."""
import json, glob, sys
from pathlib import Path
from zoneinfo import ZoneInfo
import numpy as np, pandas as pd
NY = ZoneInfo('America/New_York')
BARS, SIG = sys.argv[1], sys.argv[2]
MID = sys.argv[3] if len(sys.argv) > 3 else '12:30'
COST = 0.0001  # per side
sig = pd.read_csv(SIG)
sig = sig[sig.date >= '2018-01-01']

def intraday(t):
    rows = []
    for f in sorted(glob.glob(f'{BARS}/{t}_*.json')):
        rows += json.loads(Path(f).read_text())
    d = pd.DataFrame(rows)
    ts = pd.to_datetime(d['date'], utc=True).dt.tz_convert(NY) + pd.Timedelta(minutes=10)
    d = d.assign(day=ts.dt.strftime('%Y-%m-%d'), hm=ts.dt.strftime('%H:%M'))
    d = d[(d.hm > '09:30') & (d.hm <= '16:00')]
    p = d.pivot_table(index='day', columns='hm', values='close')
    mid = p.loc[:, [c for c in p.columns if c <= MID]].ffill(axis=1).iloc[:, -1]
    return mid / p.loc[:, [c for c in p.columns if c <= '16:00']].ffill(axis=1).iloc[:, -1]  # mid / IEX close same day

tick = sorted(set(sig.a) | set(sig.b))
close = {}
for t in tick:
    s = pd.concat([sig[sig.a == t][['date', 'pa']].rename(columns={'pa': 'p'}), sig[sig.b == t][['date', 'pb']].rename(columns={'pb': 'p'})])
    close[t] = s.drop_duplicates('date').set_index('date')['p'].sort_index()
C = pd.DataFrame(close).sort_index()
# midday price = official close * (IEX mid / IEX close), so midday is consistent with the daily series
MR = pd.DataFrame({t: intraday(t) for t in tick})
M = C * MR.reindex(C.index)
M[(M / C - 1).abs() > 0.15] = np.nan

# re-fit hedge (formation 252, gap 63) per pair per day, as in out_of_sample_zscores
pairs = sig[['slug', 'a', 'b']].drop_duplicates('slug').values.tolist()
lc = np.log(C)
fits = {}
for slug, a, b in pairs:
    la, lb = lc[a], lc[b]
    for i in range(315, len(C)):
        x, y = lb.iloc[i - 315:i - 63 + 1].to_numpy(), la.iloc[i - 315:i - 63 + 1].to_numpy()
        # formation = 252 days ending gap days before the last close
        ok = ~(np.isnan(x) | np.isnan(y))
        if ok.sum() < 200: continue
        X = np.column_stack([np.ones(ok.sum()), x[ok]])
        coef, *_ = np.linalg.lstsq(X, y[ok], rcond=None)
        r = y[ok] - X @ coef
        fits[(slug, C.index[i])] = (coef[0], coef[1], r.mean(), r.std(ddof=1))
dirs = sig.set_index(['date', 'slug'])['direction'].to_dict()
zc = sig.set_index(['date', 'slug'])['z'].to_dict()

def zof(slug, a, b, d_fit, pa, pb):
    f = fits.get((slug, d_fit))
    if f is None or not f[3]: return None
    return (np.log(pa) - (f[0] + f[1] * np.log(pb)) - f[2]) / f[3]

def book(d_sig, z_by_slug):
    out = {}
    for slug, a, b in pairs:
        dr = dirs.get((d_sig, slug)); z = z_by_slug.get(slug)
        if dr in ('a_advantaged', 'b_advantaged'):
            out[slug] = (a, b) if dr == 'a_advantaged' else (b, a)
        elif z is not None and not np.isnan(z) and abs(z) >= 2:
            out[slug] = (b, a) if z > 0 else (a, b)
    return out

def ret(bk, P0, P1):
    if not bk: return 0.0
    r = [(P1[l] / P0[l] - 1) - (P1[s] / P0[s] - 1) for l, s in bk.values()]
    r = [x for x in r if not np.isnan(x)]
    return float(np.mean(r)) / 2 * 2 if r else 0.0   # each pair gets equal money, long and short legs

def turnover(b0, b1):
    """fraction of gross traded when moving from book b0 to b1 (legs equal-weighted over pairs held)."""
    w = lambda bk: {k: v for slug, (l, s) in bk.items() for k, v in ((l + '|' + slug, 1 / len(bk)), (s + '|' + slug, -1 / len(bk)))} if bk else {}
    w0, w1 = w(b0), w(b1)
    return sum(abs(w1.get(k, 0) - w0.get(k, 0)) for k in set(w0) | set(w1))

days = [d for d in C.index if '2019-01-02' <= d <= '2026-10-15']
rows = []
prevA = prevB = None
for i, d in enumerate(days[1:], 1):
    d0 = days[i - 1]
    # A: book from close d0 held to close d
    zA = {slug: zc.get((d0, slug)) for slug, _, _ in pairs}
    bA = book(d0, zA)
    rA = ret(bA, C.loc[d0], C.loc[d])
    tA = turnover(prevA or {}, bA); prevA = bA
    # B: same close book to midday, then midday book to close
    m = M.loc[d]
    if m.isna().any():
        rB, tB = rA, tA; prevB = bA; nomid = 1
    else:
        tB = turnover(prevB or {}, bA)
        r1 = ret(bA, C.loc[d0], m)
        zM = {}
        for slug, a, b in pairs:
            f = fits.get((slug, d0)); z0 = zc.get((d0, slug))
            if f is None or z0 is None or np.isnan(z0) or not f[3]: zM[slug] = z0; continue
            zM[slug] = z0 + (np.log(m[a] / C.loc[d0, a]) - f[1] * np.log(m[b] / C.loc[d0, b])) / f[3]
        bM = book(d0, zM)
        tB += turnover(bA, bM)
        r2 = ret(bM, m, C.loc[d])
        rB = (1 + r1) * (1 + r2) - 1; prevB = bM; nomid = 0
    rows.append(dict(date=d, rA=rA, rB=rB, tA=tA, tB=tB, nomid=nomid))
R = pd.DataFrame(rows).set_index('date')
def st(r):
    i = (1 + r).cumprod(); yrs = len(r) / 252
    return f'CAGR {i.iloc[-1] ** (1 / yrs) - 1:+.2%}  Sharpe {r.mean() / r.std() * np.sqrt(252):.2f}  maxDD {(i / i.cummax() - 1).min():.1%}'
print('days', len(R), 'days without midday bars', int(R.nomid.sum()))
for c in ('A', 'B'):
    g = R['r' + c]; n = g - COST * R['t' + c]
    print(c, 'gross', st(g), '| net 1bp', st(n), '| turnover/day', round(R['t' + c].mean(), 3))
for y in range(2019, 2027):
    k = R.index.str.startswith(str(y))
    print(y, f"A {(1 + R.rA[k] - COST * R.tA[k]).prod() - 1:+.1%}  B {(1 + R.rB[k] - COST * R.tB[k]).prod() - 1:+.1%}")
# sanity: A vs the z in signals
