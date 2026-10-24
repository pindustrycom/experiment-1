"""Weekly reversal proxy on Exp1's 21 stocks: buy the 5 worst of the week on Friday at 12:30 vs at the close."""
import sys, numpy as np, pandas as pd
sys.argv = [sys.argv[0], sys.argv[1], sys.argv[2], '12:30']
exec(open('scripts/midday.py').read().split('# re-fit hedge')[0])
fri = [d for d in C.index if pd.Timestamp(d).weekday() == 4 and '2019-01-04' <= d <= '2026-10-09']
res = {'close': [], 'mid': []}
for i in range(1, len(fri) - 1):
    f0, f1, f2 = fri[i - 1], fri[i], fri[i + 1]
    w = (C.loc[f1] / C.loc[f0] - 1).dropna(); k = w.nsmallest(5).index
    res['close'].append((C.loc[f2, k] / C.loc[f1, k] - 1).mean())
    m = M.loc[f1]
    if m.isna().any(): res['mid'].append(res['close'][-1]); continue
    wm = (m / C.loc[f0] - 1).dropna(); km = wm.nsmallest(5).index
    res['mid'].append((C.loc[f2, km] / m[km] - 1).mean())
for k, r in res.items():
    r = np.array(r); print(k, f'mean/week {r.mean():+.3%}  ann {(1 + r).prod() ** (52 / len(r)) - 1:+.1%}  Sharpe {r.mean() / r.std() * np.sqrt(52):.2f}  weeks {len(r)}')
d = np.array(res['mid']) - np.array(res['close']); print('mid - close per week', f'{d.mean():+.3%} t={d.mean() / d.std() * np.sqrt(len(d)):.2f}')
