"""Run the range-trading grid on SPY/QQQ 1-minute bars (exp7-data release: Tiingo IEX, 2019-01 to 2026-10).
Usage: python run_etf.py DATA_DIR OUT_JSON"""
import glob
import json
import os
import sys

import pandas as pd

sys.path.insert(0, os.path.dirname(__file__))
from rangebt import run, stats


def load(folder, minutes):
    d = pd.concat(pd.read_csv(f) for f in sorted(glob.glob(f'{folder}/*.csv')))
    d['ts'] = pd.to_datetime(d['date'], utc=True).dt.tz_convert('America/New_York')
    d = d.drop_duplicates('ts').set_index('ts').sort_index().between_time('09:30', '15:59')
    if minutes > 1:
        d = d.groupby(pd.Grouper(freq=f'{minutes}min', origin='start_day', offset='30min')).agg(
            open=('open', 'first'), high=('high', 'max'), low=('low', 'min'), close=('close', 'last'),
            volume=('volume', 'sum')).dropna()
    d['day'] = d.index.strftime('%Y-%m-%d')
    dead = d.volume.eq(0).groupby(d.day).transform('mean') > 0.5
    return d[~dead]


def main(data, out):
    res = {}
    for sym in ('SPY', 'QQQ'):
        for tf in (5, 15, 60):
            d = load(f'{data}/{sym}', tf)
            days = sorted(d.day.unique())
            intraday = tf < 60
            g = d.groupby('day')
            oc = (g.close.last() / g.open.first() - 1)
            res[f'{sym} {tf}m open-to-close hold'] = stats([(pd.Timestamp(x), 1, r, 0) for x, r in oc.items()], 252)
            for setup in ('fade', 'deviation', 'breakout'):
                for n in (24, 48):
                    for tgt in (('eq', 'edge') if setup != 'breakout' else ('eq',)):
                        for cost in (0.0, 0.3, 1.0):
                            tr = run(d, setup, n=n, tgt=tgt, cost_bp=cost, intraday=intraday,
                                     max_hold=None if intraday else 35)
                            key = f'{sym} {tf}m {setup} n={n} tgt={tgt} cost={cost}'
                            res[key] = stats(tr, 252, days)
                            print(key, {k: v for k, v in res[key].items() if k != 'by_year'}, flush=True)
    with open(out, 'w') as f:
        json.dump(res, f, indent=1)


if __name__ == '__main__':
    main(sys.argv[1], sys.argv[2])
