"""Run the range-trading grid on Bitcoin and Ether, the market the video is about.
Bars: Binance spot klines from data.binance.vision (public monthly files), 2019-01 to last month, 24/7.
Positions may run overnight; a trade that hits neither stop nor target is closed after 4x the lookback.
Usage: python run_crypto.py CACHE_DIR OUT_JSON"""
import io
import json
import os
import sys
import urllib.request
import zipfile

import pandas as pd

sys.path.insert(0, os.path.dirname(__file__))
from rangebt import run, stats

URL = 'https://data.binance.vision/data/spot/monthly/klines/{s}/{tf}/{s}-{tf}-{m}.zip'


def load(cache, sym, tf):
    months = pd.period_range('2019-01', pd.Timestamp.now('UTC').tz_localize(None).to_period('M') - 1, freq='M').strftime('%Y-%m')
    frames = []
    for m in months:
        f = f'{cache}/{sym}-{tf}-{m}.csv'
        if not os.path.exists(f):
            raw = urllib.request.urlopen(URL.format(s=sym, tf=tf, m=m), timeout=60).read()
            zipfile.ZipFile(io.BytesIO(raw)).extract(f'{sym}-{tf}-{m}.csv', cache)
        frames.append(pd.read_csv(f, header=None, usecols=range(6), names=['t', 'open', 'high', 'low', 'close', 'volume']))
    d = pd.concat(frames)
    # Binance switched its spot files from milliseconds to microseconds in 2025
    d['t'] = pd.to_datetime(d.t.where(d.t > 1e14, d.t * 1000), unit='us', utc=True)
    d = d.drop_duplicates('t').set_index('t').sort_index()
    d['day'] = d.index.strftime('%Y-%m-%d')
    return d


def main(cache, out):
    os.makedirs(cache, exist_ok=True)
    res = {}
    for sym in ('BTCUSDT', 'ETHUSDT'):
        for tf in ('5m', '15m', '1h'):
            d = load(cache, sym, tf)
            days = sorted(d.day.unique())
            g = d.groupby('day').close.last()
            hold = g.pct_change().dropna()
            res[f'{sym} buy and hold'] = stats([(pd.Timestamp(x), 1, r, 0) for x, r in hold.items()], 365)
            for setup in ('fade', 'deviation', 'breakout'):
                for n in (24, 48):
                    for tgt in (('eq', 'edge') if setup != 'breakout' else ('eq',)):
                        for cost in (0.0, 2.0, 5.0):
                            tr = run(d, setup, n=n, tgt=tgt, cost_bp=cost, intraday=False, max_hold=4 * n)
                            key = f'{sym} {tf} {setup} n={n} tgt={tgt} cost={cost}'
                            res[key] = stats(tr, 365, days)
                            print(key, {k: v for k, v in res[key].items() if k != 'by_year'}, flush=True)
    with open(out, 'w') as f:
        json.dump(res, f, indent=1, default=float)


if __name__ == '__main__':
    main(sys.argv[1], sys.argv[2])
