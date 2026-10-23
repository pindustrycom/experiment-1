# Daily stop losses on every paper pot (2026-10-23)

Joaquin asked for a stop loss on every pot ("if the stock meets the stop loss, we can just sell, then begin the strategy the
next day") and a cap on what the weekly reversal pot can lose in a day. No real money anywhere.

## The rule

| Pot | Checked on | Stop | After a stop |
|---|---|---|---|
| Weekly reversal (plain) | none: kept without a stop to compare (Joaquin, 2026-10-23) | | |
| Weekly reversal, 5% stop (new pot `reversal_stop`, replayed from 2026-10-16) | each stock's daily low (Yahoo) | 5% below the previous close; 20 stocks at 5% each, so this caps the pot's day near -5% | sold at the stop (or the open, on a gap), bought back at that close if the rule still holds it |
| Earnings drift, insider buying, best ideas, fund conviction, Pelosi copy | each stock's daily low | 8% | same |
| Trend following (ETFs, long and short) | daily low (high for shorts) | 4% | same |
| BTC breakout | 4-hour bar lows | 6% below BTC at the last US close (or the entry price) | cash until the next US close, then the rule again |
| Exp7 (SPY day trades) | 1-minute bars | the day down 1% | flat for the rest of the day; back next morning |
| Exp1 (pairs) | closes only | a pair slice down 5% in a session | that pair sits in cash for the next session |
| Exp2 (combined portfolio) | closes only | the pot down 2% in a session | cash for the next session |

Exp1 and Exp2 only have closing prices, so their stop can't cut a day short: it only keeps them out of the next session.
Code: experiments/strategies/strats/stops.py (+ crypto.py), experiments/exp7_vwap/exp7/strategy.py, experiments/hub/portfolio.py.
Past rows are not rewritten; the stops apply from the next run on.

## Backtest

Exp1 and Exp2, 2019 to 2026-10-15, USD, from research/portfolio_2019 (close-checked, out one session):

| Rule | Exp1 | Exp2 |
|---|---|---|
| No stop | -0.3%/yr, worst drop -22% | +17.9%/yr, worst drop -12% |
| Pair 3% / pot 1.5% | -1.8%/yr, -25% | +17.3%/yr, -13% (22 stops) |
| Pair 5% / pot 2% (chosen) | -0.6%/yr, -20% | +18.1%/yr, -11% (8 stops) |
| Pair 8% / pot 3% | -0.5%/yr, -22% | +18.1%/yr, -11% (2 stops) |

Neither stop changes much: close-checked stops can't stop a bad day, as the 2026-10-17 intraday study already found
(no stop beat holding for any pot).

Weekly reversal and trend, 2019 to 2026-10-23, Yahoo daily bars, 1 bp a side (strategies.yml trial run on PR #31):

| Rule | % a year | Worst day | Worst drop | Days below -5% | Stops | 2025 to now |
|---|---|---|---|---|---|---|
| Reversal, no stop | +28.1 | -24.0% | -67% | 23 | 0 | +56% |
| Reversal 3% | +19.7 | -17.4% | -56% | 9 | 7,742 | +37% |
| **Reversal 5% (chosen)** | +22.4 | -17.4% | -57% | 11 | 3,098 | +52% |
| Reversal 8% | +27.3 | -17.5% | -57% | 20 | 1,013 | +54% |
| Reversal 10% | +28.3 | -17.9% | -58% | 21 | 565 | +54% |
| Trend, no stop | +2.8 | -2.2% | -8.2% | 0 | 0 | +14% |
| **Trend 4% (chosen)** | +2.8 | -2.1% | -7.9% | 0 | 137 | +14% |

The 5% cap halves the reversal pot's days below -5% and its worst day goes from -24% to -17%. It costs about 6 points a
year: these stocks often dip 5% during the day and come back, and the stop sells the dip. The worst days left are
overnight gaps (a stock opening 15% down sells at the open), which no daily stop can prevent. An 8% stop costs under
1 point a year but cuts few bad days. The reversal replay uses today's S&P 500 members, which flatters it.

BTC: only the cached 4-hour bars from 2026-06-25 exist (+6.1%, worst day -3.6%); a 6% stop never fired there, 4% would
have cost 1 point. Exp7: see the `SPY 3m flip stop…` rows in exp7's backtest results.

What moved on the bad day (2026-10-22, SPY -1.2%), from the same trial run:
- Reversal: COHR -12.5%, LRCX -11.8%, LITE -11.5%, FLEX -11.1%, CIEN -10.7% (optical and chip stocks, last week's losers
  fell again). With the 5% stops it would have lost -4.4% instead of -7.0%.
- Pelosi: VST -9.8%, BE -8.7%, INTC -8.0%, TEM -5.8%, AMZN -4.6%.
- 2026-10-23 (SPY -0.5%): reversal +0.7%, Pelosi -0.8% (BE -11.2%), the rest within -0.6%.
