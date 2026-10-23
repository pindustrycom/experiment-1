# Daily stop losses on every paper pot (2026-10-23)

Joaquin asked for a stop loss on every pot ("if the stock meets the stop loss, we can just sell, then begin the strategy the
next day") and a cap on what the weekly reversal pot can lose in a day. No real money anywhere.

## The rule

| Pot | Checked on | Stop | After a stop |
|---|---|---|---|
| Weekly reversal | each stock's daily low (Yahoo) | 5% below the previous close; 20 stocks at 5% each, so this caps the pot's day near -5% | sold at the stop (or the open, on a gap), bought back at that close if the rule still holds it |
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

Strategy pots, BTC and Exp7: run on real data by the pull request's trial runs (strategies.yml prints "Stop backtest";
exp7.yml adds "SPY 3m flip stop…" rows to its grid). Results below.
