# Range trading (Camille van Merrienboer / "Trading Wizard" video, 2026-10-19)

## What he publishes
Nothing testable. His site (camillevm.com), YouTube channel (@camillevanmerrienboer), Skool profile and Whop page describe
him as a crypto day trader since 2017 and promote two free Skool groups ("Millionaire Trading", about 8k members, and
"The Market Wizards"), both private behind a sign-up. No page states an entry, stop, target, timeframe or market rule.
The video Joaquin pasted is motivational: the only method content is "a range is predefined horizontal support and
resistance", scalping on lower timeframes and market structure. The client result quoted (700 to 21,000 dollars in
65 trades, 61% wins) has no rules, dates or market attached. We did not sign up.

## What was tested instead
The textbook horizontal-range method crypto range traders describe (rangebt.py):
- range = highest high and lowest low of the last N bars (N = 24 or 48), equilibrium EQ = middle; only when the range
  is sideways (Kaufman efficiency ratio of those closes below 0.3).
- fade: buy in the bottom 15% of the range, sell in the top 15%, stop 15% of the width beyond the edge, target EQ or the
  opposite edge.
- deviation: a close outside the range, then a close back inside within 6 bars; trade back into the range, stop beyond
  the deviation's extreme, target EQ or the opposite edge (the "sweep and reclaim" setup).
- breakout (control): follow a close outside the range, stop at EQ, target one width further.
Next-bar-open fills, stops and targets on bar highs/lows (stop wins ties), 1x notional, no leverage.

Data: SPY/QQQ Tiingo IEX 1-minute bars 2019-01 to 2026-10 (exp7-data release), regular hours, flat at the close,
5/15/60-minute bars, 0 / 0.3 / 1 bp per side (results_etf.json). Bitcoin/Ether Binance spot 5m/15m/1h bars 2019-01 to
2026-09, 24/7, closed after 4x the lookback if nothing hits, 0 / 2 / 5 bp per side (results_crypto.json, run by
.github/workflows/range-trading.yml).

## Results (selected; full grid in the JSON files)
| Test | Trades | Win % | Avg trade (bp) | Annual | Sharpe | Worst drop |
|---|---|---|---|---|---|---|
| SPY 15m fade, N=48, EQ, 0.3 bp | 2,247 | 39.9 | -1.6 | -4.9% | -0.56 | -40% |
| SPY 15m deviation, N=24, edge, 0.3 bp | 836 | 38.6 | +1.7 | +1.7% | 0.32 | -11% |
| QQQ 15m breakout, N=48, 0.3 bp | 1,225 | 56.2 | +4.9 | +7.6% | 0.81 | -16% |
| SPY / QQQ open-to-close hold | | | | +6.2% / +8.2% | 0.49 / 0.52 | -15% / -25% |
| BTC 15m fade, N=48, EQ, 5 bp | 9,217 | 35.7 | -11.5 | -76% | -3.6 | -100% |
| BTC 15m deviation, N=48, edge, 0 bp | 2,841 | 29.5 | +7.0 | +23% | 0.83 | -38% |
| BTC 15m deviation, N=48, edge, 2 bp | 2,841 | 29.4 | +3.0 | +6.5% | 0.35 | -54% |
| ETH 15m deviation, N=24, edge, 2 bp | 3,963 | 33.6 | +3.9 | +15.7% | 0.60 | -61% |
| ETH 1h deviation, N=48, EQ, 2 bp | 793 | 46.7 | +19.2 | +17.0% | 0.70 | -54% |
| ETH 1h deviation, N=48, EQ, 5 bp | 793 | 46.7 | +13.2 | +10.0% | 0.48 | -56% |
| BTC / ETH buy and hold | | | | +49% / +47% | 0.96 / 0.88 | -77% / -79% |

Best crypto rows by year at 2 bp: ETH 15m deviation edge +69 / +101 / +41% in 2020-22, then -16 / +0.5 / -26 / -4% in
2023-26. BTC 15m deviation edge +79 / +83% in 2020-21, -41% in 2022, -4 / -30% in 2025-26.

## Reading
- Fading the edges loses everywhere, before and after costs. Prices that sit at a range edge break out more often than
  the fade's stop allows. On 5-minute crypto bars with a realistic 5 bp taker fee it goes to zero.
- Deviation-and-reclaim is the one setup with a gross edge, and only on crypto (3-8 bp a trade on 15m, 13-23 bp on 1h).
  It needs maker-level fees (about 2 bp) to survive, and nearly all of it came in 2019-2022; 2023-2026 is flat to
  negative. On SPY/QQQ it is roughly zero.
- No version wins 61% of trades with a positive result; the setups that make money win 30-47% and rely on bigger winners.
- Nothing here beats simply holding: the ETFs open-to-close, or BTC/ETH round the clock.
- Not proposed as an experiment. If tracked on paper, the candidate would be ETH 1h deviation (N=48, EQ target, 2 bp).
