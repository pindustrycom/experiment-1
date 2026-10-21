# Open-source GitHub bot strategies (2026-10-21)

Research only, paper only. Rules from popular open-source trading bot repos, ported by hand (nothing from those repos is
installed or run, no exchange keys), backtested from 2019 with costs against buy-and-hold.

| file | what |
|---|---|
| `strategies.py` | freqtrade-strategies (9), Jesse example-strategies (7), quant-trading Dual Thrust, with source links |
| `allocation.py` | Faber 10-month average, Antonacci dual momentum (GEM) |
| `engine.py` | one-asset backtester: next-open fills, freqtrade minimal_roi / stoploss / trailing, ATR stops, stats |
| `run.py` | everything on BTC/ETH (Binance, 10 bp a side) and SPY/QQQ (Yahoo + exp7-data, 2 bp a side) |
| `robust.py` | the 4h crypto Donchian rule across channels, filters, costs and other coins |

`.github/workflows/github-bots.yml` runs both on pull requests and on demand; results are in the job summary and artifact.

## Results (2019 to Sep/Oct 2026, after costs)
- All 9 freqtrade bots lose after Binance's 0.1% fee (Strategy001 BTC -27.9%/yr; gross +3.8%).
- Stock rules (golden cross, 10-month average, GEM, RSI(2), Turtle, MACD+EMA) trail holding SPY (+17.2%/yr) and QQQ (+22.6%/yr).
- 4 of 92 tests pass the edge test (net Sharpe above holding, alpha t > 2, positive in both halves): about chance.
- The one robust rule: Jesse's Donchian on 4h bars. BTC +52.2%/yr, Sharpe 1.28, worst drop -50% vs holding +49.5%, 0.97,
  -77%; trades last ~10 days, in the market 44% of the time; holds up at 25 bp costs and channel lengths 10 to 55.
  ETH +44.4% (0.98) vs +47.6% (0.89); BNB and SOL earn less than holding with smaller drops.
