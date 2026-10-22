# 151 Trading Strategies (Kakushadze & Serur 2018)

Research only, paper only. Zura Kakushadze and Juan Andrés Serur, "151 Trading Strategies" (SSRN 3247865) describes
strategies across every asset class. This folder backtests the ones free daily data can test, from 2019 to the last
close, with costs, against holding SPY.

| file | what |
|---|---|
| `strategies.py` | each testable strategy as a target-weight builder, named by the paper's section |
| `engine.py` | weight backtester: positions drift between rebalances, costs on turnover, borrow fee on shorts, idle cash earns T-bills |
| `data.py` | Yahoo daily bars (dividends included), FRED 3-month interest rates |
| `run.py` | runs everything, writes `out/table.md`, `out/results.json`, `out/daily_returns.csv` |

`.github/workflows/kakushadze151.yml` runs it on pull requests and on demand (the sandbox has no market data access).

## Assumptions
- Trades at the close the signal is computed on (a market-on-close order); the table also shows the Sharpe ratio when
  trading one day later.
- Costs a side: 2 bp stocks and ETFs, 5 bp leveraged and VIX funds, 1 bp FX, 10 bp bitcoin. Borrow a year on shorts:
  0.5% stocks and ETFs, 5% leveraged and VIX funds and bitcoin.
- Stocks: today's S&P 100 members, so the stock tests carry survivorship bias (the long-only ones are compared with
  an equal-weight portfolio of the same stocks, which shares the bias).
- Futures and commodities through ETFs (GLD, USO, DBA, TLT...), which include roll costs. FX: Yahoo spot plus the
  OECD 3-month interbank rate of each currency (FRED, monthly) so carry is counted. Bitcoin and FX days are folded
  onto NYSE trading days.

## Not tested (data we do not have)
- Options, all of chapter 2 (covered calls, spreads, straddles...) and 3.5 implied volatility, 6.2 dispersion, 7.4
  skew risk reversal, 7.5 variance swaps: need options-chain history. (Exp2 already runs a synthetic variance-swap
  sleeve.)
- 3.2 earnings momentum, 3.3 value, 3.6 multifactor, 3.17 M&A arbitrage: need point-in-time fundamentals, estimates
  and deal lists.
- 3.11 weighted-regression mean reversion and 3.19 optimised stat arb: variants of 3.10 that need a risk model; 3.20
  market making and 6.3 index-ETF intraday arbitrage: need tick and quote data.
- Fixed income (chapter 5: bullets, barbells, ladders, curve trades, swap spreads...): need bond prices and curves.
- 6.1 cash-and-carry, 9.1 roll yield, 9.2 hedging pressure, 10.1-10.2 futures hedging and calendar spreads: need
  individual futures contracts (and COT data).
- 9.6 commodity pricing models, 11 structured assets, 12 convertibles, 13 tax arbitrage, 14 weather / energy /
  inflation, 15 distressed, 16 real estate, 17 cash, 20 infrastructure, 21 tax efficiency, 18.3 crypto sentiment,
  19.2-19.4 macro fundamentals: no data, or not a trading rule.

## Results
Filled in from the Actions run.
