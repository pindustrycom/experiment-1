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

## Results (2019-01-02 to 2026-10-20, after costs; full table in `results.md`)
Holding SPY: +17.2%/yr, Sharpe 0.93, worst drop -34%. 47 rows tested; 2 pass the edge test (net Sharpe above SPY,
alpha t-stat above 2, positive in both 2019-22 and 2023-26):

| strategy | CAGR | Sharpe | worst drop | alpha t | 2019-22 | 2023-26 | next-close Sharpe |
|---|---|---|---|---|---|---|---|
| 7.2 Vol carry: short VIXY, long VIXM hedge | +17.2% | 1.06 | -19% | 2.7 | +32.2% | +3.2% | 0.95 |
| 8.2 FX carry, G7 currencies | +6.7% | 1.05 | -8% | 2.8 | +3.8% | +9.8% | 0.68 |

Both are weaker than they look. Vol carry made almost everything in 2019-22 and +3%/yr since. FX carry uses deposit
rates implied by the CurrencyShares funds (FRED blocks GitHub runners), which lag rate moves by up to a year, and its
Sharpe drops to 0.68 when trading a day later. Dollar carry (+4.8%, t 2.7) and momentum + carry (+5.5%, t 2.6) have
real-looking alpha but Sharpe below SPY.

Other notes:
- Best raw return: 3.1 momentum, long the top S&P 100 decile, +28.1%/yr (Sharpe 1.06) but beta 1.12, alpha t 1.5, and
  today's S&P 100 list flatters it. The long/short version: +6.3%, Sharpe 0.35.
- 3.12 200-day average on SPY (long or cash): +12.6%/yr, Sharpe 1.02, worst drop -19% vs -34%. Less return, smoother.
- 19.5 pre-FOMC drift: +3.0%/yr while invested 8 days a year (Sharpe 0.86, t 2.0); gone if you buy a day late.
- Losers: daily stock mean reversion (-18%/yr, 730x turnover; -5% even before costs), low-vol long/short (-28%/yr),
  alpha combo, commodity value, weekly futures contrarian, channel rule.
- Six markets: no technical rule beats simply holding all six (Sharpe 1.31, +19%/yr). Best rule is the 10/50/200
  three-average one (all six Sharpe 0.99, +11%/yr); it helped long bonds (0.68 vs -0.07 holding) and the euro, and
  hurt everywhere else.
