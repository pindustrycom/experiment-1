# Risk premia study (2026-10-22)

Research only, paper only. After the 151 Trading Strategies tests (`../kakushadze151`) found almost nothing that
beats holding SPY, this asks the question the evidence supports better: do the returns investors are paid for
taking risks others avoid still pay, and does a diversified mix of them beat SPY?

| file | what |
|---|---|
| `premia_data.py` | Ken French daily factors (5 factors + momentum), Cboe PUT / BXM index history (Cboe CDN, Yahoo fallback) |
| `premia.py` | Part A: each premium by decade and before / after its paper. Part B: sleeves and portfolios since 2012 and 2019 |

Reuses the backtester, data loader and FX carry / VIX carry / trend rules from `../kakushadze151`.
`.github/workflows/risk-premia.yml` runs it on pull requests and on demand.

## Assumptions
- French factors are paper long/short portfolios of all US stocks; they pay a cost haircut (1%/yr, momentum 2%/yr)
  and are only tradeable approximately (factor ETFs). Put-write and buy-write indexes pay 0.5%/yr.
- Portfolio weights use only the trailing year, set at each month-end close; leverage is financed at the T-bill rate
  (no spread), so the levered risk-parity rows are slightly flattered.
- French data ends about a month before today, so the portfolio windows stop there.

## Results (Actions run 39716937909; French data to 2026-08-31)

**A. Which premia still pay?** Sharpe of each long/short premium after the cost haircut:

| premium | 1990-99 | 2000-09 | 2010-18 | 2019-26 | before its paper | after its paper |
|---|---|---|---|---|---|---|
| Market (stocks minus cash) | 0.95 | -0.03 | 0.78 | 0.76 | | |
| Size (small minus big, paper 1981) | -0.48 | 0.52 | -0.10 | -0.24 | 0.71 | -0.11 |
| Value (HML, 1992) | -0.11 | 0.62 | -0.34 | -0.03 | 0.72 | 0.12 |
| Profitability (RMW, 2015) | 0.33 | 0.77 | -0.01 | 0.12 | 0.39 | 0.12 |
| Investment (CMA, 2015) | -0.10 | 0.70 | -0.12 | -0.13 | 0.46 | -0.15 |
| Momentum (UMD, 1993) | 1.48 | 0.01 | 0.28 | 0.05 | 0.97 | 0.19 |
| Put-write over cash (2007 on) | | 0.06 | 0.60 | 0.53 | | |
| Buy-write over cash (2002 on) | | 0.13 | 0.53 | 0.45 | | |

The famous stock factors made almost nothing after 2009 and lost about 70-130% of their Sharpe after publication.
Option selling earns about the market's Sharpe but is 0.9 correlated with it: it is a calmer way to hold stocks,
not a separate source of return.

**B. Does a mix beat SPY?** (2019-01-02 to 2026-08-31; 2012 on is similar)

| portfolio | CAGR | vol | Sharpe | worst drop |
|---|---|---|---|---|
| Equal weight, all 14 sleeves | +6.7% | 4.5% | 1.46 | -8% |
| Inverse vol, all sleeves | +5.4% | 3.8% | 1.41 | -7% |
| Risk parity at 10% vol (up to 2x) | +8.1% | 7.6% | 1.07 | -15% |
| 70% SPY + 30% inverse-vol premia | +13.9% | 13.8% | 1.01 | -25% |
| 60/40 SPY/IEF | +10.9% | 11.5% | 0.96 | -21% |
| Hold SPY | +17.5% | 19.3% | 0.93 | -34% |

Since 2012: equal weight +4.2%/yr Sharpe 1.06, 70/30 +11.6% Sharpe 0.98, SPY +15.1% Sharpe 0.94.

The sleeves that earned their keep since 2019: VIX carry (Sharpe 1.11), FX carry (1.05), gold (0.99), all with
correlation to SPY under 0.15. Value, size, investment, momentum, long bonds and trend were near zero.

Takeaways: a diversified premia mix gives a smoother ride (Sharpe 1.0-1.5, drops under 10%) but much less return;
levering it up to SPY's risk did not keep the Sharpe (0.95) because the trailing-year risk estimates lagged. Adding
30% of the mix to SPY trimmed the worst drop from -34% to -25% and cost about 3.6 points a year. Nothing here makes
more money than holding SPY in a bull market; it buys steadier returns.
