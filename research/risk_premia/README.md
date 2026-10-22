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

## Results
Filled in from the Actions run.
