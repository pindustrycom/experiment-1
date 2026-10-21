# Copy-trader backtests

Research only, paper only. Tests the five copy ideas from the "Copying known traders" doc (2026-10-21) against simply
holding SPY. Every rule enters at the first close after the information was public and pays 5 bp a side.

| idea | script | data |
|---|---|---|
| Best ideas: each fund's biggest position | `backtest.py` | trader-skill cache: 13F top 25 of 23 famous funds since 2013 |
| Conviction + consensus: stocks at 7.5%+ of a long-term fund | `backtest.py` | same |
| Pelosi pot: her disclosed buys, held to a disclosed sale or a year | `backtest.py` | trader-skill cache: House PTRs since 2019 |
| Activist 13D: buy after a known activist's original 13D | `activist13d.py` | SEC EDGAR master index, Yahoo prices |
| Activist short reports | not tested | no free list of reports exists |

`.github/workflows/copy-traders.yml` runs both scripts on pull requests and on demand and keeps the outputs as the run's
artifact. Known biases: the 23 funds and Pelosi were picked because they are famous today; the 13F prices and the 13D
ticker list miss companies that were later delisted.
