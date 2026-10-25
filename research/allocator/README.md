# The allocator: one pot that splits its money across all the others by market regime

Joaquin, 2026-10-25. Paper research, 2020-01-02 to 2026-10-15 (2019 is used only to learn), USD, after each pot's own costs plus 2 bp per unit moved between pots.

## Every strategy and experiment

| Pot / experiment | Status | Daily history used here |
|---|---|---|
| Exp1 pairs | live pot | 2019 replay (fundamentals only, no news history) |
| Exp2 volatility-timed mix | live pot | 2019 replay of the frozen model (an upper bound: synthetic variance swap) |
| Exp7 VWAP + 9 EMA day trades | live pot | rebuilt from the SPY 1-minute cache, 0.3 bp, 1% day stop |
| Trend following | live pot | pots replay |
| Weekly reversal | live pot | pots replay (today's S&P 500 members: flatters it) |
| Weekly reversal, 5% stop | live pot | not here: needs daily lows not in this container (close twin of reversal) |
| Best ideas, Fund conviction, Pelosi copy | live pots | pots replay from 13F / PTR history |
| BTC breakout | live pot | not here: no BTC 4-hour bars in this container |
| Earnings drift, Insider buying | live pots | no history (tracked forward only) |
| SPY | benchmark, also an option for the allocator | |
| Exp3 picker, Exp4-10 resonance, trader-skill, range trading, 151 strategies, GitHub bots, Quant lab | research only, no pot | not tradeable inputs |

## How the allocator decides
At every week's last close, and whenever the regime changes:
1. Regime known at that close: K-Means (Calm / Trending / Crisis) on SPY's past month of return and volatility, fitted only on data up to the previous year end (SPY since 1998).
2. Each pot's past return per unit of variance in that same regime, from all earlier days (its whole record until it has 40 days in the regime).
3. Money to the pots with a positive score, in proportion to it, at most 40% each; nothing to pots that lost in this regime; cash if none qualifies.

## Results, 2020-01-02 to 2026-10-15 (1000 at the start)

| | Per year | Volatility | Sharpe | Worst drop | 1000 became |
|---|---|---|---|---|---|
| Exp2 alone | +17.0% | 7.4% | 2.17 | -12.3% | 2,901 |
| Allocator, hindsight (look-ahead, not tradeable) | +17.5% | 9.8% | 1.69 | -15.0% | 2,976 |
| **Allocator (regimes)** | **+14.5%** | **9.9%** | **1.42** | **-16.6%** | **2,502** |
| Allocator without regimes | +14.7% | 10.7% | 1.34 | -18.5% | 2,531 |
| Equal weight, all 8 pots | +17.2% | 13.7% | 1.22 | -24.4% | 2,920 |
| Pelosi copy | +35.4% | 31.0% | 1.13 | -41.1% | 7,769 |
| Best ideas | +24.7% | 23.2% | 1.07 | -32.0% | 4,460 |
| Fund conviction | +18.3% | 21.4% | 0.89 | -35.7% | 3,119 |
| Weekly reversal | +26.8% | 36.8% | 0.83 | -67.1% | 5,003 |
| SPY | +15.4% | 20.0% | 0.82 | -33.7% | 2,637 |
| Exp7 VWAP | +5.3% | 10.4% | 0.54 | -13.1% | 1,414 |
| Trend following | +2.2% | 4.8% | 0.47 | -7.5% | 1,158 |
| Exp1 pairs | -0.2% | 9.2% | 0.03 | -22.0% | 990 |

By year (%):

| | 2020 | 2021 | 2022 | 2023 | 2024 | 2025 | 2026 |
|---|---|---|---|---|---|---|---|
| Allocator (regimes) | +19.8 | +21.9 | -6.4 | +24.4 | +18.5 | +17.8 | +5.4 |
| Equal weight | +31.6 | +22.0 | -8.7 | +23.2 | +24.2 | +19.5 | +9.0 |
| SPY | +18.3 | +28.7 | -18.2 | +26.2 | +24.9 | +17.7 | +14.1 |

Average book by regime (439 rebalances): Calm: Exp2 39%, trend 17%, SPY 13%, best ideas 11%, conviction 11%, Pelosi 6%, reversal 5%. Trending: Exp2 33%, SPY 13%, best ideas and conviction 12% each, reversal 8%, Exp7 8%, Pelosi 7%, Exp1 7%. Crisis: Exp7 39%, trend 21%, Exp1 8%, the rest (about a third) in cash.
Today (Calm): Exp2 40%, trend 16%, SPY 12%, best ideas 11%, conviction 11%, Pelosi 5%, reversal 5%.

## Reading it
- It does what an allocator should: about SPY's return with half the volatility and half the worst drop (Sharpe 1.42 against 0.82). In 2022 it lost 6% against SPY's 18%.
- Regimes help a little, not a lot: the same rule without regimes has Sharpe 1.34 and a -18.5% worst drop. Perfect hindsight would add about 3 points a year, the most this idea could ever give.
- It does not beat simply holding every pot equally on return (+17.2%), only on risk. And it trails Exp2 alone, the best pot on these numbers. Exp2's history is a frozen model's replay with a synthetic variance swap; with tradeable VIX products its Sharpe fell from 1.54 to about 0.95 in the OpenBB test, so the allocator's 40% in Exp2 is the weakest part of this result.
- Survivorship flatters reversal and the copy pots; Exp1 has no news in its replay.

Scripts: scripts/build_returns.py (collects each pot's daily returns), scripts/study.py, scripts/allocator.py (the rule). Data: returns.csv, equity_daily.csv, result.json.
