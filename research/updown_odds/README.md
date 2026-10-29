# What tilts the odds of up vs down? (2026-10-29)

Joaquin's question: a stock's next move looks like a coin flip, so what can you observe that moves the odds away from
50%? Paper research only.

**Short answer.** The coin is not fair, but the bias is in time, not in signals. SPY rises on 54.6% of days, 65% of
months and 93% of 5-year spans (2001-2026). Of 36 published signals we had not tried, one moves the odds
measurably: one-day reversal, at 50.8%, and 5 bp a side turns that into a loss. Feeding all of them to a
walk-forward model gives 50.0% (AUC 0.501).

## Data and method

- Point-in-time S&P 500 (members as they were on each day, ~500 a day, 1,138 names 1999-2026, from the public
  sp500-data set, the same source as the survivorship fix). A delisted stock keeps its last price. Sectors and SPY/VIX
  from the papers-data cache; SPY/QQQ 1-minute bars 2019-2026 from exp7-data.
- **Stock signals:** each period, the 5% of members the signal likes most (Sharma ch.3 tails). A hit means the stock
  beats the members' median over the next period, so 50% is a coin flip by construction and the market's drift
  cancels out. Also shown: hits on beta-adjusted returns (Sharma ch.4), plain up-rates, and the top-minus-bottom 5%
  spread after 5 bp a side (Sharma ch.9).
- **Market conditions:** P(SPY up next day or next month | condition) against P(up | otherwise), by an OLS of the up
  indicator on the condition.
- Newey-West t-stats; Holm correction across all 36 tests (full period 2001-2026, mostly after each paper's own
  sample, so out of sample for the original claims). Subperiods 2013+ and 2019+ check for publication decay
  (Sharma ch.8).

## 1. Base rates: the horizon is the biggest tilt (`base_rates.py`)

| Holding period | SPY up | One S&P 500 stock up | Stock beats SPY |
|---|---|---|---|
| 1 day | 54.6% | 51.2% | 49.5% |
| 1 week | 58.4% | 53.5% | 49.8% |
| 1 month | 65.4% | 56.3% | 49.9% |
| 3 months | 70.5% | 59.4% | 49.8% |
| 1 year | 80.8% | 65.1% | 48.3% |
| 3 years | 87.0% | 72.5% | 46.0% |
| 5 years | 92.7% | 75.6% | 43.7% |

The index's odds improve with time because a small daily drift adds up while the noise partly cancels. A single
stock's odds of beating the index *fall* with time: a few big winners carry the average, so most stocks lag it
(Bessembinder 2018).

## 2. Stock signals, 2001-2026 (`study.py`, `results/atlas.csv`)

| Signal (paper) | Freq | P(top 5% beats median) | t | Holm p | Net top-bottom spread |
|---|---|---|---|---|---|
| One-day reversal | day | **50.8%** | 3.5 | **0.01** | -35%/yr (gross +10%) |
| One-day reversal within sector | day | 50.6% | 3.0 | 0.10 | -33%/yr |
| Fade the overnight gap | day | 49.9% | -0.5 | 1 | -52%/yr |
| One-week reversal (our pot) | week | 50.5% | 1.1 | 1 | +4.2%/yr, t 0.7 |
| One-week reversal within sector (Da-Liu-Schaumburg) | week | 50.1% | 0.3 | 1 | +4.1%/yr |
| Volume shock (Gervais-Kaniel-Mingelgrin) | week | 49.4% | -1.9 | 1 | -5.1%/yr |
| Earnings week, volume-spike proxy (Frazzini-Lamont) | week | 50.4% | 1.4 | 1 | +0.2%/yr |
| Overnight minus intraday return (Lou-Polk-Skouras) | week | 49.3% | -1.5 | 1 | +0.8%/yr |
| 12-1 momentum | month | 49.8% | -0.3 | 1 | +1.5%/yr |
| 52-week-high proximity (George-Hwang) | month | 48.7% | -1.4 | 1 | -5.5%/yr |
| One-month reversal | month | 47.8% | -2.5 | 0.44 | -2.3%/yr |
| Industry momentum 6m (Moskowitz-Grinblatt) | month | 50.1% | 0.1 | 1 | +4.5%/yr |
| Low volatility 3m | month | 50.0% | 0.0 | 1 | -2.6%/yr |
| Low beta (betting against beta) | month | 48.8% | -0.8 | 1 | -3.0%/yr |
| Low MAX / lottery (Bali-Cakici-Whitelaw) | month | 48.2% | -1.7 | 1 | -3.6%/yr |
| Amihud illiquidity | month | 49.3% | -0.9 | 1 | +4.5%/yr |
| Same-calendar-month seasonality (Heston-Sadka) | month | 50.4% | 0.5 | 1 | +0.1%/yr |
| Smooth momentum, frog in the pan (Da-Gurun-Warachka) | month | 49.7% | -0.3 | 1 | -0.6%/yr |
| Dividend month (Hartzmark-Solomon) | month | 50.1% | 0.4 | 1 | -0.8%/yr |
| Earnings month, volume-spike proxy | month | 50.2% | 0.5 | 1 | +0.3%/yr |

Two things look like edges but are not direction:
- **Low volatility** stocks rise in 59.2% of months against 55.8% for all members (t 3.3), and beat the median on
  beta-adjusted returns (t 5.1). They do not beat the median on plain returns (50.0%). They rise more often but by
  less, so this is the known low-risk anomaly (better return per unit of risk), not better odds of a gain.
- **One-month losers** beat the median only 47.8% of the time, yet the top-bottom spread is about zero before costs.
  The losers lag most months and then rebound sharply now and then. Hit rate and money are not the same thing.

## 3. Market conditions (SPY)

| Condition (paper) | P(SPY up) when true | Otherwise | t | Holm p | 2019+ when true |
|---|---|---|---|---|---|
| SPY above its 10-month average, next month (Faber) | 68.7% | 52.6% | 2.6 | 0.35 | 65.3% (vs 72.2%) |
| SPY above its 200-day average, next day | 55.5% | 52.3% | 2.5 | 0.40 | 56.0% |
| Day before a market holiday (Ariel) | 58.4% | 54.5% | 1.2 | 1 | 67.6% (74 days) |
| After a 20% VIX jump | 62.4% | 54.5% | 1.7 | 1 | 63.6% (44 days) |
| After a 2% down day | 56.7% | 54.6% | 0.7 | 1 | 60.3% |
| Monday | 55.1% | 54.5% | 0.4 | 1 | 60.9% |
| Turn of the month (Lakonishok-Smidt) | 54.2% | 54.7% | -0.3 | 1 | 55.2% |
| VIX above 30 | 54.6% | 54.6% | 0.0 | 1 | 53.7% |
| Under 20% of members above their 50-day average | 55.0% | 54.6% | 0.2 | 1 | 51.4% |
| November-April (Halloween) | 64.7% | 64.7% | 0.0 | 1 | 62.2% |
| After an up day | 53.5% | 56.0% | -2.1 | 1 | 55.0% |
| Overnight move up vs open-to-close up | 54.1% | 53.3% | 0.9 | 1 | 56.3% vs 54.2% |

Intraday momentum (Gao-Han-Li-Zhou), SPY and QQQ 1-minute bars 2019-2026: the sign of the first half hour
(overnight included) or of the 12th half hour matches the sign of the last half hour on 50.5-52.0% of days; trading
it nets -1.7% to +0.1% a year after 0.3 bp a side.

The trend filters (10-month and 200-day) are the closest to real: they mostly say "stay out of bear markets", and
the 10-month version has not helped since 2013.

## 4. Everything at once (`combo.py`, `results/combo.json`)

All 20 stock signals plus two market-state inputs, ranked 0-1 each month, into gradient-boosted trees and a logistic
regression refit each January on all earlier months (last month purged; Coqueret & Guida ch.4 and ch.10).
Out of sample, 2006-2026:

| Model | AUC | P(top 5% beats median) | Net top-bottom spread |
|---|---|---|---|
| Boosted trees | 0.501 | 48.8% | -2.2%/yr |
| Logistic | 0.501 | 49.7% | +0.8%/yr |

Calibration: when the trees said 58% the stock beat the median 49.1% of the time; when they said 42%, 49.2%.

## What this means for us

- With public price, volume, calendar and dividend data on S&P 500 stocks, the best measurable tilt is under one
  percentage point, and it lives in one-day horizons where trading costs are bigger than the tilt.
- The odds that do move come from being paid to hold risk: equity drift (time in the market) and selling insurance
  (Exp2's variance swap, the one strategy that passes our edge tests).
- Already tested elsewhere in the project and not repeated here: news and LLM sentiment (Exp1, Exp3), 10-K tone,
  Piotroski and quality, VWAP/EMA intraday, options-based vol signals, insider and congress copying, BTC models.
- Not testable with our free data: order-book imbalance and order flow (needs paid tick data), option skew and
  put-call ratios history, analyst revisions, short interest. The literature finds these work mostly in small stocks
  or at sub-second horizons.

## Rerun

    python panel.py SP500_DIR PAPERS_DIR panel.pkl       # ~20 s; SP500_DIR has membership_intervals.parquet + prices.parquet
    python study.py panel.pkl EXP7_DATA_DIR out/          # ~1 min
    python base_rates.py panel.pkl out/                   # ~2 min
    python combo.py panel.pkl out/                        # ~15 min on 4 cores
