# Does Reddit chatter predict stock returns? (plus scandal drops, oil, memory chips)

Joaquin's idea (2026-11-23): bots that read past prices don't work; the edge is in real-time information that shapes
a company's future, such as Reddit sentiment, news speculation and shocks like the NVIDIA CEO scandal (Jensen Huang's
affair with an alleged Chinese spy came out ~Nov 8 2026, he resigned Nov 12 and was reinstated ~Nov 23).
Paper research only. Nothing traded or deployed.

**Verdict: no tradeable edge from Reddit.** Unusual Reddit attention comes before slightly *lower* returns (as the
academic studies found), bullish talk helped before 2023 and hurt after, and no trading rule beats SPY after costs.
Nothing passes the Holm multiple-testing correction.

## Data

- **Reddit:** every post in r/wallstreetbets, r/stocks, r/investing, r/StockMarket and r/options, 2018-06 to
  2026-11-21, from the public [Arctic Shift](https://github.com/ArthurHeitmann/arctic_shift) archive: 3.6M posts.
  395k (post, ticker) mentions of point-in-time S&P 500 stocks. A ticker counts if it is in capitals in the title
  ("TSLA") or written $ticker; tickers that are everyday words (ALL, NOW, LOW, WIN...) count only as $ticker.
  Collected on Actions (`collect_reddit.py`, `reddit-sentiment.yml`) into the `reddit-sentiment-data` release.
  Arctic Shift's ticker search times out on r/wallstreetbets, so the collector pages through every post in time order.
  Posts only, not comments. Archived scores are not used because they were recorded after the post.
- **Prices:** point-in-time S&P 500 daily prices (github.com/Johnbrick123/sp500-data), `panel.py`.
- Most of the volume is the 2021 meme mania (226k of 395k mentions, GME alone 190k). 2023-26 average 69 mentions a day.

## Method (`reddit_study.py`)

- A post counts for day d if made before 15:30 New York; trade at d's close.
- Signals per stock-day: attention spike today (att1), last 5 days vs the prior 60 (att5), the same for WSB only
  (wsb5), title tone over 5 and 20 days (sent5, sent20; WSB word list incl. rockets/puts), "DD" research posts (dd20).
- Controls: 1/5/21-day returns, 21-day volatility, dollar volume, abnormal volume, and the stock's usual Reddit level
  (so a signal is not just "popular on Reddit").
- Train 2018-06 to 2022-12, test 2023-01 to 2026-11; rule directions fixed on train. 5 bp per side (Sharma's S&P
  500 cost), SPY as the benchmark, Holm across tests. Sentiment is also tested "slowed down" (20 days), per Sharma ch. 11.

## Results (`results/reddit.json`)

Fama-MacBeth with controls, next 5 days, per 1 standard deviation of the signal:

| signal | train 2018-22 | test 2023-26 |
|---|---|---|
| attention spike att1 | -4.6 bp (t -2.3) | -5.5 bp (t -1.6) |
| 5-day attention att5 | -2.9 bp (t -1.5) | -3.9 bp (t -1.5) |
| WSB attention wsb5 | -3.7 bp (t -1.9) | -5.2 bp (t -1.8) |
| tone 5d sent5 | +5.5 bp (t 2.0) | -12.6 bp (t -2.5) |
| tone 20d sent20 | +4.6 bp (t 1.9) | -5.2 bp (t -1.8) |
| DD posts dd20 | -3.3 bp (t -1.3) | -6.6 bp (t -2.5) |

Holm-adjusted test p-values: best 0.11 (att5 next day). None pass.

Next-week market-adjusted return by attention quintile (stocks with 5+ posts in 20 days), test period:
Q1 (quietest) +0.26%, Q5 (most hyped) -0.44%. The same direction holds in train, but it is small and noisy.

Trading rules, test 2023-26, net of costs:

| rule | result |
|---|---|
| short top-2% attention spikes, 1 day, SPY-hedged | Sharpe 0.24, t 0.5 |
| bullish minus bearish quintile, weekly (sent5) | -24%/yr (the train sign flipped) |
| long most-bullish quintile | +3.1%/yr vs SPY +18.9%, alpha t -1.4 |
| long least-hyped quintile | +27.6%/yr but beta 1.46, alpha +2.0% (t 0.2) |

**The biggest hype spikes** (7x normal attention and 20+ posts in a day, `hype_events.py`): -8.8% over the next week
in 2023-26 looks striking, but it is First Republic Bank's collapse. One event per stock per month: 49 events,
mean -4.6%, **median -0.1%**, t -1.6, half went up.

## Scandal drops (`scandal_study.py`, cross-check of PR #61)

Google News headlines week by week 2012-2026, with CEO scandal events extracted by Claude Haiku (`collect_scandals.py`).
170 events matched to S&P 500 members, 43 with a 3%+ market-adjusted drop (average -6.4%). Buying the close the day
after the drop and holding 20 days: **-1.2% per trade, 40% winners**. That is no better than an ordinary drop of the
same size (-1.5%, t -1.2). Arrests and harassment cases kept falling (-10% and -4% over 20 days). This agrees with
PR #61: scandal drops are not reliably overdone. They bounce only when the cause is reversed, as in Huang's
reinstatement or Musk's SEC settlement, and that reversal is the unpredictable news itself.
Baseline: all S&P 500 drops of 10%+ since 2012 regain ~3% vs the market over 60 days.

## Side questions (`macro.py`, weekly 2010-2026, Newey-West t)

- **Memory chips (RAM):** no free DRAM price series, so memory makers' stocks (MU, WDC, STX) stand in. In the same
  week QQQ moves 0.30x the memory basket (t 16) and NVDA 0.53x. Last week's memory move does not predict next week
  (QQQ t -2.0, NVDA t -1.7, nothing since 2023).
- **Oil / Iran war:** FRED was unreachable from Actions, so S&P 500 oil producers stand in for the oil price. When oil
  stocks rise, tech lags the market in the same week (t -5.8). Last week's oil move predicts nothing (QQQ t -0.9).
  2026 (Iran war): oil up meant tech down, priced the same week.
- **Satellite parking lots / Pentagon pizza:** not testable here (paid imagery; no history). Published work finds the
  satellite edge went to the funds that bought the data and moved prices before earnings.

## What this means

Public information, Reddit included, is priced within the day or week. What we could measure is weak, unstable
across periods and smaller than costs. An edge would need information that is truly not public yet, or faster than
daily data. That is beyond free data and beyond paper trading at this size.
