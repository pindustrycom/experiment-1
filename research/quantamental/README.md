# Quantamental: what the books and papers say, and what our backtests found

Joaquin, 2026-10-27, from a 7-slide "quantamental" Instagram carousel. Paper research only: no live pot.

## What each source actually claims, in plain words

**Damodaran, *Investment Valuation*.** A company is worth the cash it will hand out in the future, discounted for time
and risk (the DCF formula on the slide). It is a method for valuing one business carefully, not a claim that cheap
stocks beat the market. Damodaran himself says most valuation "edges" are small and that the story you believe matters
as much as the numbers.

**Grinold and Kahn, *Active Portfolio Management*, and Qian, Hua and Sorensen, *Quantitative Equity Portfolio
Management*.** The quant side: turn each idea into a score for every stock, combine scores, and build the portfolio with
a risk model. Their "fundamental law" says your information ratio is about IC x sqrt(breadth): how well a score ranks
next month's returns (IC, typically 0.02 to 0.05), times the square root of how many independent bets you make. The
catch, which the books stress and the slide skips: bets on 500 stocks are not 500 independent bets (they share market,
sector and factor moves), so the real IR is far below what the formula gives with the naive count.

**Asness, Frazzini and Pedersen, "Quality minus junk" (2019).** Stocks that are profitable, growing, safe and pay out
cash deserve higher prices, and the market does pay more for them, but not enough: buying quality and shorting junk
earned a positive risk-adjusted return in the US and 24 other countries from 1957 to 2016. Much of it is "safety",
close to the low-beta and profitability factors that were already known.

**Piotroski, "Value investing: the use of historical financial statement information" (2000).** Among the cheapest
fifth of stocks by book-to-market, nine yes/no accounting checks (the slide's list) separate winners from losers: high
scorers beat low scorers by about 23% a year in 1976-1996. Piotroski says the effect lives in small, illiquid, unfollowed
stocks with no analyst coverage, exactly where it is hard to trade, and later studies found it much weaker in large caps
and after publication.

**Loughran and McDonald, "When is a liability not a liability?" (2011).** Generic negative-word lists (Harvard IV-4)
misread financial text: about three quarters of their "negative" words (tax, cost, liability, capital) are neutral in a
10-K. Their own finance lists fit the market's reaction to a 10-K better. The claim is about measurement and the
reaction in the days around the filing; they found little reliable predictability of returns weeks later.

**Lopez de Prado, *Advances in Financial Machine Learning* (2018), ch. 10 bet sizing, and HRP (2016).** Size from
the strength of the signal and the risk of the position; inverting a noisy covariance matrix (mean-variance) is
unstable, so he proposes hierarchical risk parity, which clusters similar bets and splits money between clusters.
The claim is better out-of-sample risk, not higher returns.

**Bailey and Lopez de Prado, "The deflated Sharpe ratio" (2014).** If you try many strategies and keep the best, its
Sharpe is inflated by luck; the deflated Sharpe gives the probability the winner's true Sharpe is above zero after
counting the trials, skew and fat tails. Point-in-time data matters for the same reason: a backtest that uses numbers
before they were filed, or restated later, looks great and trades badly.

## How we tested it

- Universe: S&P 500 members at each month end, 2010 to 2026-09, from the project's `papers-data` cache. That cache
  only knows today's members and the date each joined (no removed names), so about 270 names a month in 2010 rising to
  490 in 2026. Companies that later left the index (failures, takeovers) are missing: long-only books are flattered
  (the equal-weighted universe beats SPY by 1%/yr) and long-short books lose their worst junk. The published factor
  series below have no such bias and are the cleaner check.
- Fundamentals: SEC XBRL company facts. A fiscal year counts from the day its 10-K was first filed; every number is
  the latest value filed by that day.
- Text: every 10-K's word counts on the Loughran-McDonald and Harvard lists (`text.py`).
- Monthly rebalance, 10 bp per side, 0.5%/yr borrow on shorts. Stats as in the earlier edge tests.

Full tables: the workflow run's step summary (artifact `quantamental-results`) and `sizing_results/`.

## Results

Our backtests, 2010-02 to 2026-09 (SPY: +14.6%/yr, Sharpe 1.02), on the S&P 500 as it was each month: companies that
later left the index count while they were members (experiments/exp2_sector_ssm/exp2/pit.py; SEC filings found for 270
of the 324 that left since 2010). The first run used today's members only; its numbers are in brackets.

| Strategy | Per year | Sharpe | Alpha vs SPY (t) | Fama-French 6 alpha (t) |
|---|---|---|---|---|
| F-score, cheap high minus cheap low (Piotroski's test) | -5.1% [-2.6%] | -0.41 | -3.1% (-1.3) | -5.8% (-2.1) |
| F-score 7-9, long only | +13.5% [+15.4%] | 0.88 | -1.5% (-1.2) | -1.1% (-1.1) |
| QMJ top 30% minus bottom 30%, value weighted (the paper's form) | +2.9% [+3.8%] | 0.14 | +5.1% (0.9) | +1.2% (0.3) |
| QMJ top 30%, long only, equal weighted | +14.0% [+14.8%] | 0.99 | +0.6% (0.6) | +0.1% (0.2) |
| 10-K negativity falling, finance list (LM), long-short | +0.6% [-0.8%] | 0.13 | +0.1% (0.1) | -0.2% (-0.1) |
| Same with the generic Harvard list | -1.6% [-1.5%] | -0.32 | -1.5% (-1.4) | -1.9% (-1.6) |
| Composite (F + QMJ + LM), top 30% long only | +13.9% [+15.1%] | 0.94 | -0.2% (-0.2) | -0.3% (-0.3) |
| Every member, equal weighted | +13.6% [+15.6%] | 0.87 | -1.5% (-1.1) | -0.6% (-0.7) |

Deflated Sharpe: no book reaches 0.5 even counting only this study's 22 variants; with the project's ~300 earlier
trials every one is 0.00. Information coefficients are 0.000 to 0.016 a month (only payout has t above 2); the
fundamental law's naive IR of 0.3 to 1.1 assumes 350 independent bets a month, and the realized IRs are near zero.
10-K tone and the filing-week reaction: one standard deviation more negative words moves the 4-day return by about
-0.09% on either list (t about -1), so the finance list is not measurably better on these large companies.

Published factor series (every US stock, no survivorship):

| Factor | Before publication | After publication | 2018-2026 |
|---|---|---|---|
| AQR QMJ (USA) | +4.1%/yr, t 3.6 (1957-2013) | +3.1%/yr, t 1.1 | +0.9%/yr, t 0.3 |
| French RMW profitability | +3.3%/yr, t 2.6 (1963-2013) | +1.8%/yr, t 0.9 | +2.2%/yr, t 0.8 |
| French HML value | +5.3%/yr, t 2.7 (1963-1992) | +2.1%/yr, t 0.9 | -0.6%/yr, t -0.1 |

Sizing across our pots (2020-01 to 2026-10-15, `sizing_results/`): risk parity, inverse vol and HRP cut the worst drop
from -23% to -8%/-15% and lift Sharpe from 1.23 to 1.32-1.40, but scaled to the same volatility they earn only
+1 to +2%/yr more than equal money (t 0.5 to 1.1); without Exp2 they do worse. Mean-variance and conviction/vol
sizing add nothing (t 0.0 and 0.1).

**Verdict:** the ideas are real in the long histories the papers used, but on large US stocks since 2010 none of them
beats SPY by more than luck allows, the published factors themselves faded after publication, and sizing changes risk,
not return. Nothing here earns a live pot.
