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

- Universe: S&P 500 members at each month end (membership history from Wikipedia's change list, the project's
  `papers-data` cache), 2010 to 2026-09. Names that left the index stay in the months they were members; names with no
  Yahoo history (mostly takeovers) are missing, which mainly hides junk losers, so it biases against the signals.
- Fundamentals: SEC XBRL company facts. A fiscal year counts from the day its 10-K was first filed; every number is
  the latest value filed by that day.
- Text: every 10-K's word counts on the Loughran-McDonald and Harvard lists (`text.py`).
- Monthly rebalance, 10 bp per side, 0.5%/yr borrow on shorts. Stats as in the earlier edge tests.

Results: see `out/README.md` from the workflow run and `sizing_results/`.
