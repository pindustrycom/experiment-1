# Daily big-mover algorithms for a $20 account (2026-10-30)

Joaquin's idea: stocks can rise ~8% in a day; catch 100 of those, reinvest everything, and $20 grows a lot.
Paper research only, no real money.

**The math is right about the prize and silent about the cost.** 100 straight +8% wins compounded is 1.08^100 = 2,200x
($20 -> ~$44,000), even more than 800%. But each pick also has losers. With +8% wins, the share of trades you must win
just to break even (compounding) is:

| Typical losing trade | Hit rate needed to break even |
|---|---|
| -2% | 21% |
| -4% | 35% |
| -8% | 52% |

**What the algorithms actually hit.** Across 2000-2026, the stock any of these rules picks rises 8%+ the next session
on 2-5% of trades, and falls 8%+ about as often. None gets near 21%, let alone 35%.

## Data and method (`panel.py`, `study.py`)

- Point-in-time S&P 500 (members as they were each day, delisted names kept, halted days skipped), daily
  open/high/low/close 2000-01 to 2026-10-28, from the public sp500-data set (same source as the survivorship fix).
  $5 price floor. Opens outside the day's high-low range are skipped as bad prints.
- One position per day (or equal split across the top 5), all money reinvested, long only (a $20 cash account cannot
  short), fractional shares. Commission-free broker, so the cost is the spread and slippage: 5 bp a side base
  (Sharma ch.9 for S&P 500 names), 15 bp stress (top movers have wider spreads).
- Close rules decide at today's close and hold to tomorrow's close; gap rules decide at today's open from the
  overnight gap and sell at the close, or at +8% if the high touches it (with a -4% stop where noted; if both touch
  the same day, the stop is assumed first).
- 3x Nasdaq is rebuilt from QQQ daily (3x the day's return, minus 0.95%/yr fee and 2%/yr financing on the borrowed
  2x), the 200-day average seeded from QQQ's 1999 history. BTC uses the btc_ssm study's daily series (2021-2026), 10 bp a side.
- t is the t-stat of the average trade after costs.

## Results, 5 bp a side (`results/results.csv` also has 15 bp)

| Rule | $20 from 2000 | $20 from 2019 | $20 from 2025 | Win % | Up 8%+ | Down 8%+ | Avg trade | t (2000+) |
|---|---|---|---|---|---|---|---|---|
| Chase yesterday's #1 gainer | $0.00 | $0.71 | $13.79 | 46% | 2.7% | 2.7% | -14 bp | -2.7 |
| Chase top 5 gainers | $0.00 | $6.92 | $28.95 | 46% | 2.1% | 1.9% | -11 bp | -5.2 |
| Buy yesterday's #1 loser | $0.00 | $0.67 | $22.00 | 49% | 2.8% | 3.4% | -21 bp | -3.3 |
| Buy 5 biggest losers | $0.00 | $1.77 | $21.85 | 49% | 2.2% | 2.4% | -11 bp | -4.7 |
| Breakout: 20-day high on 2x volume | $0.00 | $1.09 | $27.50 | 47% | 1.1% | 1.0% | -12 bp | -3.0 |
| Most volatile stock (20-day vol) | $0.00 | $0.75 | $42.13 | 46% | 4.8% | 4.2% | -6 bp | -0.9 |
| Gap-and-go: biggest gap up, open to close | $0.00 | $0.00 | $2.00 | 43% | 2.3% | 2.9% | -46 bp | -9.4 |
| Biggest gap up, sell at +8% or close | $0.00 | $0.00 | $1.33 | 43% | 5.1% | 2.8% | -44 bp | -9.4 |
| Biggest gap up, +8% target, -4% stop | $0.00 | $0.00 | $2.30 | 40% | 4.2% | 0% | -45 bp | -11.5 |
| Gap fill: biggest gap down, open to close | $0.55 | $0.20 | $10.58 | 52% | 3.9% | 3.7% | +8 bp | 1.3 |
| Biggest gap down, +8% target, -4% stop | $55.66 | $1.24 | $8.25 | 49% | 6.7% | 0% | +8 bp | 1.8 |
| Random S&P 500 stock each day (control) | $0.02 | $3.27 | $14.54 | 48% | 0.6% | 0.5% | -7 bp | -2.5 |
| Hindsight: tomorrow's #1 gainer (impossible) | astronomical | | | 100% | 54% | 0% | +1,023 bp | |
| SPY hold | $170 | $69 | $26.87 | | | | | |
| QQQ hold | $179 | $97 | $28.10 | | | | | |
| 3x QQQ hold (TQQQ-like) | $11.95 | $398 | $38.97 | | | | | |
| 3x QQQ only when QQQ > 200-day average | $282 | $375 | $32.24 | | | | | |
| 3x QQQ the day after a QQQ down day | $6.26 | $154 | $34.70 | | | | | |
| BTC hold (from 2021) | | $54.65 | $16.95 | | | | | |
| BTC: buy the day after a +5% day (from 2021) | | $19.09 | $18.19 | 42% | 3.0% | 2.0% | +2 bp | 0.0 |
| BTC: buy the day after a -5% day (from 2021) | | $21.89 | $19.50 | 47% | 8.0% | 4.6% | +21 bp | 0.4 |

Max drawdowns: every stock-picking rule loses 87-100% from peak at some point; SPY -55%, QQQ -83%, 3x QQQ hold -100%,
3x QQQ with the 200-day switch -95% (2000-02) and -55% since 2019.

Base rates (`results/base_rates.json`): a random S&P 500 stock rises 8%+ the next day 0.59% of the time and falls 8%+
0.54%. On 54% of days at least one member rises 8%+ (the median #1 gainer makes 8.5%), so the movers exist; the rules
just cannot tell which one it will be.

## What this says

1. **Daily trading costs ~25%/yr before you are right about anything.** Turning the whole account over every day at
   5 bp a side is 10 bp a day, about 25% a year. The random-stock control shows it: -22%/yr. A rule needs a real
   +10 bp/day edge just to tie a coin flip, and none of these has one over 26 years.
2. **Yesterday's movers are not tomorrow's.** Chasing gainers, buying losers and breakouts all hit 8% about as often
   as they lose 8%, with win rates of 46-49%. The 2025-26 rows look better (Most volatile $20 -> $42) but have t of
   0.5-1.4 over 456 days and the same rules lose everything over 2000-2026. That is luck in a hot market, the same
   lesson as the edge tests.
3. **Gap-ups fade.** Buying the biggest gap up at the open loses 0.45% a trade (t -9.4, the strongest effect here).
   The mirror (shorting it) needs a margin account and borrow, so it is not open to a $20 account, and on these opens
   it is the kind of number that needs minute data before anyone trusts it.
4. **The only aggressive thing that beat holding the index over 26 years is not stock picking:** 3x Nasdaq, held only
   while QQQ is above its 200-day average (checked daily, ~6 switches a year, in the market 72% of the time).
   $20 -> $282 since 2000 vs $179 for QQQ, but it fell 95% in 2000-02 and 55% in 2022, and since 2019 it is about
   equal to just holding 3x ($375 vs $398). Its edge is avoiding leveraged crashes, not catching movers.
5. **Small caps were not tested** (no free point-in-time small-cap data). They move more, but their spreads are
   20-100 bp, and at 15 bp every rule above is already deeply negative, so smaller and more volatile only makes the
   cost problem worse.

Practical notes for $20: a cash account avoids the US pattern-day-trader rule (a margin account under $25k gets 3 day
trades per 5 days) and T+1 settlement lets you re-use sale money the next day. A Canadian broker that converts CAD to
USD on each trade (Wealthsimple charges 1.5%) would cost more than any edge here.

Rerun: download `prices.parquet`, `membership_intervals.parquet`, `no_trade_days.csv` from
github.com/Johnbrick123/sp500-data release `data`, and `papers-data.tar.gz` from this repo's `papers-data` release;
`python panel.py SP500_DIR panel.pkl`; `python study.py panel.pkl BTC_DAILY.csv PAPERS/prices/QQQ.csv OUT_DIR`
(BTC_DAILY = research/btc_ssm daily_returns.csv in project files). About 1 minute.

## Parametrizing the randomness (`params.py`, `results/params.csv`)

Joaquin's follow-up: luck is random, but a random process has parameters. Each rule's daily account return fitted
as a random process (2000-2026, 5 bp a side): mean m, volatility s, growth after volatility drag g = m - s^2/2,
Kelly bet f* = m/s^2 (best share of the account to put in; 0 or below means do not play), Student-t tail index
(lower = fatter tails), and the lognormal odds of outcomes in one year.

| Rule | Mean/day | Vol/day | Drag/day | Growth/day | Kelly bet | P(profit, 1 yr) | P(double, 1 yr) |
|---|---|---|---|---|---|---|---|
| Chase yesterday's #1 gainer | -14 bp | 4.4% | 10 bp | -24 bp | -0.74 | 19% | 3.1% |
| Buy yesterday's #1 loser | -21 bp | 5.3% | 14 bp | -35 bp | -0.77 | 16% | 4.5% |
| Most volatile stock | -6 bp | 5.7% | 16 bp | -23 bp | -0.19 | 27% | 8.9% |
| Gap-and-go | -46 bp | 4.1% | 8 bp | -55 bp | -2.81 | 2% | 0.06% |
| Gap down, +8% target, -4% stop | +8 bp | 3.5% | 6 bp | +1 bp | 0.62 | 53% | 11.7% |
| Random stock (control) | -7 bp | 2.4% | 3 bp | -10 bp | -1.32 | 25% | 0.6% |
| SPY hold | +4 bp | 1.2% | 1 bp | +3 bp | 2.66 | 66% | 0.07% |
| QQQ hold | +5 bp | 1.7% | 1 bp | +3 bp | 1.65 | 62% | 1.1% |
| 3x QQQ hold | +12 bp | 5.1% | 13 bp | -1 bp | 0.47 | 49% | 18.8% |
| 3x QQQ above 200-day | +9 bp | 3.2% | 5 bp | +4 bp | 0.89 | 58% | 12.4% |

What the parameters say:
- **The 8% dream needs growth of 770 bp per trade** (ln 1.08). The best rule here grows 4 bp a day. The gap is
  about 200x, not a tuning problem.
- **Volatility is what makes winners visible.** The most volatile stock rule has the worst growth but a 9% chance
  of doubling in a year, 15 times the random-stock control. Higher vol buys more lottery tickets and more losers;
  it does not move the mean. This is how the lucky screenshots are produced.
- **Kelly says no bet** on every stock-picking rule (f* < 0). For QQQ the parameters say about 1.6x leverage
  maximises growth (model ~9.7%/yr log vs 8.2% at 1x and 3.1% at 3x before fees): 3x is past the Kelly point,
  which is why plain 3x QQQ went to $12 from $20 since 2000. Kelly assumes the parameters are known and stable.
- **Tails are fat** (t index 1.6-3.5; 2-3 means extreme days are far more common than a bell curve says), so the
  lognormal odds understate both jackpots and wipe-outs. The simulation in `lucky.py` gives 0.9% doubling for
  the random-stock trader against 0.6% from the lognormal.
- **Parameters drift.** The same rules had positive means in 2025-26 and negative over 26 years. An estimate of
  m from one year has a standard error of about s/sqrt(252) per day (~28 bp for a single stock), larger than the
  edge being looked for, so a year of data cannot tell a +10 bp rule from a -10 bp one.
