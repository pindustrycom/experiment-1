# "Just go long on Monday" (open to close, Nasdaq) — is it an edge?

Joaquin pasted a video (2026-11-17) claiming: one NQ contract, buy Monday open, sell Monday close, since 2020,
made $191k; t-stat 3.2; 1 in 1,000 shuffles beat it; both halves (2020-22, 2023-now) work; profit spread across the
day ($57k / 32k / 44k / 34k); after a red Friday ~$800 a Monday vs ~$350 after a green one; dropping the 10 best
Mondays cuts $191k to $104k and t below 2.

**Verdict: real in 2020-2026, absent for the 21 years before, and fragile. Not an edge worth a pot yet.**

- We reproduce the numbers (QQQ as NQ proxy): **t 3.6, $216k per NQ contract gross, $211k after $15/round trip**,
  shuffle p 0.0005. SPY shows the same (t 3.3). No other weekday is anywhere close (all |t| < 0.6).
- **Before 2020 it does not exist.** QQQ 1999-2019: 985 Mondays, -2.8 bp, t -0.6. SPY 1995-2019: -0.9 bp, t -0.3.
  2009-2019 alone: QQQ +2.2 bp t 0.6, SPY t 0.0. Every 3-year window before 2019 has Monday |t| < 2.
  Full history: QQQ t 0.9, SPY t 1.0, and a random weekday beats it 70% (QQQ) / 90% (SPY) of the time.
- **Fragile.** Drop the 10 best Mondays: t 2.4 ($117k); drop 20: t 1.4; drop 30: t 0.5. Drop the 10 *worst*: t 5.8.
  Year by year only 2020 (t 2.6) and 2023 (t 4.0) are individually significant; 2021, 2022 are ~0 (the video's "dead year").
- **Not spread across the day.** First hour (09:30-10:30) is $102k of $211k and the only significant window
  (t 3.4); 10:30-close together is t 1.95. It does lean up all day vs other weekdays (+10 bp vs -0.2 bp), so partly drift.
- **Red Friday story does not hold up as a Monday thing.** After a red Friday $884 vs $494 after green (matches the
  video's direction), but the difference is t 1.3, and Thu/Fri show the same red-vs-green gap (t 1.6, 1.4). It is the
  ordinary 1-day reversal we already tested (research/updown_odds: 50.8% hit rate, killed by costs), not weekend fear.
- **Multiple testing.** 5 weekdays x 2 markets = 10 choices, Bonferroni bar t 2.81, plus the start year (2020) chosen
  after looking. Alpha after 2 bp vs SPY since 2020: **+7.9%/yr, t 2.8, beta 0.09** — right on the bar; since 2023
  t 2.6; since 1999 t -0.3. The project's ~300-strategy bar is t 3.8 (edge tests, 2026-10-26). Sharma ch.2 (Harvey et al.
  2016, Hou et al. 2020) is exactly this: a data-mined calendar anomaly that needs a stricter bar and a new sample.
- **As a 500 CAD pot it loses to holding.** 2020-now: Monday rule (2 bp) 500 -> 926, Sharpe 1.24, max DD -9%;
  hold QQQ 500 -> 1599, Sharpe 0.81, DD -36%. 2023-now: 718 vs 1277. It is in the market 19% of days, so the
  money only beats holding with leverage (one MNQ is ~$50k notional, far too big for this pot).
  Holding Friday close to Monday close (weekend + Monday) is about as good since 2023 (Sharpe 1.87).

## What would change the verdict
The 2020+ pattern is consistent with newer flows (retail and weekly/0DTE options opening Mondays), so it could be a new
regime rather than luck. The honest test is forward: it was "found" in Nov 2026, so only Mondays from now count.
At ~22 bp a Monday and ~1.1% Monday vol, one year of new Mondays gives t ≈ 1.4 if the effect is fully real; reaching t 2.8 takes ~4 years.
A paper pot (QQQ Monday open->close, 2 bp, 500 CAD) costs nothing to run as that forward test — offered, not started.

## Data and method
- Daily OHLC: github.com/Johnbrick123/sp500-data (prices.parquet; QQQ 1999-03..2026-11-13, SPY 1995-01..).
  Open->close is same-day, so no split/dividend adjustment needed. NQ $ = 20 x 41 x QQQ open x return (NDX/QQQ ~ 41).
- Intraday split: exp7-data release, Tiingo IEX 1-minute QQQ bars (2020-01..2026-10), holiday prints dropped;
  4-window product matches daily open->close (corr 0.997).
- Shuffle: 2,000 permutations of weekday labels; reported both "Monday's t vs shuffled Monday" (the video's test)
  and "Monday's t vs the best of 5 shuffled weekdays" (accounts for picking Monday after looking).
- Costs: NQ $15 per round trip; ETF 2 bp round trip.

    python monday.py DAILY.pkl EXP7_DIR OUT_DIR   # DAILY.pkl = prices.parquet rows for QQQ/SPY

Results in `results/results.json`; per-day QQQ 2020+ in `results/qqq_daily_2020.csv`.
