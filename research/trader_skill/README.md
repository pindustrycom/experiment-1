# trader-skill, tested

[trader-skill](https://github.com/Jakeschincariol/trader-skill) (Jake Schincariol, MIT, one commit on 2026-10-14) is a pack
of 11 Claude skills with standard-library Python tools. It was read before running: it only calls public endpoints (SEC
EDGAR, OpenFIGI, the House Clerk, Yahoo/Stooq, news RSS), shells out only to `pdftotext`, has no install script, and its
order skill never places anything itself. Its 175 offline tests pass.

Nine skills are plumbing for a real brokerage account (budget sizing, risk limits, order tickets, tax lots, alerts,
rebalancing, screening, earnings facts, news). Two produce a signal that can be tested on history, and `run.py` does that
with the pack's own readers, pinned to commit `5189e79`:

- **copycat**: buy an investor's 13F top 5 or top 10 holdings the trading day after the filing, hold to the next filing.
  Run for the 23 investors the pack lists, from 2013, also "with hindsight" (entering at the quarter end) to see what the
  45-day filing delay costs.
- **pelosi**: every stock (or option) buy a House member discloses in a PTR, 2019 onward, entered the day after the
  disclosure: event study at 5 to 252 trading days against SPY, and a portfolio holding each buy for 126 trading days.
  Groups: all members, buys of $50k or more, Pelosi alone, and sells as a check. Also run on the trade date (hindsight).

Costs: 5 bp a side. Prices: Yahoo daily adjusted closes, so delisted names are missing (an upward bias; coverage is
reported). Scanned paper PTRs cannot be read and are skipped.

The workflow `.github/workflows/trader-skill.yml` runs it on GitHub (this container cannot reach SEC or the House site).
Results: `results/summary.md` and `results.json` in the `trader-skill-results` release.

## Result (run of 2026-10-19, 23 filers from 2013, 4,577 House PTRs 2019-2026)

- **Copying 13Fs does not beat SPY as a rule.** All 23 filers equal-weighted: top-10 copy CAGR 14.5% vs SPY 14.1%,
  Sharpe 0.81 vs 0.86, alpha -0.2% (t -0.1). A few filers did well (Elliott +19.8%/yr since 2020, Druckenmiller +12.2%,
  Burry +13.5%), but picking the winners out of 23 after the fact is selection bias; only Elliott's t (2.8) gets close
  to surviving it. Berkshire's copy trailed SPY by 2.6%/yr. The 45-day delay costs about 1%/yr (hindsight version).
- **Copying Congress does not work either.** All 11,582 priced House stock buys, entered after disclosure: -0.3% vs SPY
  after 63 days; the portfolio of them trailed SPY by 4%/yr. Buys of $50k+ show +2.1% at 63 days (t 3.2, overlapping
  events), but as a portfolio the alpha is +0.5% (t 0.2). Sells did not predict drops.
- **Pelosi is the one outlier**: 60 buys, portfolio CAGR 36.8% vs SPY 15.5% (alpha 21%, t 2.0), mostly a few large
  tech/AI positions held a year (+14.9% excess at 252 days, nothing at 21-63 days). One person, 60 trades: a story, not a
  tested edge.

Biases: Yahoo has no delisted stocks (upward bias, 62-100% of copied weight priced), 781 scanned paper PTRs are unread.
