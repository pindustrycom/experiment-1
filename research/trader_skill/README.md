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
