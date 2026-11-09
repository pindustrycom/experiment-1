# Audit of every strategy and pot, and a candidate built from it (2026-11-09)

Paper research only. Joaquin asked for an audit of the current strategies and whether something better can be
built. Standards as before: point-in-time S&P 500, after-cost returns, comparison with SPY, Newey-West t statistics,
and the multiple-testing bar from the edge tests (about 300 strategies tried so far, so a single result needs
t of about 3.8 to count). Reference: Sharma, *The Quantamental Revolution*, ch. 3 (as-was universe), ch. 4 (timing vs
tilting, beta), ch. 9 (trading costs).

**Short answer.** Nothing in the project beats SPY with evidence except Experiment 2's volatility trade, and that
shrinks to a weak lead once it uses instruments that exist. A candidate built only from what survived the audit
("core-plus") made about the same as SPY since 1998 with smaller drawdowns; its lead over SPY (t 1.5 since 2019) is
not significant. Holding SPY stays the best practical choice.

## Files

| file | what |
|---|---|
| `audit_pots.py` | every pot with a daily replay, 2019-01 to 2026-10: return, risk, beta, alpha, and SPY held at the same beta |
| `reversal_long.py` | the weekly reversal rule on the point-in-time S&P 500 since 1998, at 1, 5 and 10 bp, with a next-open fill |
| `build.py` | the core-plus candidate (spec fixed before running), its parts, and a 27-setting sensitivity grid |
| `common.py` | stats helpers |
| `inputs/` | `returns_pit.csv` (pot replays after the survivorship fix), `vol_products_monthly.csv` (tradeable vol sleeves, openbb-ideas run) |
| `results/` | JSON and CSV output of the three scripts |

Price data: release `papers-data` (point-in-time members with delisted stocks, SPY, QQQ, VIX, T-bill) and
`btc-ssm-data` (BTC hourly bars). Run:

```
python audit_pots.py inputs/returns_pit.csv <papers-data>/prices results/pots.json
python reversal_long.py <papers-data> results
python build.py <papers-data> inputs/vol_products_monthly.csv <btc-ssm-data>/btc_1h.csv ../../experiments/strategies results
```

## 1. Audit, pot by pot

Replays 2019-01 to 2026-10 (`results/pots.json`). "SPY at same beta" is plain SPY held at the pot's beta, borrowing
at T-bill + 1% when above 1.

| pot | per year | Sharpe | worst drop | beta | alpha per year (t) | SPY at same beta | verdict |
|---|---|---|---|---|---|---|---|
| SPY | +17.2% | 0.78 | -34% | 1.00 | | | benchmark |
| Exp2 (synthetic vol swap + stocks) | +17.7% | 1.88 | -13% | 0.29 | +9.5% (4.7) | +7.2% | real, but the swap cannot be traded as modelled |
| Exp1 (pairs) | -0.3% | -0.29 | -22% | 0.02 | -2.9% (-0.9) | +3.0% | no edge |
| Exp7 (VWAP + EMA) | +2.9% | 0.06 | -17% | 0.04 | +0.1% (0.0) | +3.3% | no edge; negative since 2023 |
| Trend (8 ETFs) | +2.3% | -0.07 | -8% | 0.06 | -1.2% (-0.8) | +3.7% | below cash |
| Weekly reversal | +18.8% | 0.59 | -69% | 1.37 | -0.1% (0.0) | +21.5% (drop -44%) | leveraged SPY with bigger falls |
| Pelosi copy | +29.4% | 0.93 | -41% | 1.16 | +10.0% (1.4) | +19.1% | 59 buys, one stock and one year carry it |
| Best ideas | +25.3% | 0.99 | -32% | 1.05 | +6.6% (1.8) | +17.8% | +2.8% (t 0.8) without one fund's GME |
| Fund conviction | +19.9% | 0.85 | -36% | 1.00 | +2.5% (1.0) | +17.2% | SPY |
| Top 3 (reversal, Pelosi, Exp2) | +23.9% | 1.03 | -37% | 0.93 | +6.7% (1.9) | +16.4% | picked in hindsight; alpha t 0.4 in 2026 |

Not in the table because there is no long replay: earnings drift and insider buying (live only, 17 sessions), the
two BTC pots (section 3), the two straddle pots (5 sessions of results, about -10%), the regime allocator (Sharpe 1.36, 40% in
the synthetic Exp2 sleeve, so it inherits that sleeve's problem).

### What the audit found beyond the earlier edge tests

1. **Weekly reversal has no edge over 28 years either.** On point-in-time members 1998-2026 (`reversal_long.py`):

   | | per year | Sharpe | worst drop | beta | alpha (t) |
   |---|---|---|---|---|---|
   | SPY | +9.3% | 0.45 | -55% | 1.00 | |
   | Equal-weight members | +10.8% | 0.50 | -59% | 1.00 | +1.6% (1.4) |
   | Reversal, 1 bp a side (the pot's setting) | +14.6% | 0.49 | -71% | 1.47 | +5.6% (1.2) |
   | Reversal, 5 bp a side | +10.3% | 0.39 | -73% | 1.47 | +1.8% (0.4) |
   | Reversal, 10 bp a side | +5.2% | 0.26 | -79% | 1.47 | -2.9% (-0.6) |
   | Reversal, 5 bp, filled at Monday's open | +9.8% | 0.38 | -75% | 1.44 | +1.4% (0.3) |
   | SPY held at beta 1.47 | +10.8% | | -72% | | |

   The pot turns over about 100 times its value a year, so the cost setting decides the result. Sharma (ch. 9)
   measures about 5 bp for the most liquid 550 US stocks; the week's 20 worst fallers trade wider than that. At
   5 bp it is SPY at 1.5x leverage: same return, same -72% drop. Its alpha against its own universe is -0.6% (t -0.2).
2. **The pots charge 1 bp a side** (`live.COST`). That is fine for ETFs and too low for single stocks. It flatters
   reversal by about 4 points a year and barely matters for the slow pots.
3. **The survivorship fix is not on the site.** PR #38 is still a draft, so the replay figures the site shows for
   reversal and Top 3 are still the flattered ones (about +28%/yr instead of +18%). The live pots themselves are not
   affected: they pick from today's members each day, which is point-in-time by construction.
4. **Signals use the same close they trade at.** Every rule ranks on the closing price and assumes a fill at it.
   For reversal a fill at the next open changes little (table above), so this is not what drives the results.
5. **Delisted stocks are handled correctly** in the point-in-time data (SIVB, Lehman and Enron run down to cents
   and stay in the book until the weekly rebalance).
6. **The leaderboard and Top 3 select on luck.** Picking the best of 16 pots after 17 sessions, or the best three of
   nine replays, guarantees a winner whether or not any pot has an edge.
7. **Straddle pots.** They buy at the ask and sell at the bid a day later on names whose options are already
   priced for big moves. Five sessions cannot settle it, but the cost per round trip is large and the first 25
   straddles lost roughly a tenth of their premium a day.

## 2. Candidate: core-plus

Built only from what survived: market beta pays; volatility can be forecast and direction cannot; the timed short
volatility trade is the one edge; BTC breakout cuts BTC's drawdown. The spec was written before running anything
(top of `build.py`):

- **A, core.** SPY sized at min(1.5, 15% / its volatility over the last 22 sessions), traded a day after the signal
  and only when the size moves by more than 0.10; 2 bp; borrowing at T-bill + 1%.
- **B, vol overlay.** Experiment 2's timed short 2nd-month VIX future at half the research size.
- **C, BTC sleeve.** 5% in the live vol-sized BTC breakout.

| | period | per year | Sharpe | worst drop | alpha (t) |
|---|---|---|---|---|---|
| SPY | 1998-2026 daily | +9.3% | 0.45 | -55% | |
| A | 1998-2026 daily | +9.1% | 0.50 | -46% | +1.6% (1.3) |
| A on QQQ (QQQ itself +10.6%, -83%) | 1999-2026 daily | +13.9% | 0.64 | -54% | +5.3% (3.1) |
| SPY | 2006-2026 monthly | +12.2% | 0.72 | -43% | |
| A + B | 2006-2026 monthly | +13.3% | 0.79 | -27% | +2.4% (1.1) |
| B alone, on cash | 2006-2026 monthly | +3.2% | 0.69 | -3% | +0.8% (1.7) |
| SPY | 2019-02 to 2026-09 monthly | +17.1% | 0.86 | -20% | |
| A + B + C (core-plus) | 2019-02 to 2026-09 monthly | +20.1% | 0.99 | -18% | +3.9% (1.5) |
| 95% SPY + 5% BTC held | 2019-02 to 2026-09 monthly | +19.4% | 0.94 | -22% | +1.6% (1.1) |

- The core does what volatility sizing is known to do: about the same return as SPY with a fifth less risk and
  shallower drops (2008: -21% vs -37%). It lags in sharp rebounds (2009, 2020, 2025). Across 27 nearby settings its
  alpha t runs 0.2 to 1.4, so the fixed setting was not a lucky one, and none is significant.
- The VIX-future overlay earned in both halves (Sharpe 0.59 before 2019, 0.89 after) but adds under 1 point a year
  at this size, and one mini VIX future needs an account of about 40,000 USD at this sizing.
- BTC breakout (25 bp, 6% stop, 2019-2026): vol-sized +56%/yr with a -35% drop vs holding +49.5% and -73%. The
  difference in return against holding is t -0.2: it is a risk cut, not an edge. Most of core-plus's extra return
  since 2019 comes from simply owning 5% BTC in a period when BTC rose.
- With about 300 strategies already tried, t 1.5 is far below the bar (3.8). Core-plus is a sensible
  lower-drawdown way to hold the market, not a demonstrated edge.

## 3. What would be worth doing

1. Charge 5 bp a side on single-stock pots and merge the survivorship fix, so the site shows honest numbers.
2. Show each pot next to SPY held at the same beta on the Markets page; that is the comparison that exposes
   leverage dressed up as skill.
3. If a new paper pot is wanted, core-plus is the one to add as a forward test. For a small real account only the
   core (A) and the BTC sleeve (C) are tradeable.
4. The only place left with real evidence of an edge is selling volatility with a forecast gate. The SPY and Deribit
   option chains being collected daily are the data that can test it with real prices, from about February 2027.
