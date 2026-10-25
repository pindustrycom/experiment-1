# Vol-and-curve regimes, weekly walk-forward, 500 CAD test on 2026

## Method
- **Market state at each close**: VIX level, and the *spread* = VIX minus SPY's realized 21-day vol (implied vs realized curve). Positive spread = options price more fear than the market shows; negative = the market is moving more than options expected (stress).
- **Map**: K-Means, 3 groups, named by VIX level: **Calm**, **Normal**, **Stressed**.
- **Weekly recalculation**: every Friday the map is refitted on all data up to that Friday (1998 on), the past is relabelled, and every strategy's record per group from Jan 2020 up to that Friday is recomputed. The next week is labelled with that week's map only (no look-ahead). 361 refits in total.
- **Fluke check**: each week, a regime counts as "good" for a strategy only if its record there so far has t-stat >= 1 (and 40+ days). Then the *following* week's real returns are split into days in "good" vs other regimes. If the good days don't earn more out of sample, the regime pattern was a fluke.
- BTC Donchian and reversal-with-stop aren't included: their data (BTC 4h bars, daily lows) isn't in this container.

## Training 2020–2025: where each strategy does best

| Strategy | Best regime (end 2025) | Same best, % of weeks since 2021 | Next week, good-regime days (ann.) | Next week, other days (ann.) | Verdict |
|---|---|---|---|---|---|
| Exp2 (vol sleeve + stocks) | Calm (Sharpe 3.26) | 91% | +17.1% (Sharpe 2.36) | +13.8% (1.71) | Holds |
| Conviction (13F) | Normal (1.45) | 100% | +30.1% | +13.4% | Holds |
| Best ideas (13F) | Normal (1.45) | 100% | +30.2% | +22.6% | Holds |
| Trend | Calm (1.41) | 47% | +5.0% | +1.6% | Holds, weak |
| SPY | Calm (1.31) | 34% | +17.0% | +15.6% | Mostly noise |
| Weekly reversal | Stressed (1.16) | 99% | +6.3% | +36.0% | Fluke: it works everywhere |
| Pelosi copy | Normal (1.43) | 87% | +31.4% | +46.8% | Fluke |
| Exp7 (SPY VWAP+EMA) | Stressed (1.31) | 100% | 0.0% | +9.0% | Fluke |
| Exp1 (pairs) | Stressed (0.54) | 82% | -28.8% (29 days) | +1.2% | No edge anywhere |

A stable "best regime" is not enough. Reversal and Exp7 kept the same best regime almost every week, yet the next week's money did not come from those regimes.

Diagrams: `spread_<strategy>.png`. Left: average next-day return on the VIX × spread map (blue = made money, red = lost), with the three regime centres. Right: Sharpe in each regime, recomputed every Friday.

## Test: Jan 2 – Oct 1 2026, 500 CAD, same weekly refit
2026 had 124 Calm days, 64 Normal days, 0 Stressed days. Rule: in today's regime, hold the strategies whose record there is positive with t >= 1, sized by return/variance, max 40% each. The rest stays in T-bills. Rebalance every Friday and on any regime change, with a 2 bp switching cost. All strategies trade in USD; CAD value uses Bank of Canada USD/CAD (1.3706 → 1.4243, which adds about +3.9% on its own).

| | End value (CAD) | Made | Worst drop | Sharpe |
|---|---|---|---|---|
| **Regime method** | **561.61** | **+61.61** | -4.6% | 1.78 |
| Regime method, no fluke filter | 561.57 | +61.57 | -4.5% | 1.79 |
| Equal weight, all strategies | 568.84 | +68.84 | -6.0% | 1.74 |
| SPY buy and hold | 586.67 | +86.67 | -8.5% | 1.64 |
| Weekly reversal alone (hindsight best) | 693.68 | +193.68 | | |

It ran mostly Exp2 40%, Trend about 30%, plus SPY and the copy pots, and switched to the 13F copies + Exp7 on Normal days. It gave up returns for a smoother ride, and it underweighted reversal (2026's big winner) because reversal's regime record never passed the t >= 1 bar.

Files: `result.json` (everything, including every rebalance), `weekly_refit_table.csv`, `test_equity_cad.csv`, `test_500cad.png`, `scripts/study.py`.
