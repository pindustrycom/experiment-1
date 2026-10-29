# Bitcoin as a partially observed system: Kalman filter + GRU hybrid

Research only, paper only. First prototype of Joaquin's research notes (Oct 28, 2026): a Kalman filter on log price and
latent trend, stochastic-vol / jump features as inputs, a GRU that learns (1) the dynamics mismatch and/or
(2) time-varying process and measurement noise Q_k, R_k, and optionally sentiment (Crypto Fear & Greed) and implied vol
(Deribit DVOL) as extra channels. The test compares it with simple baselines, scoring risk and direction separately, and
with the live BTC Donchian 4h pot after costs.

Full tables: [results/tables.md](results/tables.md). Raw numbers: `results/results.json`.

## Verdict

- **Risk:** yes, but the gain comes from the GRU, not from the Kalman filter. The hybrid with the uncertainty head
  has the best hourly predictive likelihood, beating GARCH by 0.027 nats/hour (t = -8.1) and EWMA by 0.048 (t = -8.1)
  over 2021-2026. It is ahead of GARCH in every test year and in low-, mid- and high-vol regimes. In the rare jump
  hours GARCH is slightly better (both are far ahead of EWMA there). But a standalone GRU
  of similar size gets most of the way (-0.024 vs GARCH). The hybrid's extra edge over the plain GRU is 0.003
  (t = -1.4, not significant), and it comes only from 2021-22.
- **Direction:** no. The dynamics head (branch 1) adds nothing over the Kalman filter: dNLL +0.0124 vs +0.0129, and
  IC 0.008 (t 1.2). No model has an hourly IC with |t| above 2.0. Every out-of-sample R2 against a zero forecast is
  negative.
- **Side channels:** Fear & Greed and DVOL add nothing once trailing realized vol is an input (2023-26 ablation; the
  differences are within seed noise). DVOL alone strongly predicts next-day realized vol (t 17), so the information
  is real but redundant. F&G works through its extremes (|F&G - 50|, t 6.5), not its level. Both relate to today's
  and tomorrow's vol about equally, so the data cannot separate "measurement" from "leading input". The notes' rule
  of not counting the same evidence twice applies, and there was nothing to gain from adding them as a measurement.
- **Trading after costs:** nothing beats the Donchian pot. 2021-01 to 2026-10, daily decisions:

  | strategy | 10 bp CAGR / Sharpe / max DD | 25 bp CAGR / Sharpe |
  |---|---|---|
  | BTC buy and hold | 19.1% / 0.59 / -77% | same |
  | **Donchian 4h (live pot rule)** | 22.0% / 0.75 / -50% | 16.1% / 0.60 |
  | Hybrid (uncertainty head), long/flat on drift sign | 17.1% / 0.60 / -70% | 3.8% / 0.29 |
  | Hybrid (both heads), long/flat | 15.3% / 0.56 / -64% | 0.1% / 0.20 |
  | Vol targeting with the hybrid's risk forecast | 7.8% / 0.40 | -2.2% / 0.19 |
  | Vol targeting with 30-day hist vol | 18.2% / 0.58 | 17.1% / 0.56 |

  No strategy's alpha against hold survives Holm correction (all adjusted p = 1.0). The sharper risk forecast makes
  vol targeting worse: it reacts to every hourly vol swing, and the extra turnover costs more than the forecast gains.

**Recommendation:** do not add a live pot. Nothing here beats the Donchian pot or buy-and-hold after costs. The one
real result is a better hourly risk forecast (likelihood in every year and vol regime). The obvious use is as a risk/sizing
input, but in this test that use made returns worse, not better.

## What was run

- **Data:** Binance spot BTCUSDT hourly bars, 2018-01 to 2026-10-27 (public files, 190 outage hours excluded from
  scoring). Crypto Fear & Greed daily from 2018-02, used from 02:00 UTC on its own day. Deribit DVOL hourly from
  2021-03, lagged one extra hour. All of it was collected by the `btc-ssm.yml` workflow into the `btc-ssm-data` release.
- **Walk-forward:** yearly expanding window, test years 2021-2026. Each fold trains on everything before the test
  year, holds out the last 6 months for early stopping, and refits every baseline. Features are normalized with
  training rows only, and labels are only ever the next hour's return. `test_causal.py` checks that changing the future
  never changes past features, vol forecasts, normalizers or any model's forecast.
- **Models** (`models.py`), all predicting N(mu, var) for the next hourly log return:
  random walk + 30-day hist vol; RW + seasonal EWMA (lambda fit on train); RW + seasonal Gaussian GARCH(1,1) (MLE);
  ridge on the same inputs; Kalman filter [log price, trend] with fixed A, B, C, Q, R; Kalman + vol/jump inputs (B u_k
  drift, q_l scaled by the EWMA variance, inputs: trailing RV, bipower jump share, momentum, volume, hour of day);
  standalone GRU (32 units); hybrids = Kalman + inputs + GRU(innovation, log S, inputs) with a dynamics head g(h_k),
  an uncertainty head (log multipliers on q_l, q_m, R), or both. The GRU sees an innovation only after it is observed,
  and its heads at k shape the prediction for k+1. Neural models: 3 seeds each, scored as a seed ensemble.
- **Side-channel ablation:** GRU and hybrid with inputs price / +RV / +F&G / +DVOL / all, test 2023-2026, all trained
  from 2021-04 so DVOL is available throughout.
- **Trading:** decided at 00:00 UTC from the average of the last 24 hourly drift forecasts (long/flat or long/short), or
  vol targeting at 2.5%/day (cap 2x) from the summed hourly variance forecasts. 10 and 25 bp per side. Donchian:
  the live pot's rule on 4h bars built from the same Binance hours, 25 bp at the 25 bp row (10 bp at the 10 bp row).
  Edge test as in the project's edge tests: alpha vs BTC hold with Newey-West t, Holm across all model strategies.

## Caveats

- Every model is Gaussian. The 1% VaR is breached about 1.5% of hours (Kupiec rejects for all except the badly
  over-wide fixed Kalman). Fat tails/jumps in the predictive distribution (Student-t or a jump mixture) are the
  next step for the risk side, and should help most in jump hours, where the hybrid does not beat GARCH.
- The Kalman state here is [log price, trend]. The richer state in the notes ([.., log variance, log jump intensity]
  with an EKF/UKF or particle filter) was not built. Since the gain came from the GRU's noise head and not from the
  filter, a learned variance model might do as well without the filter.
- One model family, one frequency (1h), and 3 seeds. Many configurations have been tried project-wide, so a single
  t of 2 is not evidence (see the project's edge tests).

## Reference book check (Sharma, *The Quantamental Revolution*)

- Ch. 7 (7.2.3-7.2.7, regimes): Sharma prefers statistical jump models to HMMs and asks for sensitivity checks
  ("if your model's bear regime disappears when you extend the sample by six months, it's time to worry"). Here the
  hybrid's risk gain holds in every yearly refit and across 3 seeds (seed sd 0.002-0.003 against a gain of 0.027).
  The dynamics/direction side does not hold up. Cortese, Kolm & Lindström (2023), cited there, find vol and search
  trends drive crypto regimes. That fits the finding here that vol-type inputs matter and directional ones do not.
- Ch. 9 (TCA): costs set at 10 bp (exchange taker fee level) and 25 bp (the live pot's setting) per side.
- Ch. 11 (NLP sentiment helps only when slowed down): Fear & Greed is daily and lagged, and still adds nothing
  beyond realized vol at the hourly horizon.
- Coqueret & Guida ch. 10: walk-forward, early stopping on a held-out tail of the training window, no tuning on test.

## Reproduce

```
gh release download btc-ssm-data --pattern btc-ssm-data.tar.gz   # or run the btc-ssm workflow
mkdir data && tar xzf btc-ssm-data.tar.gz -C data
pip install torch numpy pandas scipy
python run.py data out          # 204 training jobs, about 70 min on 4 CPU cores
python evaluate.py data out     # results.json, tables.md, daily_returns.csv
```

# Round 2 (Oct 29): continuous-time stochastic volatility, and the three follow-ups

Joaquin's question: "the idea was Brownian motion with continuous volatility instead of constant: can the GRU's
volatility go into the continuous-time equation?" The hourly model above already is dS/S = mu dt + sigma_t dW with
sigma_t from the GRU, discretized at dt = 1 hour. It cannot predict direction because sigma_t scales a symmetric shock.
Round 2 tests his model list (GBM, Heston, Heston + jumps, rough vol + jumps, plus log-vol OU), each fitted by
particle-filter likelihood with yearly walk-forward refits (`sv.py`, `sv_run.py`). It then runs the three ways a vol
forecast could pay: (1) trading volatility against Deribit's implied vol, (2) smoothed position sizing, (3) fat tails
for the GRU hybrid (`sv_eval.py`). Full tables: [results/sv_tables.md](results/sv_tables.md).

## Findings

- **Next-hour distribution:** the fat tails matter more than the volatility dynamics. Every model with a fat-tailed
  predictive distribution beats GARCH by a lot (t -23 to -31): Heston + jumps -0.144, log-vol OU -0.137 (the particle
  cloud gives a scale mixture), rough + jumps -0.128. **The best is the KF+GRU hybrid with a Student-t output,
  -0.169** (standalone GRU-t -0.160, so the hybrid is a little better here). It is ahead in every year. The Gaussian
  GRU's gain (-0.027) was mostly the variance level, and follow-up 3 adds the tails. The GRU-t's 1% VaR is the
  first one that is calibrated (0.99% breaches, Kupiec p = 0.86).
- **Next-day realized vol:** HAR-RV is best (QLIKE -0.053 vs GARCH, t -4.6), then Heston + jumps (-0.033, t -5.1).
  The GRUs are no better than GARCH. They predict the next hour, and holding that variance flat for a day loses the
  mean reversion the SDE models build in.
- **Next-30-day realized vol:** this is where the continuous-time models pay off. **Rough vol + jumps is best**
  (QLIKE -0.175 vs GARCH, t -6.9, R2 0.41), then Heston + jumps (-0.153) and HAR (-0.136). The GRUs are worse than
  GARCH (no term structure). Log-vol OU fits hours well but its fitted long-run level is too high, so its multi-day
  forecasts overshoot badly. Use the jump models for horizons beyond a day.
- **Follow-up 1, the volatility trade:** BTC has a real variance risk premium. Implied 30-day variance (DVOL^2) was
  above the next 30 days' realized variance by 16.6% of the strike on average (2021-03 to 2026-09). Selling a 30-day
  variance swap every day earned 12.9% of the strike per trade after 1 vol point per side costs: t = 3.1 (Newey-West,
  overlapping), Sharpe about 1.3, 72% winners, uncorrelated with Donchian (0.03). At 2 vol points per side it falls to
  t 2.2. Filtering with the models (sell only when implied > forecast) raises the per-trade payoff to 19-21% and t
  to 3.8-3.9 (rough, Heston + jumps; Sharpe about 1.6). But it trades less, and no filter beats always-short
  significantly (best t vs always-short -0.3, Holm p >= 0.11). So the premium is real, and the models don't
  improve on just collecting it. Tail risk: the worst trade lost 1.87x its strike (realized variance 2.9x implied).
  This is a synthetic swap: Deribit lists options, not variance swaps, so a tradeable version needs an option strip
  and will cost more (Exp2's tradeable version kept about 60% of the synthetic Sharpe). t 3.1 is below the
  project-wide Bonferroni bar of 3.8.
- **Follow-up 2, sizing:** vol-targeting with any fast forecast still loses to slow ones, even with day-ahead
  forecasts and a 25% no-trade band. Turnover of 20-60x a year eats the gain. **Donchian scaled by 30-day
  historical vol** is the best sizing result: 27.3%/yr, Sharpe 0.89, max DD -36%, against the plain Donchian rule's
  16.1%, 0.60 and -50% (both at 25 bp per side on the 4h trades, +10 bp on resizing). It is not significant (t vs
  Donchian 1.2), and the model forecasts (rough, Heston + jumps: Sharpe 0.71) do not beat the slow historical one.
- **Follow-up 3, fat tails:** done above. A Student-t output fixes the tail calibration and is the single biggest
  gain in the study, but it changes no trading result.

## Recommendation (round 2)

There is still no directional edge. The one candidate strategy is the **BTC variance risk premium** (short 30-day
variance against DVOL). It is the same kind of edge as Exp2's vol sleeve, the only thing that passed the project's
edge tests, it is uncorrelated with Donchian, and it is t 3.1 here. Before any paper pot it needs a tradeable version:
Deribit option strips or ATM straddles with real bid/ask from Deribit's history. Rough vol + jumps is the forecast to
use for its 30-day sizing. A small variant worth paper-tracking cheaply is Donchian scaled by 30-day historical vol.

# Round 3 (Oct 29): the volatility trade with real Deribit prices

`vrp_trade.py`. On the 1st of each month at 00:10 UTC, sell the Deribit BTC expiry closest to 30 days at the real bid,
delta-hedge daily in perpetuals (5 bp), and settle at Deribit's delivery price. The option chains with bid/ask come
from Tardis.dev's free first-of-month files (`fetch_options.py`, `btc-ssm.yml`). Tardis rate-limits free downloads,
so the data was collected over several runs; it now covers **66 months, 2021-04 to 2026-09** (every month). P&L is in variance-swap strike units, comparable with round 2's (K - RV)/K.

| version, 66 months | mean / month | t | winning months | worst |
|---|---|---|---|---|
| Synthetic swap at DVOL (round 2's trade, same dates) | +9.5% | 1.5 | 74% | -1.87 |
| Strip (dK/K^2 OTM options, the variance-swap replication), at mid | +8.0% | 1.3 | 71% | -1.53 |
| **Strip sold at bid + hedge costs (tradeable)** | **+5.1%** | **0.8** | **70%** | -1.56 |
| ATM straddle sold at bid + hedge costs | -1.2% | -0.2 | 56% | -1.35 |

By entry year, tradeable strip (synthetic in brackets): 2021 +19.8% (+18.6%), 2022 +8.4% (+13.1%), 2023 +4.6%
(+17.0%), 2024 +7.2% (+7.9%), 2025 +6.5% (+4.1%), 2026 -17.6% (-5.3%). The premium has been shrinking year by year
(a maturing options market is a plausible reason, not tested here).

- The tradeable strip keeps about 54% of the synthetic premium. The bid/ask spread costs about 2.9% of the strike a
  month and hedging about 3.0%. The straddle keeps nothing: ATM vol is priced closer to fair, and the premium is in
  the wings.
- Capital view (one straddle per BTC of capital): 2.3%/yr, Sharpe 0.25, max DD -18.7%. The monthly correlation with
  Donchian is -0.22 (Donchian Sharpe over the same windows: 0.66).
- Losing months are the crash months: May 2021, May-Jun 2022, Nov 2022 (FTX), Mar 2023, Mar 2024, Aug 2024, Mar 2025, Oct 2025, Feb 2026 (the worst strip month, -1.56x the strike), Jun, Aug and Sep 2026.
- **Verdict:** the premium is real and survives real prices in sign, and it has been shrinking: in 2026 it is negative
  after costs. With 66 monthly trades it is not significant (t 0.8 tradeable, 1.5 even synthetic, against the project bar of 3.8). It does not justify a pot.
  Monthly entries are the limit here: daily entries need the paid Tardis data (or a Deribit history export), which
  would give about 30x the trades and settle the question.

# Round 4 (Oct 29): can the tradeable vol trade be made to work?

`vrp_variants.py`, on the same 66 first-of-month chains. Changes from round 3: Deribit's option fees are now charged
(0.03% of the underlying per contract plus 0.015% delivery on in-the-money options, capped at 12.5% of the option
price), the hedge runs on the hourly path, and three families of changes are tested, 140 variants at the bid in all,
Holm-corrected:

- **Hedging:** daily (round 3), every 4 hours, or only when the book's delta drifts past a band of 0.1, 0.2 or 0.3 BTC
  per straddle of vega (checked hourly).
- **Strikes:** the full strip 0.6-1.6 F, a narrow strip 0.8-1.25 F, the put wing only, the call wing only, the ATM
  straddle, a 25-delta strangle, and an iron condor (with 10-delta wings bought as crash insurance).
- **Timing:** always sell; sell only when ATM implied variance is above the rough-vol 30-day forecast (round 2's best
  long-horizon model, walk-forward); sell only when it is 20% above; or size each month by the gap.

| variant (sold at bid, all costs) | months traded | mean / month (strike units) | t | Holm p | worst |
|---|---|---|---|---|---|
| Round 3 book: strip, daily hedge, always (now with fees) | 66 | +2.8% | 0.4 | 1 | -1.59 |
| Strip, band 0.1, always | 66 | +4.0% | 0.7 | 1 | -1.27 |
| Call wing, band 0.1, always | 66 | +11.6% | 2.1 | 1 | -1.40 |
| Strip, band 0.1, **rough-vol gate** | 28 | +7.9% | 3.4 | 0.16 | -0.54 |
| 25-delta strangle, band 0.1, rough-vol gate | 28 | +9.9% | 3.7 | 0.06 | -0.53 |
| **Call wing, band 0.1, rough-vol gate** | 28 | **+11.3%** | **4.4** | **0.006** | -0.21 |

(Means count the months the gate stays out as zero.)

What works and what does not:

1. **The timing is the missing piece, and it is where the volatility models pay.** In the 28 months the rough-vol
   gate sells, the strip earns +18.6% of the strike a month. In the 38 it skips, -6.7% (the synthetic premium in
   those months is -22%: implied vol was below what the model expected). The gate stays out of most crash months:
   May 2021, May 2022, Nov 2022, Mar 2023, Oct 2023, Aug 2024 and Mar 2025. The gap correlates 0.39 with the call-wing P&L.
   Checks: random gates that trade 28 months beat the rough-vol gate's t only 0.3-0.5% of the time. Other model
   forecasts help too (HAR t 3.5, GARCH 2.8, Heston+jumps 2.3 on the call wing), while a model-free gate (implied vs
   trailing 30-day realized) does not (t 1.4). So it is the forward-looking forecast that helps, not just "sell when IV is high".
2. **Hedging on a band instead of every day** saves part of the hedge cost and drops less of the move: at 0.1-0.2 it
   adds about 1-4% of the strike a month on every book.
3. **The premium sits in the call wing.** Short delta-hedged OTM calls are the best book in every hedge scheme; short
   puts lose (the crash months), the straddle is near zero, and the iron condor is worse (its wings cost 9% of the
   strike in spread and 7% in fees for little protection).

Capital view (one straddle of vega per BTC of capital, monthly compounding, 2021-04 to 2026-09):

| book | CAGR | Sharpe | max DD | worst month | 2021 | 2022 | 2023 | 2024 | 2025 | 2026 |
|---|---|---|---|---|---|---|---|---|---|---|
| Call wing, band 0.1, rough-vol gate | 10.6% | 1.83 | -1.9% | -1.0% | +35.7% | +6.7% | 0 | +17.1% | +1.5% | +1.2% |
| 25-delta strangle, band 0.1, gate | 9.6% | 1.70 | -3.2% | -2.5% | +29.2% | +10.9% | 0 | +13.0% | +2.3% | 0.0% |
| Strip, band 0.1, gate | 7.7% | 1.49 | -2.6% | -2.6% | +24.2% | +10.2% | 0 | +9.7% | +0.4% | -0.3% |
| Round 3 book (strip, daily, always) | 4.6% | 0.48 | -9.2% | -7.5% | | | | | | |

Correlation with the Donchian pot: -0.03 to -0.13.

**Verdict.** The gated call wing is the first BTC result in this study to clear the project's t 3.8 bar after Holm
correction (t 4.4, Holm p 0.006), and it is close to uncorrelated with Donchian. Reasons for caution:
- It was picked from 140 variants on only 28 traded months.
- Most of the money came in 2021 and 2024. 2025-26 earned about 1-2% a year.
- The gate sat out all of 2023 only because the rough-vol forecast ran high that year.
- Fills at the bid assume small size, and perpetual funding is ignored.

It deserves a forward paper pot, not real money: about 5 trades a year means a forward test takes years to confirm
on its own. Daily entries (collecting Deribit chains every day from GitHub Actions, free going forward) would give
about 20x the trades.

# Round 5 (Oct 29): the seven next tests from the literature review

`round5.py` (items 1, 3-6), `collect_chains.py` + `.github/workflows/deribit-chains.yml` (item 2), `daily_entries.py`
(item 7). Results in `results/round5.json` and `results/daily_entries.json`. The review itself is a Claude Doc in the project.

| # | Test | Result | Verdict |
|---|---|---|---|
| 1 | Vol-size the other pots (long-run vol / 21-day EWMA vol, cap 2x, 25% band, 5 bp; 2020-2026, point-in-time replays) | Sharpe up on long-stock books: SPY 0.81 -> 0.90, best ideas 1.05 -> 1.17, conviction 0.88 -> 1.00, Pelosi 1.11 -> 1.16; down on reversal 0.63 -> 0.59, Exp2 2.15 -> 2.05, Exp7 0.54 -> 0.51. No gain significant (t 0.1-0.9) | As the literature says: helps risk assets a little, hurts short-horizon and vol books. Not worth a change |
| 2 | Daily Deribit BTC + ETH chains | Collector built (public API, 2 requests a day at 00:10 UTC, Tardis-compatible columns, release `deribit-chains`) | Starts once merged |
| 3 | Risk reversal (sell 25-delta call, buy 25-delta put, band 0.1 hedge) | Always +3.4% of gross vega a month, t 0.9; gated +1.3%, t 0.6. The reverse (sell put, buy call) loses -17.1%, t -4.5 | No edge after costs. The skew result is confirmed (calls rich, puts not), but buying the put costs what the call earns |
| 4 | Whalley-Wilmott band (k 0.5-128) on the call wing | Hedge cost 2.6-6.7% of the strike vs 5.5% for the fixed band. Always-sell t 2.1 -> 2.8 (k 0.5); gated t 4.4 -> 4.3-4.6 | Saves hedge cost, same result once gated. Keep the simple band |
| 5 | BTC variance premium as a predictor of the next 30 days' return | BTC: t -0.3 (vs trailing RV) / -0.9 (vs rough forecast); Donchian: t 0.4 / -2.4 (monthly -1.6). Holding BTC only when the premium is above its median: Sharpe 0.38 -> 0.55, in the market 24% of days | No usable return signal (the S&P result of Bollerslev et al. does not carry over) |
| 6 | Half-Kelly leverage on Donchian, 1% daily VaR capped at 8%, cap 3x (2021-2026) | Hist-30 vol: +36.0%/yr, Sharpe 0.90, DD -47%, avg 1.4x when long; the live vol-sized rule +27.3%, 0.89, -36%; plain 1x +16.1%, 0.60, -50%. With the GRU-t forecast: +21.5%, 0.64 (worse) | Kelly only adds leverage at the same Sharpe. The sharper GRU forecast sizes worse than plain 30-day vol, again |
| 7 | Daily entries | Code reproduces round 4 on the monthly chains (+11.3%, 28 of 66 on). Synthetic DVOL swap entered daily, gated: t 4.0 on 30-day blocks (Newey-West), against t 2.5 from the 1st-of-month entries only | Daily entries roughly double the evidence. The real test waits for about 3 months of item 2's chains |

Bottom line: the only new result that matters is item 7's synthetic. With daily entries, the gated variance premium
clears t 4 without relying on 28 months. Everything else (sizing other pots, risk reversals, Whalley-Wilmott, the
premium as a return signal, Kelly with the GRU) adds nothing. The daily chains collector is the next step that can
change a verdict.
