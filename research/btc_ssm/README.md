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
