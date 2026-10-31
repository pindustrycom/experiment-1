# Time-scale ladder: month -> week -> day -> hour -> minute (2026-10-31)

Joaquin's idea: an "executive" monthly forecast of volatility and direction feeds a weekly model, which feeds a
daily one, then hourly, then minute. Each layer is an EKF state-space model with a GRU on top, run as a feedback
loop: forecasts flow down (the future), each finer layer's latest state flows back up (the present), so the
minute layer is only a small correction and the minute data also help the monthly forecast. Paper research only.

**Short answer.** The ladder works for **volatility** and does nothing for **direction**. Feeding the finer layers'
latest state up cuts the monthly volatility forecast error by about 46% against the standard single-scale model
(HAR), and the daily error by 9%; the GRU adds a further 12% at the weekly layer. No layer predicts up vs down: hit
rates equal the share of up periods, and every direction rule trails or ties holding SPY after costs.

## Data and method (`bars.py`, `ladder.py`, `gru.py`, `trade.py`)

- SPY 1-minute bars 2019-01 to 2026-10 (Tiingo IEX, exp7-data release), regular hours, half days and the holidays
  the feed fills with flat bars removed: 1,966 days, 766,740 minutes. QQQ as a check. Periods: calendar month,
  ISO week, day (close to close, overnight gap included), 7 hourly bins from 09:30 (the last is 30 min), minute.
  Each period: log return and realized variance (sum of squared 1-minute returns, plus the gap for day and up).
- **Layer (EKF).** State = drift and log variance. Log variance: Kalman filter on log realized variance
  (AR(1) around a long-run level, maximum likelihood on training years), intraday time-of-day pattern removed.
  Drift: Kalman filter on returns whose noise is the variance the volatility state predicts (the nonlinear link).
- **Ladder.** Each layer's next-period forecast = least-squares mix, fitted on 2019-2022, of its own state (plus
  HAR terms: last 1, 5, 22 periods), the parent's forecast for the parent period ahead (down), and the child
  layer's state at the end of the current period (up). Processed month first; every input is known before the
  period it forecasts starts.
- **GRU.** One GRU (32 units, last 32 periods) shared by all five layers with a layer id, learning each layer's
  remaining log-variance error and the standardized return. Train 2019-2021, early stop on 2022, mean of 3 seeds.
- **Test 2023-01 to 2026-10.** Volatility: QLIKE loss (the standard for variance forecasts; lower is better) and
  Diebold-Mariano t vs HAR (Newey-West). Direction: hit rate, correlation. Trading: 0.5 bp a side.

## Volatility forecasts (test, SPY; `results/ladder_eval.csv`)

Change in QLIKE against HAR on the same layer (negative = better), DM t in brackets:

| Layer | Last period | Own layer EKF | HAR + parent + child (ladder) | Ladder + GRU |
|---|---|---|---|---|
| Month (46) | -9% | +3% | **-46% (t -3.7)** | -49% |
| Week (200) | +14% | +3% | -8% (t -1.2) | **-19%** (GRU vs ladder t -4.8) |
| Day (959) | +28% | +2% | **-9% (t -4.1)** | -12% (GRU vs ladder t -1.4) |
| Hour (6,713) | +254% | +99% | 0% (t -0.3) | +1% |
| Minute (374,010) | +77% | **-5% (t -3.0)** | -1% | -5% (GRU vs ladder t -2.6) |

QQQ gives the same pattern: month -39% (t -2.6), day -8% (t -3.4), week -7%, hour -4%, minute: own filter best.

What the fitted links say:
- **Up (the present) carries the gain.** The monthly forecast puts its weight on the end-of-month weekly state
  (0.76), the daily forecast on the last-hour state (0.26). Recent vol at a finer scale beats the coarser layer's
  own average. This is the known heterogeneous-market result behind HAR (Corsi 2009), here extended to intraday.
- **Down (the future) adds little.** Parent weights are 0.03-0.14 once a layer has its own history. Only the hour
  layer leans on the day forecast, and HAR on the hour layer already captures that.
- The month result rests on 46 test months; treat the size (-46%) as rough, the direction as solid (both ETFs).

## Direction (test; `results/gru_eval.csv`, `results/trade_direction.csv`)

| Layer | Up share | Ladder hit | GRU hit | Best direction rule vs hold (CAGR, Sharpe) |
|---|---|---|---|---|
| Month | 65% | 61% | 39% | hold 19.4%, 1.52 beats all |
| Week | 58% | 55% | 59% | GRU long/short 20.4%, 1.41 vs 19.4%, 1.35 (it is long almost all the time) |
| Day | 56% | 56% | 54% | GRU long/flat 19.9%, 1.37 vs 19.6%, 1.30 (tie) |
| Hour | 52% | 52% | 49% | hold intraday 6.7%, 0.63; ladder long/flat 5.9%, 0.69 |
| Minute | 50% | 49% | 49% | every rule loses: ladder long/short -16.6%/yr, GRU -96%/yr |

(Hour and minute "hold" is intraday only, without the overnight gap.) No layer's hit rate beats the up share, and
correlations with the next return are 0.00-0.05 with no consistent sign across layers or ETFs. Trading every
minute or hour on these signals turns over the position thousands of times a year; at 0.5 bp a side that alone
is a loss.

## Using the volatility forecast (`results/trade_voltarget.csv`)

Day-scale SPY sized to 10%/yr volatility (max 2x), test 2023-2026:

| Forecast | CAGR | Sharpe | Max DD |
|---|---|---|---|
| Last day's variance | 7.6% | 0.80 | -11.2% |
| HAR | 9.5% | 1.00 | -10.7% |
| Ladder | 9.9% | 1.03 | -11.1% |
| Ladder + GRU | 10.3% | 1.05 | -11.0% |

Sharper volatility forecasts give a slightly better risk-controlled ride; the step from HAR to the full ladder is
small (Sharpe +0.05) and within noise over four years. Holding SPY unlevered did 19.6% at Sharpe 1.30 here.

## What it means

The structure Joaquin described is a sound way to model **risk**: the feedback from fine to coarse scales is real
and large for monthly volatility. It gives no edge on **direction** at any scale, so it cannot pick the 8% days.
Its uses are sizing (how much to hold), stops and option pricing, such as the Exp2 volatility sleeve and the BTC
call-wing gate, which already earn from volatility forecasts.

Rerun: `python bars.py EXP7_DIR SPY bars.pkl; python ladder.py bars.pkl OUT; python gru.py OUT; python trade.py OUT`
(EXP7_DIR = exp7-data.tar.gz from this repo's exp7-data release; numba, scipy, torch; ~4 min on 4 cores).

## Can the ladder flag the 8% days? (`big8.py`, `sign_check.py`, `results/big8.json`)

Joaquin's follow-up: there must be something in the month-to-minute idea that can point at the 8% days. Test on
every point-in-time S&P 500 stock-day (daily OHLC 2000-2026; intraday high-low range as the finest layer),
train 2000-2015, test 2016-2026 (1.36 million stock-days, 0.97% of which move 8%+ either way).

**1. Size: yes, strongly.** The volatility ladder (day, week, month, quarter variance of the stock, the
intraday range, volume, the earnings cycle ~63 days back, the market's own volatility) ranks the 8% days with
AUC 0.89 (vol alone 0.82). The weekly and monthly layers carry most of it (30% and 28% of the model's gain).

| Flag | P(moves 8%+ either way) | P(up 8%+) | P(down 8%+) | Up share |
|---|---|---|---|---|
| Any stock-day | 0.97% | ~0.5% | ~0.5% | ~51% |
| Top 10% of stock-days by forecast | 6.8% | | | |
| The #1 flagged stock each day | **14.6%** (15x the base rate) | 6.7% | 7.9% | 48% |
| Top 5 flagged each day | 10.0% | 4.9% | 5.1% | 50% |

**2. Sign: no.** The flagged stock is as likely to fall 8% as to rise 8%. Given that a flagged move *did*
happen, a little is predictable (AUC 0.56; after a down day for the market or the stock, the big move is more
often up, a rebound), but you do not know in advance which flagged day will be the big one. Turned into a trade,
buying the flagged stock with the best up-score each day: 2,299 trades 2016-2026, 49% winners, -5 bp a trade
(t -0.4); the full model version took $20 to $0 vs $91 for SPY.

**3. Where it could pay: options, untested.** A ladder that knows *when* a big move is likely but not *which way*
is exactly what an option straddle (a call plus a put) bets on, and the size-given-big-move tilt is what a
call bets on. The #1 flagged stock moves 4.5% on average (absolute) against 1.4% for a typical stock-day. Whether
that beats the option prices depends on implied volatility, which already includes known earnings dates and
usually sits above realized volatility. No free history of stock option prices exists, so this can only be tested
going forward, by recording option chains for the flagged names each evening.
