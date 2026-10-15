# Methodology

Splittability Monitor is a research and paper-tracking tool. Every score is a **relative risk
index**, not a probability of divergence and not a trading signal. No orders are ever placed.

Each pair gets three separate outputs:

| Output | Range | Meaning |
|---|---|---|
| Splittability Index (SI-v1, SI-v2) | 0–100 | Relative risk that the pair's prices will drift apart |
| Directional bias | A advantaged / B advantaged / neutral / uncertain | Which company the documented evidence favours, if either |
| Evidence confidence | 0–1 | Quality and completeness of the evidence behind the score |

## Point-in-time rules (no look-ahead)

All scores, snapshots and backtests are computed "as of" a cutoff timestamp, and only inputs that
were knowable at that instant are used (`backend/app/scoring/pit.py`, `market_calendar.py`):

* A daily close for date *d* is knowable from 16:00 America/New_York on *d* (DST handled by
  `zoneinfo`). Early-close days are treated conservatively as 16:00.
* A news event counts only if it was **published and analysed** before the cutoff, so late-ingested
  news can never leak into earlier scores.
* An SEC fundamental fact counts only if its **filing date** is on or before the cutoff. A
  restatement filed before the cutoff supersedes the original; one filed later does not.
* Market statistics use only bars up to the last completed session.
* Walk-forward backtests re-run the same functions at each historical cutoff. News components (N, P)
  are reported as unavailable for historical dates because no point-in-time news archive exists
  before ingestion started; they are never back-filled with today's knowledge.

## Prices and dividends

Divergence is measured on **total-return** prices (adjusted for splits and dividends) when the
provider supplies them (Tiingo adjusted close). The free Stooq fallback is split-adjusted, but its
dividend treatment is not guaranteed, so its series are flagged as price-return. Each stored observation records
its provider and an `is_total_return` flag; snapshots and outcomes carry that flag so price-return-only
results can be separated in research.

When outcomes are evaluated, both endpoints (base date *t* and *t+H*) are read from the **current**
adjusted series. Dividend re-basing multiplies every historical adjusted close by the same factor,
so it cancels in the ratio PA(t+H)/PA(t); reading both points from one series keeps the measure
consistent even after the provider re-bases its history.

## SI-v1 (market-only benchmark)

SI-v1 = (25·C + 15·D + 25·R) × 100 / 65, version `1.0`, ported line for line from the original
`run_experiment.py` (`index_components` and `calculate_rows`).

| Component | Definition (daily log total returns) |
|---|---|
| C — correlation weakness | clip((0.90 − ρ₁ᵧ) / 0.90, 0, 1), ρ₁ᵧ over the last 252 sessions |
| D — correlation deterioration | clip((ρ₅ᵧ − ρ₁ᵧ) / 0.30, 0, 1), ρ₅ᵧ over the last 1260 sessions |
| R — past return divergence | clip(\|100·(PA_t/PA_{t−252} − 1) − 100·(PB_t/PB_{t−252} − 1)\| / 50, 0, 1), the one-year total-return gap in percentage points |

As in the original, a score needs the last 1261 closes with at most 10 missing and at least 1250
returns; otherwise SI-v1 is unavailable (never re-weighted). Price history therefore starts in 2009
by default so that the 2015 backtest observations have five years behind them. Backtest observations
are taken on the last trading day of the month (end of January for annual runs, as in the
original). The correlation-only and previous-return-gap baselines use ρ₁ᵧ and the one-year gap.

## SI-v2 (news-enhanced research hypothesis)

SI-v2 = 100 × (w_N·N + w_F·F + w_M·M + w_P·P + w_S·S), default weights
N 0.35, F 0.25, M 0.20, P 0.10, S 0.10. Weights are configurable in Settings and are hypotheses,
not fitted values. Every component lies in [0, 1].

### Missing data

A component that cannot be computed defensibly is **unavailable**, never zero. The score is then
re-normalised over the available weight and marked **provisional** if, and only if:

* the market component M is available (required), and
* the available weight (coverage) is at least 0.60.

Otherwise SI-v2 is unavailable. "No material news" is different from "no news data": N = 0 is
reported only when news collection actually succeeded for both companies on at least 3 of the last
7 days and at least 80% of relevant stories were analysed.

### N — asymmetric news risk

For each analysed event (one Claude interpretation of one story for one pair):

weight = severity × asymmetry × source_reliability × extraction_confidence × exp(−age / τ)

* τ = expected persistence in days, clamped to [7, 365].
* Events count only if material, with at least one quote, and if ≥ 50% of the quoted evidence was
  found verbatim in the source text (deterministic check, `verify_evidence`).
* Deterministic guard on the AI's asymmetry: events scoped `industry_wide` or `both_companies`
  are capped at 0.3; `unrelated` counts 0.
* Repeated coverage of one development (same `thread_key`, linked by Claude to a prior event) counts
  once, using the strongest item.

N = 1 − exp(−Σ weight / 0.5) over the last 180 days. The signed sum (positive favours A) feeds
the directional bias.

### P — event persistence

P = 1 − exp(−Σ weight × π / 0.5), where π = clip(persistence_days / 365, 0, 1), floored at 0.5
for structural event types (M&A, executive change, restructuring, regulatory change, market-share
change, litigation).

### M — market relationship deterioration

Mean of the available sub-features (at least 2 required):

* correlation deterioration: clip((ρ₅₀₄ − ρ₆₃) / 0.5, 0, 1)
* out-of-sample spread dislocation: hedge ratio fitted on a 252-session formation window that ends
  before the 63-session evaluation window; clip(\|z_last\| / 3, 0, 1)
* cointegration breakdown: Engle–Granger p-value over 252 sessions, clip((p − 0.05) / 0.45, 0, 1)

### F — fundamental divergence

Point-in-time SEC XBRL facts (revenue, operating income, diluted EPS) for the latest period and the
same period a year earlier, filed within 200 days of the cutoff. Mean of available sub-features (at
least 2): |Δ revenue growth| / 0.20, |Δ operating-margin change| / 0.05, |Δ EPS growth| / 0.50,
each clipped to [0, 1]. The signed differences feed the directional bias.

### S — structural exposure differences

Mean of available sub-features (at least 2): SIC code distance (same 4-digit 0, same 3-digit 1/3,
same 2-digit 2/3, else 1), |β_A − β_B| / 0.5 versus SPY over 504 sessions, |ln(vol_A/vol_B)| / ln 2
over 252 sessions, |ln(revenue_A/revenue_B)| / ln 10.

### Directional bias

Combines the news signal tanh(signed N mass / 0.5) (weight 0.6) and the signed fundamental
difference (weight 0.4). If both are strong (|x| ≥ 0.2) and disagree, the bias is **uncertain**.
|combined| < 0.15 is **neutral**. Recent price moves are deliberately not used.

### Evidence confidence

Weighted mean of available parts: source credibility (0.20), news coverage (0.15), source
agreement — multiple publishers or an official source (0.15), data completeness = SI-v2 coverage
(0.20), AI certainty × verified-quote ratio (0.15), and freshness of prices and news (0.15).

## Versioning

* Each SI-v2 definition (weights + all parameters) is hashed; its version is `2.0+<hash>`.
  Changing weights in Settings creates a new immutable model version. Stored scores keep the version
  they were computed with.
* Each Claude interpretation stores model id, prompt version (`news-v1.0`) and schema version.
* Prediction targets are versioned `div1-H{H}-T{threshold}`.

## Forward-tested predictions

Every trading day after the close (18:30 New York) the worker writes an **immutable** snapshot per
active pair and horizon: SI-v1, SI-v2 (+ status, model and weight versions), components, confidence,
direction, baselines (correlation-only, previous-return-gap), base date, base prices, provider and
total-return flag. Database triggers reject any UPDATE or DELETE of snapshots, outcomes, scores and
component calculations.

When a horizon matures, the outcome is evaluated:

D(t, H) = | ln(PA(t+H)/PA(t)) − ln(PB(t+H)/PB(t)) |

on total-return prices, with H counted in NYSE sessions. The event "diverged" is D ≥ threshold.

| Horizon (sessions) | 30 | 90 | 180 | 252 | 504 |
|---|---|---|---|---|---|
| Default threshold | 0.049 | 0.085 | 0.120 | 0.141 | **0.200** |

The two-year (504) threshold 0.20 reproduces the original experiment. Shorter horizons scale by
√(H/504), the dispersion of a random-walk log-return gap; per-horizon overrides are configurable.
Unresolved predictions are never counted as correct or incorrect.

Two hypothetical dollar-neutral paper trades are recorded with each outcome (research only; no
costs, borrow fees or execution): a convergence trade when the out-of-sample spread |z| ≥ 2 (short the
rich leg, long the cheap leg), and a directional trade following the bias when it is A- or
B-advantaged.

## Evaluation

The performance page and PDF report compute, per horizon: ROC-AUC with industry-clustered bootstrap
95% intervals, PR-AUC, Brier score (on score/100, labelled as a pseudo-probability because the
indices are not calibrated), calibration by decile, divergence rate by score quartile, precision and
recall at alert thresholds, Spearman correlation with divergence magnitude, results by industry,
threshold sensitivity, and comparisons against correlation-only and previous-return-gap baselines.

Live forward results, walk-forward backtests and the originally reported results are always shown
separately.

### Walk-forward backtest

Train 2015–2019, embargo 2020–2021 (no observations whose labels overlap the test period), test
2022–2024. SI-v1 and a partial SI-v2 (F, M, S re-weighted over available weight) are computed at each
observation date with the point-in-time rules above. An alternative weight set is grid-searched on
the training period only and reported on the test period, together with ablations (without F,
without S) and alternative weights (equal, market-only).

### Reported benchmark (not reproduced here)

The original experiment (100 pairs, 300 observations, 2015–2019 train, 2020–2021 embargo,
2022–2024 evaluation, target |cumulative log total-return gap| ≥ 0.20 over two years) reported:
SI-v1 ROC-AUC 0.6655, correlation-only 0.6596, previous-return-gap 0.6180, top-quartile divergence
74.7% vs bottom-quartile 33.3%. SI-v1 did not anticipate the KO/PEP divergence from the January 2024
observation. These numbers are stored as reported and are not evidence that SI-v2 works.

## Claude news analysis

* Source facts (URL, publisher, timestamps, full text, filing form and items) are stored first.
* Claude produces a separate structured interpretation validated by Pydantic
  (`EventInterpretation`). Article text is passed as untrusted data inside delimiters and the system
  prompt instructs the model never to follow instructions found in it; delimiter-like text inside
  articles is neutralised.
* Quotes are verified deterministically against the source; unverified evidence lowers or removes
  the event's weight.
* Scoring is deterministic code; Claude never produces a score.
* Each call is logged with tokens and cost; a monthly budget is checked **before** each call.
