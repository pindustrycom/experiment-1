# Experiment 3: NVDA, an EKF + SSM with sentiment by category

One stock, a fixed split: fit on 2018-2021, freeze, test on 2022-2025. Target: NVDA's log return
over the next 5 sessions. Paper research only.

The daily state measurements (`exp3/features.py`):

| Group | Measurements |
|---|---|
| Price | 1, 5 and 20-session returns, 20-session volatility, distance from the 200-session average |
| Market cap index | NVDA market cap (close x SEC-filed shares, carried through splits) vs its own year, its 20-session change; QQQ and SPY returns |
| Sentiment, per category | last week's average, a long memory (half-life ~63 sessions) and last week's volume |

Sentiment categories: **news** (Google News headlines), **saturation** (what headlines and posts say
about NVIDIA's markets saturating or expanding), **speculation** (Reddit: wallstreetbets, stocks,
investing, StockMarket; Twitter/X has no free history), **brokers** (analyst upgrades, downgrades and
price-target changes from Yahoo Finance). Claude Haiku scores each headline and post once
(`exp3/score.py`); a category with nothing said is unobserved, and the EKF skips it.

Models (`exp3/model.py`): the EKF + SSM from Experiment 2 (`exp3/ekf.py`) with and without the
sentiment measurements, ridge with and without, momentum, drift and no change.

Run: `python -m exp3 run --data data --out results` (needs TIINGO_API_KEY, ANTHROPIC_API_KEY,
SEC_USER_AGENT). The GitHub workflow `.github/workflows/exp3.yml` runs it and caches the collected
data in the `exp3-data` release.
