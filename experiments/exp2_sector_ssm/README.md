# Experiment 2: tech sector state-space model

A pruned S4D (diagonal state-space) model reads a daily state vector for the US tech sector and
predicts the XLK ETF's log return over the next 5 trading sessions, on its own and relative to SPY.
Every prediction is written to a ledger before its outcome exists, scored once it does, and compared
with simple baselines. Paper tracking only: no orders, no money.

Published at `/exp2/` on the project's GitHub Pages site; the full method is on its Method page
(`exp2/site.py`, `method_page`).

## State vector (one row per session, data up to that close only)

* Daily log returns of XLK, 12 large tech companies (AAPL MSFT NVDA AVGO ORCL CRM AMD ADBE CSCO QCOM
  INTC TXN) and related series (SPY, SMH, XLY, XLF, XLE, XLI, XLU, TLT, UUP, GLD), from Tiingo.
* Sector state: momentum, relative strength vs SPY, volatility, drawdown, breadth, dispersion, volume.
* News: Finnhub headlines for the 12 companies, screened for roundups and duplicates, scored by
  Claude Haiku in batches for sector relevance and sentiment; daily tone, smoothed tone, surprise,
  volume, and a flag for days before news collection began. A story counts for the first session
  whose 16:00 New York close follows it.

## Model

Linear encoder, 2 S4D layers (32 channels x 32 complex modes, S4D-Lin init, ZOH), Gaussian head for
mean and variance of both targets. After training, the weakest modes (by kernel L1 bound) are pruned
to 12.5%, 25% or 50% and fine-tuned; the smallest network that validates within tolerance is kept.
Three seeds are averaged. Retrained weekly on all sessions with known outcomes.

Baselines on the same rows: no change, drift, 20-day momentum OLS, ridge on the same inputs.

## Running

```
pip install torch==2.14.1 --index-url https://download.pytorch.org/whl/cpu
pip install -r requirements.txt
python -m pytest -q                       # offline tests, synthetic data
TIINGO_API_KEY=... FINNHUB_API_KEY=... ANTHROPIC_API_KEY=... python -m exp2 run --state state --site site
```

`.github/workflows/exp2.yml` runs this every weekday after the close, saves `state/` and the pages
to the `exp2-state` release, and commits the ledger to the `exp2-ledger` branch. The dashboard
workflow (`pages.yml`) adds the pages to the site. Claude spending is capped at $5 a month
(`EXP2_CLAUDE_BUDGET_USD`).

## Layout

```
exp2/config.py     universe, horizon, model and training settings
exp2/data.py       Tiingo prices, Finnhub news (cached in state/)
exp2/news.py       screening, session assignment, Haiku scoring, news features
exp2/features.py   state vector and targets
exp2/model.py      S4D layer, network, pruning
exp2/train.py      training, ensemble, baselines, prediction
exp2/ledger.py     forward-test ledger and scores
exp2/backtest.py   walk-forward backtest
exp2/site.py       static pages
exp2/cli.py        the daily run
```
