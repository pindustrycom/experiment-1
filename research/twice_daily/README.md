# Running the pipeline twice a day (2026-10-24)

**Cost:** news AI (Haiku) about $0.10 a run (Oct 23: 129 calls, $0.097). A second run only analyses stories that arrived since the first, so the monthly total barely moves (about $2-3 vs the $50 cap). Tiingo 22 calls, Finnhub 20, SEC 69 per run: far under limits. The repo is public, so Actions minutes are free; the full run takes about 13 min.

**Exp1 pairs, close only vs close + 12:30 rebalance** (2019-01 to 2026-10-15, 10-min bars, 1bp/side, `midday.py`):
- Close only: -0.61%/yr net, Sharpe -0.02
- + 11:00: -0.55%/yr; + 12:30: -0.80%/yr; + 14:30: -0.49%/yr. Noise; turnover +30%.

**Weekly reversal proxy (5 worst of Exp1's 21 stocks), buy Friday midday instead of Friday close** (`rev.py`):
- Close +19.8%/yr; 12:30 +26.9% (+0.11%/week, t=2.2); 11:00 +0.07%/week (t=1.2); 14:30 +0.08%/week (t=1.7).
- Only 21 large caps; needs an S&P 500 check before trusting it. This moves the Friday trade, it does not need a second daily run.

Not tested: Exp2 already runs twice (morning forecast, evening score); Exp7 already trades on minute bars; trend is monthly; earnings drift and insider buying hold 60-120 days, so entering half a day earlier is worth a few bp per trade at most. BTC Donchian is a 4-hour system checked only once a day live, so it is the one pot where more frequent runs fit the design.

## BTC Donchian: how often the trades are placed (`btc_freq.py KLINE_CACHE`, Binance 1h bars, 2019-01 to 2026-10; run once from a temporary workflow)

Signals always from finished 4h bars; a run trades at the hour it happens. 25bp a side (live setting):

| Run schedule | Return/yr | Sharpe | Worst drop |
|---|---|---|---|
| Every 4h (what the pot already credits) | +45.1% | 1.15 | -52.6% |
| Twice a day, every day (06:00 + 18:00 NY) | +41.9% | 1.10 | -47.5% |
| Once a day, every day (18:00 NY) | +38.8% | 1.04 | -47.3% |
| Once a day, weekdays (live cron) | +36.5% | 1.00 | -44.1% |
| Twice a day, weekdays (12:00 + 18:00 NY) | +34.8% | 0.96 | -51.8% |
| BTC hold | +49.6% | 0.98 | -77.2% |

At 10bp the 4h line reproduces the research number (+52.1%, Sharpe 1.28). crypto.py credits the pot with the 4h rule's equity, so the pot shows the 4h line; btc-4h.yml now places (journals) the trades every 4 hours so that is what live trading would do.
