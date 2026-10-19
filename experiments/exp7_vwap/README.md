# Experiment 7: VWAP + 9 EMA cross

From an Instagram reel (a Roboquant ad): on a 3-minute chart, trade when the 9 EMA crosses the session VWAP. Paper
research only. The rule and its variants are in `exp7/strategy.py`, the test grid in `exp7/backtest.py`.

Run: `python -m exp7 backtest --data data --out results` (with `TIINGO_API_KEY` set it first fetches missing
1-minute bars). The workflow `.github/workflows/exp7.yml` does this on GitHub Actions and caches the bars in the
`exp7-data` release.

Paper tracking: `python -m exp7 live --data data --out site` replays every session since 2026-10-19 with the reel's
rule (SPY, 3-minute bars, always in, 0.3 bp per side) and writes the Exp7 page and `data/daily.csv`. exp7.yml runs it
each weekday after the close and saves the page to the `exp7-state` release; pages.yml publishes it under /exp7/ and
`experiments/hub/portfolio.py` turns the daily results into a separate 1000 CAD pot (not part of the combined one).

Backtest, 2019 to 2026-10, 0.3 bp per side (annual %): SPY 3-minute +6.0 (holding SPY open to close +6.2), QQQ 3-minute
+8.2 (holding +8.3). Strong in 2020 to 2022, flat to negative in 2025 and 2026.
