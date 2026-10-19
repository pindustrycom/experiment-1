# Experiment 7: VWAP + 9 EMA cross

From an Instagram reel (a Roboquant ad): on a 3-minute chart, trade when the 9 EMA crosses the session VWAP. Paper
research only. The rule and its variants are in `exp7/strategy.py`, the test grid in `exp7/backtest.py`.

Run: `python -m exp7 backtest --data data --out results` (with `TIINGO_API_KEY` set it first fetches missing
1-minute bars). The workflow `.github/workflows/exp7.yml` does this on GitHub Actions and caches the bars in the
`exp7-data` release.
