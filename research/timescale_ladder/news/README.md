# News sentiment on flagged 8% days

The flag model (experiments/strategies/strats/flag_model.json) tells *how much* a stock will move, not which way.
This tests whether the news before a flagged day adds the direction.

1. `flagged.py`: top 2 flagged S&P 500 stocks per day from 2019 (point-in-time members), with the next day's return.
2. `collect.py` (GitHub Actions, `news-flags.yml`): Google News headlines from the 3 days before the flagged day's
   16:00 New York close, name and ticker masked, scored by Claude Haiku (sentiment, lean, earnings soon) and a word list.
3. `evaluate.py`: on flagged days that moved 8%+, does the score separate up from down (AUC), and how often would a
   call-or-put tilt have picked the right side; split 2019-2023 vs 2024+ because the model may remember old news.

Data and results: release `news-flags-data` (scores.jsonl, results.json). Paper research only.
