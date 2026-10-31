# News sentiment on flagged 8% days

The flag model (experiments/strategies/strats/flag_model.json) tells *how much* a stock will move, not which way.
This tests whether the news before a flagged day adds the direction.

1. `flagged.py`: top 2 flagged S&P 500 stocks per day from 2019 (point-in-time members), with the next day's return.
2. `collect.py` (GitHub Actions, `news-flags.yml`): Google News headlines from the 3 days before the flagged day's
   16:00 New York close, name and ticker masked, scored by Claude Haiku (sentiment, lean, earnings soon) and a word list.
3. `evaluate.py`: on flagged days that moved 8%+, does the score separate up from down (AUC), and how often would a
   call-or-put tilt have picked the right side; split 2019-2023 vs 2024+ because the model may remember old news.

Data and results: release `news-flags-data` (scores.jsonl, results.json). Paper research only.

## Result (2,858 of 3,932 stock-days scored, Jan 2019 to Oct 2026, the rest retried next run; 273 moved 8%+)

| signal on 8%+ days | AUC up vs down | 2019-2023 | 2024+ |
|---|---|---|---|
| Haiku sentiment | 0.60 (224 days) | 0.59 | 0.62 |
| Haiku lean | 0.61 | 0.61 | 0.60 |
| word list (no memory) | 0.61 (118 days) | 0.61 | 0.61 |
| the stock's own 1-day / 3-day / 20-day return | 0.51 / 0.53 / 0.53 | | |

`controls.py`: with the stock's own 1, 3 and 20-day returns in a logistic, lean keeps z 3.2 and sentiment z 2.9.
On all flagged days, trading the stock by the sign of the lean earns +38 bp/day before costs (t 2.8).
Caveats: Google News dates can be off (a later-updated article can carry an earlier date), and about a quarter of
flagged days have no headlines. Needs a forward test before trusting it.
