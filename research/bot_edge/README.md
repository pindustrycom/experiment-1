# Do open-source trading bots beat random? (2026-11-17)

Research only, paper only. No exchange keys, no live orders, nothing deployed.

Joaquin asked to download the popular open-source bots (freqtrade, Superalgos, hummingbot, Jesse,
huseinzol05/Stock-Prediction-Models, with ccxt as the data/exchange layer), backtest them on our data and measure
whether any has a real edge or is just luck: at minimum it has to beat random. PR #22 (2026-10-21) had hand-ported 9
freqtrade and 7 Jesse rules; this study runs the bots' own code wherever it can and adds every other repo listed plus
the most popular freqtrade strategy (NostalgiaForInfinity).

**Short answer: no bot passes.** __SUMMARY__

## What was tested

| repo (verified original) | stars | how it was run | what |
|---|---|---|---|
| freqtrade/freqtrade + freqtrade/freqtrade-strategies @f3340ce | 57k / 5.7k | **freqtrade 2026.10's own backtester**, offline | all 62 community strategies (incl. `futures/` and `lookahead_bias/`) + the template `SampleStrategy` |
| iterativv/NostalgiaForInfinity @3243cbe | 3.5k | freqtrade's own backtester | X7 v17.5.266 (current), plus X4/X5/X6 exactly as they stood on 2023-12-30, 2024-12-30, 2025-07-01 |
| hummingbot/hummingbot | 21k | port (its engine needs live connectors) | V2 directional controllers `bollinger_v1`, `macd_bb_v1`, `supertrend_v1` with the base-class triple barrier (3% stop, 2% target, 45 min limit, 5 min cooldown, 3m candles) |
| Superalgos/Superalgos (+ Algorithmic-Trading-Plugins) | 5.8k | approximate port of the JSON trading systems | Weak Hands Buster, BB Top Bounce (the two community BTC/USDT systems) |
| jesse-ai/jesse 3.6.1 + jesse-ai/example-strategies | 8.8k | **Jesse's own indicator functions** + Jesse order flow | KDJstrategy, MAGen, TradingView_RSI (the 7 others were done in PR #22) |
| huseinzol05/Stock-Prediction-Models | 9.5k | port of the notebooks | turtle, moving-average, signal-rolling and evolution-strategy agents |
| je-suis-tm/quant-trading | 11k | port | Awesome Oscillator, Parabolic SAR, Heikin-Ashi, RSI pattern, Bollinger W (Dual Thrust was in PR #22) |
| kernc/backtesting.py | 9.3k | port | the README's `SmaCross` |
| ccxt/ccxt | 45k | library, not a strategy | freqtrade uses it; markets are supplied offline (see `ftrun.py`) |

All repos were checked on GitHub as non-fork originals with their real star counts and history (2026-11-17).
Not tested: hummingbot's market-making controllers (need order-book data we don't have; bar-touch fills would flatter
them), Superalgos' Keltner/WEMA systems (their custom indicators are not defined in the files), the 19 other
huseinzol05 agents (same in-sample design as the ES agent; see below), OctoBot (evaluator ensembles, no fixed rules).

## Data and costs

- **Sandbox runs:** BTC 1h 2018-2026-10 (`btc-ssm-data`), ETH 1h 2019-2026-11 (`kronos-results`), SPY/QQQ Tiingo
  1-minute regular hours 2019-2026-10 (`exp7-data`), resampled to every timeframe a strategy asks for (1m to 1d).
  Crypto strategies written for 5m/15m ran on 1h here.
- **CI runs** (`.github/workflows/bot-edge.yml`): Binance spot 5-minute bars from data.binance.vision for BTC/ETH
  (2018-10 on) and a 12-coin basket (BTC ETH SOL XRP ADA DOGE LTC LINK AVAX DOT TRX ATOM, 2022-09 on), so every 5m/15m/30m
  strategy also ran at its own timeframe, and NostalgiaForInfinity ran as designed (5m, 6 open trades).
- Costs per side (Sharma ch. 9): Binance spot 10 bp, perpetuals 5 bp (zero funding: an assumption), SPY/QQQ 2 bp.
  Stock bars are not dividend-adjusted; the buy-and-hold benchmark uses the same bars.
- Sample 2019-01-01 to the end of the data unless stated; one position, full stake (freqtrade `unlimited`, 1 slot).

## How "edge vs luck" was measured

`evalkit.py`, the same for every row:

1. **Random-timing test.** The strategy's own position series is shifted in time by every possible offset (at least
   20 days away). A shift keeps exposure, number of trades, holding times and costs, and only destroys the timing.
   p = share of shifts that earn at least as much (gross, close to close); z = distance from the shifted mean in
   standard deviations. This is the "better than random entries" bar.
2. **Multiple testing.** Holm across all tested rows (337 here). The project-wide bar from past work is t ~3.8
   (Bonferroni over ~300 strategies tried, see [[edge-tests]]).
3. **After costs vs buy-and-hold.** Net Sharpe above the asset's buy-and-hold Sharpe, Newey-West alpha t above 2.
4. **Both halves.** Better than random in 2019-2022 and in 2023-2026 separately.

A row counts as an **EDGE** only if it passes all four. Rows with fewer than 10 trades are not tested.

## Results

__RESULTS__

## Reading it

__READING__

## Files

| file | what |
|---|---|
| `ftrun.py` | runs freqtrade's own CLI offline (static market table instead of an exchange call; Binance answers US runners with 451) |
| `ft_batch.py` | one freqtrade backtest per strategy and market, own timeframe where we have bars |
| `ft_score.py` | replays freqtrade's exported trades on its bars and scores them (CAGR matches freqtrade's total profit) |
| `basket_score.py` | scores a multi-coin run from freqtrade's wallet export |
| `ports.py`, `run_ports.py` | hummingbot, Superalgos, huseinzol05, quant-trading, backtesting.py ports and their runs |
| `jesse_ports.py` | Jesse example strategies with `jesse.indicators` |
| `evalkit.py`, `test_evalkit.py` | the random-timing test, Holm, halves, alpha; tests |
| `report.py` | combines all rows into `results/results.csv` and `results/table.md` |
| `prep_data.py` | bars in freqtrade's layout from the project releases or Binance |
| `results/` | every row (`results.csv`) and the summary table |

Raw freqtrade exports from CI: release `bot-edge-results`. Local reproduction:

```
pip install -r requirements.txt
python prep_data.py releases RELEASE_DIR data         # btc-ssm-data, kronos-results, exp7-data extracted here
python ft_batch.py --strategies STRATS_DIR --assets BTC,ETH --fee 0.001 --tfs 1h,4h,12h,1d --datadir data --out out
python ft_score.py data rows_ft.csv out
python run_ports.py data rows_ports.csv               # (jesse_ports.py needs `pip install jesse`)
python report.py results rows_*.csv
```

References: Sharma, *The Quantamental Revolution*, ch. 9 (costs) and ch. 8 (decay after publication); Coqueret & Guida
ch. 10 and 12 (walk-forward validation, backtest overfitting, multiple testing), ch. 16 (RL agents).
