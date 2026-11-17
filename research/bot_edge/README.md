# Do open-source trading bots beat random? (2026-11-17)

Research only, paper only. No exchange keys, no live orders, nothing deployed.

Joaquin asked to download the popular open-source bots (freqtrade, Superalgos, hummingbot, Jesse,
huseinzol05/Stock-Prediction-Models, with ccxt as the data/exchange layer), backtest them on our data and measure
whether any has a real edge or is just luck: at minimum it has to beat random. PR #22 (2026-10-21) had hand-ported 9
freqtrade and 7 Jesse rules; this study runs the bots' own code wherever it can and adds every other repo listed plus
the most popular freqtrade strategy (NostalgiaForInfinity).

**Short answer: no bot passes.** Of 365 strategy x market rows with 10+ trades, 10% beat random timing before
correction (5% would by chance), 5 survive the multiple-testing correction, and none passes every check. The five
that survive Holm are freqtrade scalpers that lose 18-69%/yr after costs, NostalgiaForInfinity X7 on a 12-coin basket
(passes everything, but only on data it was written after), and NFI X5 out of sample (fails the alpha check). The one
popular bot with a real-looking timing signal, NostalgiaForInfinity, shows it out of sample too (older versions run on
later data: X4 and X5 beat random, X6 does not), but it earns 4-13%/yr with tiny exposure and half its profit comes from
5 crash days. Nothing here beats buying and holding, and nothing is worth a paper pot over the existing BTC Donchian.

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

Tested strategy x market rows (10+ trades): 365

- raw p_random < 0.05 (beats random timing before any correction): 38 (10%; chance alone gives ~5%)
- survive Holm across all 365: 5
- random-timing z >= 3.8 (project-wide bar): 5
- net Sharpe above buy-and-hold: 20
- net CAGR above zero: 151; above buy-and-hold CAGR: 11
- EDGE (all checks): 0 (plus 1 that pass only in-sample, not counted)

| source | rows | median net CAGR | median gross CAGR | raw p<0.05 | Sharpe > hold | median win rate |
|---|---|---|---|---|---|---|
| NostalgiaForInfinity | 1 | +3.5% | +3.9% | 0 | 0 | 100% |
| NostalgiaForInfinity basket (X7, in-sample) | 1 | +12.0% | n/a | 1 | 1 | 99% |
| NostalgiaForInfinity basket (out of sample) | 3 | +3.8% | n/a | 3 | 3 | 98% |
| Superalgos | 4 | -1.8% | +0.7% | 2 | 0 | 37% |
| backtesting.py | 8 | -0.5% | +7.5% | 3 | 0 | 37% |
| freqtrade | 250 | -4.3% | +3.4% | 21 | 7 | 66% |
| freqtrade (shorts allowed) | 24 | -8.6% | +4.5% | 3 | 1 | 51% |
| hummingbot | 12 | -21.0% | -1.0% | 1 | 0 | 47% |
| huseinzol05 | 32 | +8.8% | +13.7% | 1 | 6 | 59% |
| jesse example-strategies | 12 | +0.9% | +4.3% | 1 | 0 | 50% |
| quant-trading | 18 | +8.2% | +8.5% | 2 | 2 | 50% |

Top 25 by random-timing z (all rows):

| source | strategy | asset | tf | trades | net CAGR | gross | Sharpe | hold Sharpe | z vs random | p | Holm p | z 1st half | z 2nd half | alpha t | verdict |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| NostalgiaForInfinity basket (X7, in-sample) | NostalgiaForInfinityX7 | basket of 12 | 5m | 149 | +12.0% | n/a | 2.12 | 0.66 | +7.8 | 0.000 | 0.02 | +4.9 | +6.6 | +3.6 | passes in-sample only (not counted) |
| NostalgiaForInfinity basket (out of sample) | NostalgiaForInfinityX4 | basket of 12 | 5m | 35 | +3.8% | n/a | 1.61 | 0.26 | +5.0 | 0.000 | 0.18 | +6.2 | +1.2 | +2.6 | beats random (raw p) but fails a later check |
| freqtrade | CofiBitStrategy | BTC | 5m | 5370 | -68.7% | +23.2% | -7.28 | 0.95 | +4.0 | 0.000 | 0.01 | +3.1 | +2.8 | -20.5 | beats random (raw p) but fails a later check |
| freqtrade | Scalp | BTC | 5m | 4999 | -65.8% | +22.6% | -8.07 | 0.95 | +4.0 | 0.000 | 0.02 | +2.1 | +4.2 | -23.0 | beats random (raw p) but fails a later check |
| freqtrade | Scalp | SPY | 1m | 5410 | -18.3% | +6.7% | -3.61 | 0.81 | +3.9 | 0.000 | 0.04 | +3.7 | +1.3 | -10.1 | beats random (raw p) but fails a later check |
| NostalgiaForInfinity basket (out of sample) | NostalgiaForInfinityX5 | basket of 12 | 5m | 60 | +13.4% | n/a | 0.93 | -0.30 | +3.7 | 0.000 | 0.04 | +3.7 | +2.3 | +1.7 | beats random (raw p) but fails a later check |
| freqtrade (shorts allowed) | VolatilitySystem | ETH | 1h | 195 | +185.4% | n/a | 1.13 | 0.83 | +3.2 | 0.001 | 0.18 | +3.3 | +0.8 | +3.2 | beats random (raw p) but fails a later check |
| backtesting.py | SmaCross 10/20 (README) | ETH | 1h | 3887 | -30.0% | +88.0% | -0.07 | 0.83 | +3.2 | 0.001 | 0.21 | +2.8 | +1.4 | -0.3 | beats random (raw p) but fails a later check |
| freqtrade | CofiBitStrategy | ETH | 5m | 5629 | -68.8% | +31.4% | -5.60 | 0.86 | +3.0 | 0.001 | 0.45 | +3.8 | -0.4 | -13.9 | beats random (raw p) but fails a later check |
| freqtrade | EMASkipPump | BTC | 5m | 6540 | -68.7% | +66.1% | -2.98 | 0.95 | +2.9 | 0.002 | 0.66 | +3.1 | -0.5 | -11.7 | beats random (raw p) but fails a later check |
| freqtrade | Scalp | QQQ | 1m | 5522 | -19.2% | +6.3% | -3.38 | 0.93 | +2.9 | 0.002 | 0.76 | +4.2 | -0.8 | -9.2 | beats random (raw p) but fails a later check |
| freqtrade | ReinforcedQuickie | SPY | 5m | 695 | +7.2% | +10.9% | 0.88 | 0.81 | +2.6 | 0.005 | 1.00 | +3.1 | +0.1 | +1.7 | beats random (raw p) but fails a later check |
| backtesting.py | SmaCross 10/20 (README) | BTC | 1h | 3977 | -51.1% | +35.0% | -0.87 | 0.95 | +2.5 | 0.007 | 1.00 | +2.5 | +0.8 | -2.6 | beats random (raw p) but fails a later check |
| Superalgos | Weak Hands Buster | BTC | 1h | 223 | -3.2% | +2.5% | 0.07 | 0.95 | +2.4 | 0.007 | 1.00 | +2.1 | +1.0 | +2.3 | beats random (raw p) but fails a later check |
| Superalgos | Weak Hands Buster | ETH | 1h | 320 | -6.6% | +1.3% | 0.03 | 0.83 | +2.4 | 0.006 | 1.00 | +1.8 | +1.5 | +1.7 | beats random (raw p) but fails a later check |
| freqtrade | MultiMa | ETH | 4h | 52 | +11.7% | +13.2% | 0.68 | 0.83 | +2.3 | 0.009 | 1.00 | +1.3 | +2.2 | +1.5 | beats random (raw p) but fails a later check |
| freqtrade | ReinforcedAverageStrategy | BTC | 4h | 212 | +34.6% | +42.1% | 1.10 | 0.95 | +2.3 | 0.016 | 1.00 | +2.5 | +0.7 | +2.0 | beats random (raw p) but fails a later check |
| freqtrade (shorts allowed) | VolatilitySystem | BTC | 1h | 240 | +56.2% | n/a | 0.83 | 0.95 | +2.3 | 0.008 | 1.00 | +2.1 | +1.4 | +2.3 | beats random (raw p) but fails a later check |
| backtesting.py | SmaCross 10/20 (README) | ETH | 1d | 155 | +34.5% | +39.9% | 0.77 | 0.82 | +2.3 | 0.006 | 1.00 | +2.5 | +0.3 | +1.8 | beats random (raw p) but fails a later check |
| freqtrade | CCIStrategy | QQQ | 1m | 142 | +5.4% | +6.2% | 1.08 | 0.93 | +2.2 | 0.012 | 1.00 | +0.5 | +2.8 | +2.6 | beats random (raw p) but fails a later check |
| freqtrade | Scalp | ETH | 5m | 5140 | -68.7% | +16.2% | -6.20 | 0.86 | +2.2 | 0.016 | 1.00 | +2.3 | +0.4 | -17.9 | beats random (raw p) but fails a later check |
| jesse example-strategies | MAGen | SPY | 4h | 172 | +6.5% | +7.4% | 0.49 | 0.80 | +2.1 | 0.009 | 1.00 | +1.8 | +1.4 | +1.1 | beats random (raw p) but fails a later check |
| quant-trading | awesome_oscillator | ETH | 1d | 52 | +56.9% | +59.0% | 1.08 | 0.82 | +2.1 | 0.012 | 1.00 | +1.8 | +1.1 | +2.0 | beats random (raw p) but fails a later check |
| freqtrade | MultiRSI | ETH | 5m | 4603 | -57.4% | +37.9% | -3.10 | 0.86 | +2.1 | 0.018 | 1.00 | +2.2 | +0.5 | -11.6 | beats random (raw p) but fails a later check |
| freqtrade | SmoothScalp | BTC | 5m | 1018 | -12.8% | +13.1% | -0.94 | 0.95 | +2.1 | 0.019 | 1.00 | +1.3 | +1.8 | -3.2 | beats random (raw p) but fails a later check |

Halves are 2019-2022 / 2023-2026 except the NFI basket rows, which split their own (shorter) sample at its midpoint.
Gross CAGR is n/a where the run was scored from freqtrade's wallet (baskets, leverage, position adjustment).
Every row: `results/results.csv`.

### NostalgiaForInfinity, the one that looks real

| version (written) | tested on | trades | net CAGR | Sharpe | max DD | basket hold | z vs random | Holm p | alpha t | top-5 days share |
|---|---|---|---|---|---|---|---|---|---|---|
| X7 v17.5 (2025-26) | 2023-01 to 2026-10, in-sample | 149 | +12.0% | 2.12 | -3.2% | +24.1%/yr, Sharpe 0.66, DD -69% | 7.8 | 0.02 | 3.6 | 46% |
| X4 as of 2023-12-30 | 2024-01 to 2026-10 | 35 | +3.8% | 1.61 | -0.3% | -3.2%/yr | 5.0 | 0.18 | 2.6 | 73% |
| X5 as of 2024-12-30 | 2025-01 to 2026-10 | 60 | +13.4% | 0.93 | -10.8% | -31.0%/yr | 3.7 | 0.04 | 1.7 | 49% |
| X6 as of 2025-07-01 | 2025-07 to 2026-10 | 35 | -1.5% | -0.05 | -10.6% | -23.2%/yr | 2.0 | 1.00 | 0.1 | n/a (lost) |

NFI buys deep dips across many coins with DCA rebuys and almost never sells at a loss (win rate 89-99%; one X7 ATOM
trade stayed open 963 days). Its timing beats random because it buys after crashes (2025-10-10, 2025-02-03, 2024-04-13)
and sells the bounce. Out of sample that held up in 2 of 3 versions, but the money is small: X4 was invested about 0.1%
of the time and made less than T-bills; X5's +13% came in a falling market with alpha t 1.7; X6 lost. Single-pair NFI
(as people often run it) opened 3 trades on BTC and 15 on ETH in 8 years.


## Reading it

- **Win rates mean nothing.** freqtrade strategies win 66% of trades (median) and 66% of their rows still lose money net;
  they take many small gains and a few large losses, which is what a tight ROI table plus a wide stop produces.
- **Gross edges die at the fee.** The 5-minute freqtrade scalpers beat random timing gross (CofiBit BTC 5m +23%/yr gross,
  z 4.0) and lose 66-69%/yr at Binance's 10 bp per side; the 57 BTC/ETH 5m rows go from a median +4.3%/yr gross to
  -26.7% net, and 84% of them lose.
  backtesting.py's own `SmaCross` README example makes +88%/yr gross on ETH 1h and -30% net.
- **Repo claims don't survive.** Jesse's KDJstrategy README shows +37.6% on 2019-20; here it made +19%/yr against
  +47.6% for holding BTC. huseinzol05's evolution-strategy agent trained once and tested on 2023+ BTC ranges from +1.6% to
  +63.7%/yr depending only on the random seed, and no seed beats random timing.
- **hummingbot's V2 directional controllers lose** (bollinger_v1 BTC -52%/yr, supertrend_v1 SPY/QQQ 3m about -50%/yr):
  a 2% target and 3% stop at 3-minute bars is mostly fees. Its market-making side needs order-book data we don't have.
- **Leverage is not edge.** freqtrade's VolatilitySystem (2x, pyramiding, shorts) made +185%/yr on ETH 1h with z 3.2 and
  alpha t 3.2, but fails Holm (p 0.18 after 365 tests), its 2023-26 half is no better than random (z 0.8), and on BTC
  its Sharpe is below holding.
- **freqtrade's `lookahead_bias/` examples** (strategies kept in the repo as warnings) were run too; in freqtrade's
  backtester they did not look good either (wtc, Zeus, DevilStra, GodStraNew: no edge or too few trades).
- Consistent with past work in this project ([[github-bots]], [[edge-tests]]) and with Sharma ch. 8 (published rules
  decay) and Coqueret & Guida ch. 12 (with ~300 tries, a few t ~3 results are expected by chance).

Nothing passes, so no paper pot is proposed. If one were wanted anyway, the only candidate with an out-of-sample timing
signal is NostalgiaForInfinity on a coin basket, and it would mostly sit in cash.

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
