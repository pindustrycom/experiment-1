# Do open-source trading bots beat random? (2026-11-17)

Research only, paper only. No exchange keys, no live orders, nothing deployed.

Joaquin asked to download the popular open-source bots (freqtrade, Superalgos, hummingbot, Jesse,
huseinzol05/Stock-Prediction-Models, with ccxt as the data/exchange layer), backtest them on our data and measure
whether any has a real edge or is just luck: at minimum it has to beat random. PR #22 (2026-10-21) had hand-ported 9
freqtrade and 7 Jesse rules; this study runs the bots' own code wherever it can and adds every other repo listed plus
the most popular freqtrade strategy (NostalgiaForInfinity).

**Short answer: one strategy passes, and it is small.** Of 967 strategy x market rows with 10+ trades (three rounds,
about 90 repos, plus an ML trade filter), 11% beat random timing before correction (5% would by chance), 9 survive the
multiple-testing correction, and one passes every check: **ClucHAnix_5m** from ssssi/freqtrade_strs on BTC 5-minute bars
(+13.2%/yr net 2019-2026, Sharpe 1.42 vs 0.95 for holding BTC, max drawdown -7%, alpha t 3.4, beats random in both
halves). It is a dip buyer that is in the market under 1% of the time, so it earns far less than holding (BTC +47%/yr).
Its file was last changed in December 2022, so 2023-2026 is out of sample: 17 trades, +5.8%/yr, Sharpe 1.15 vs 1.11 for
holding. On ETH it makes +9.7%/yr out of sample (Sharpe 1.07 vs 0.51) but fails the full-period checks. A 12-coin basket
run from 2023 is the deciding test (CI part `cluc_basket`).

The rest that survive Holm fail a later check: two freqtrade scalpers that lose 66-69%/yr after costs, NostalgiaForInfinity
X7 on a 12-coin basket (passes everything, but only on data it was written after), the R-Breaker day-trading rule on ETH
(its 2023-26 half is weak and it trails holding ETH), ssssi's BBMod on BTC 5m (+3%/yr), and two myhhub stock screens that
pick better than random S&P stocks yet trail the equal-weight index on return and Sharpe. NostalgiaForInfinity also shows
a timing signal out of sample (X4 and X5 beat random on later data, X6 does not), earning 4-13%/yr with tiny exposure.

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
| **Round 2 (2026-11-18)** | | | |
| askmike/gekko @ff5698f (archived) | 10k | port of the JS strategies, shipped defaults, 1h candles | MACD, PPO, RSI, StochRSI, CCI, DEMA, TMA, TSI, UO (long/cash) |
| DeviaVir/zenbot @52872fb (archived) | 8k | port, shipped defaults, own bar size (2m runs on 2m SPY/QQQ, 5m crypto) | trend_ema (the default), macd, bollinger, sar, trust_distrust |
| vnpy/vnpy + vnpy_ctastrategy @c6e231c | 46k | port of the CTA demo strategies with stop orders | DoubleMa, AtrRsi, BollChannel (15m), KingKeltner (5m), TurtleSignal (1 unit), long/short |
| whittlem/pycryptobot @1fa9aae | 2k | port of the default signal (EMA12/26 cross + MACD + 50/200 golden-cross filter) | 1h |
| Haehnchen/crypto-trading-bot @cb95633 | 3.5k | port, long/short, 15m | macd, awesome_oscillator_cross_zero, cci, parabolicsar |
| ccxt/binance-trade-bot @55f8cbb | 8.8k | port of the ratio-jumping loop (scout_multiplier 5) | 12-coin basket on 5m bars vs equal-weight basket |
| joshyattridge/smart-money-concepts 0.0.27 | 2.2k | **the library itself** | trade the direction of each BOS/CHoCH, acted on only once the swing is confirmed |
| ccxt/ccxt | 45k | library, not a strategy | freqtrade uses it; markets are supplied offline (see `ftrun.py`) |

All repos were checked on GitHub as non-fork originals with their real star counts and history (2026-11-17).
Not tested: hummingbot's market-making controllers (need order-book data we don't have; bar-touch fills would flatter
them), Superalgos' Keltner/WEMA systems (their custom indicators are not defined in the files), the 19 other
huseinzol05 agents (same in-sample design as the ES agent; see below), OctoBot (evaluator ensembles, no fixed rules).
Round 2 skipped: chrisleekr/binance-trading-bot (rewritten in 2026 into gated grid profiles with no fixed default to
port), passivbot and hftbacktest/Krypto-trading-bot (market making / grids that need tick or order-book data),
LLM "hedge fund" agents (need a paid model per bar), and frameworks without strategies of their own (backtrader, Lean,
zipline, vectorbt, whose sample strategies are moving-average crosses already covered).

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

Tested strategy x market rows (10+ trades): 967

- raw p_random < 0.05 (beats random timing before any correction): 110 (11%; chance alone gives ~5%)
- survive Holm across all 967: 9
- random-timing z >= 3.8 (project-wide bar): 17
- net Sharpe above buy-and-hold: 43
- net CAGR above zero: 369; above buy-and-hold CAGR: 19
- EDGE (all checks): 1 (plus 1 that pass only in-sample, not counted)

| source | rows | median net CAGR | median gross CAGR | raw p<0.05 | Sharpe > hold | median win rate |
|---|---|---|---|---|---|---|
| Binance-Futures-Trading-Bot | 18 | -7.6% | +0.2% | 3 | 0 | 60% |
| Binance-volatility-trading-bot | 1 | +7.8% | +8.1% | 1 | 0 | nan% |
| Chan (ep-chan-book) | 1 | +0.9% | +1.0% | 0 | 0 | nan% |
| DCA bot (opentrader) | 6 | +29.9% | +33.2% | 0 | 0 | nan% |
| KidQuant pairs | 2 | -5.9% | -5.4% | 0 | 0 | nan% |
| NostalgiaForInfinity | 1 | +3.5% | +3.9% | 0 | 0 | 100% |
| NostalgiaForInfinity basket (X7, in-sample) | 1 | +12.0% | n/a | 1 | 1 | 99% |
| NostalgiaForInfinity basket (out of sample) | 3 | +3.8% | n/a | 3 | 3 | 98% |
| PatternPy | 16 | +5.3% | +14.3% | 0 | 1 | 49% |
| PyTrendFollow | 4 | +5.8% | +6.1% | 0 | 0 | nan% |
| Superalgos | 4 | -1.8% | +0.7% | 2 | 0 | 37% |
| ai-trader (whchien) | 42 | +12.0% | +12.4% | 8 | 12 | 45% |
| algorithmic-trading-python (freeCodeCamp) | 2 | +11.2% | +12.2% | 0 | 0 | nan% |
| alpaca example-scalping | 2 | -71.5% | +14.8% | 0 | 0 | 14% |
| backtesting.py | 8 | -0.5% | +7.5% | 3 | 0 | 37% |
| backtesting.py + ML filter | 4 | -4.8% | +2.8% | 0 | 0 | 37% |
| binance-trade-bot | 1 | -31.1% | -30.5% | 0 | 0 | nan% |
| crypto-trading-bot | 24 | -9.0% | +1.0% | 3 | 0 | 34% |
| crypto-trading-bot + ML filter | 12 | -2.0% | +0.8% | 2 | 1 | 39% |
| fmzquant | 9 | +7.0% | +18.5% | 5 | 0 | 40% |
| freqtrade | 250 | -4.3% | +3.4% | 21 | 7 | 66% |
| freqtrade (shorts allowed) | 24 | -8.6% | +4.5% | 3 | 1 | 51% |
| freqtrade: nateemma/strategies | 230 | -10.7% | +6.3% | 13 | 4 | 72% |
| freqtrade: ssssi/freqtrade_strs | 29 | -0.9% | +6.3% | 8 | 1 | 65% |
| gekko | 36 | +3.2% | +7.2% | 6 | 0 | 61% |
| gekko + ML filter | 12 | +0.9% | +6.0% | 3 | 1 | 43% |
| gekko-strategies | 28 | -1.8% | +20.1% | 5 | 2 | 51% |
| grid bot (51bitquant/opentrader) | 6 | -12.0% | +5.7% | 0 | 0 | nan% |
| hummingbot | 12 | -21.0% | -1.0% | 1 | 0 | 47% |
| huseinzol05 | 32 | +8.8% | +13.7% | 1 | 6 | 59% |
| jesse example-strategies | 12 | +0.9% | +4.3% | 1 | 0 | 50% |
| myhhub/stock (instock) | 5 | +9.6% | +11.1% | 2 | 0 | nan% |
| pskrunner14/trading-bot | 3 | +7.5% | +9.5% | 1 | 1 | 64% |
| pycryptobot | 4 | +4.6% | +10.4% | 1 | 0 | 38% |
| quant-trading | 18 | +8.2% | +8.5% | 2 | 2 | 50% |
| smart-money-concepts | 5 | +5.9% | +6.0% | 0 | 0 | 38% |
| supertrend-crypto-bot | 6 | -34.3% | +9.6% | 0 | 0 | 31% |
| vnpy | 30 | -53.8% | +9.2% | 5 | 0 | 34% |
| vnpy + ML filter | 24 | -1.4% | +2.1% | 3 | 0 | 36% |
| zenbot | 24 | -18.1% | +12.0% | 1 | 0 | 30% |
| zenbot + ML filter | 16 | -0.1% | +4.5% | 2 | 0 | 38% |

Top 25 by random-timing z (all rows):

| source | strategy | asset | tf | trades | net CAGR | gross | Sharpe | hold Sharpe | z vs random | p | Holm p | z 2019-22 | z 2023-26 | alpha t | verdict |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| freqtrade: ssssi/freqtrade_strs | BBMod | BTC | 5m | 377 | +3.2% | +13.7% | 0.33 | 0.95 | +11.0 | 0.000 | 0.00 | +7.0 | +11.1 | +0.5 | beats random (raw p) but fails a later check |
| NostalgiaForInfinity basket (X7, in-sample) | NostalgiaForInfinityX7 | basket of 12 | 5m | 149 | +12.0% | n/a | 2.12 | 0.66 | +7.8 | 0.000 | 0.05 | +4.9 | +6.6 | +3.6 | passes in-sample only (not counted) |
| Binance-volatility-trading-bot | +10% jump buy, TP 0.8% trailing 0.4/0.1, SL 5% | basket of 12 | 5m | 46 | +7.8% | +8.1% | 0.50 | 0.54 | +7.3 | 0.000 | 0.18 | +11.4 | +1.0 | +1.0 | beats random (raw p) but fails a later check |
| freqtrade: ssssi/freqtrade_strs | ClucHAnix_5m | BTC | 5m | 119 | +13.2% | +16.7% | 1.42 | 0.95 | +6.8 | 0.000 | 0.00 | +5.0 | +4.8 | +3.4 | EDGE |
| myhhub/stock (instock) | keep_increasing (rising MA30, +20%) | S&P 500 (point-in-time) | 1d | 48331 | +11.5% | +13.3% | 0.49 | 0.70 | +5.8 | 0.000 | 0.00 | +7.7 | -0.1 | +0.1 | beats random (raw p) but fails a later check |
| myhhub/stock (instock) | low_backtrace_increase (+60% in 60d, no deep drop) | S&P 500 (point-in-time) | 1d | 4108 | +12.2% | +13.4% | 0.52 | 0.70 | +5.4 | 0.000 | 0.00 | +2.6 | +4.5 | +1.6 | beats random (raw p) but fails a later check |
| NostalgiaForInfinity basket (out of sample) | NostalgiaForInfinityX4 | basket of 12 | 5m | 35 | +3.8% | n/a | 1.61 | 0.26 | +5.0 | 0.000 | 0.47 | +6.2 | +1.2 | +2.6 | beats random (raw p) but fails a later check |
| fmzquant | R-Breaker (0.35/0.07/0.25, intraday) | ETH | 1h | 1016 | +22.8% | +39.7% | 0.74 | 0.83 | +4.6 | 0.000 | 0.01 | +4.6 | +1.3 | +1.9 | beats random (raw p) but fails a later check |
| freqtrade: ssssi/freqtrade_strs | E0V1E | BTC | 1h | 81 | +12.1% | +14.5% | 0.87 | 0.95 | +4.5 | 0.000 | 0.10 | +3.2 | +3.5 | +1.8 | beats random (raw p) but fails a later check |
| zenbot + ML filter | sar | BTC | 5m | 114 | +0.9% | +4.9% | 0.17 | 0.58 | +4.1 | 0.000 | 0.16 | +2.5 | +3.3 | +0.4 | beats random (raw p) but fails a later check |
| freqtrade: ssssi/freqtrade_strs | ClucHAnix | BTC | 5m | 252 | +7.5% | +14.7% | 0.52 | 0.95 | +4.1 | 0.000 | 0.07 | +1.7 | +7.2 | +0.9 | beats random (raw p) but fails a later check |
| vnpy + ML filter | DoubleMaStrategy | ETH | 5m | 109 | +5.1% | +7.1% | 0.57 | 0.63 | +4.1 | 0.000 | 0.16 | +2.0 | +4.1 | +1.3 | beats random (raw p) but fails a later check |
| freqtrade | CofiBitStrategy | BTC | 5m | 5370 | -68.7% | +23.2% | -7.28 | 0.95 | +4.0 | 0.000 | 0.03 | +3.1 | +2.8 | -20.5 | beats random (raw p) but fails a later check |
| freqtrade | Scalp | BTC | 5m | 4999 | -65.8% | +22.6% | -8.07 | 0.95 | +4.0 | 0.000 | 0.05 | +2.1 | +4.2 | -23.0 | beats random (raw p) but fails a later check |
| fmzquant | R-Breaker (0.35/0.07/0.25, intraday) | ETH | 5m | 1690 | +7.0% | +32.8% | 0.37 | 0.86 | +3.9 | 0.000 | 0.01 | +4.2 | +0.7 | +0.9 | beats random (raw p) but fails a later check |
| freqtrade | Scalp | SPY | 1m | 5410 | -18.3% | +6.7% | -3.61 | 0.81 | +3.9 | 0.000 | 0.10 | +3.7 | +1.3 | -10.1 | beats random (raw p) but fails a later check |
| fmzquant | R-Breaker (0.35/0.07/0.25, intraday) | BTC | 1h | 1100 | +9.1% | +25.6% | 0.44 | 0.95 | +3.9 | 0.000 | 0.10 | +3.5 | +1.7 | +1.1 | beats random (raw p) but fails a later check |
| freqtrade: ssssi/freqtrade_strs | ClucHAnix_5m | ETH | 5m | 217 | +7.2% | +13.3% | 0.48 | 0.86 | +3.8 | 0.000 | 0.18 | +2.8 | +2.4 | +0.9 | beats random (raw p) but fails a later check |
| NostalgiaForInfinity basket (out of sample) | NostalgiaForInfinityX5 | basket of 12 | 5m | 60 | +13.4% | n/a | 0.93 | -0.30 | +3.7 | 0.000 | 0.10 | +3.7 | +2.3 | +1.7 | beats random (raw p) but fails a later check |
| freqtrade (shorts allowed) | VolatilitySystem | ETH | 1h | 195 | +185.4% | n/a | 1.13 | 0.83 | +3.2 | 0.001 | 0.49 | +3.3 | +0.8 | +3.2 | beats random (raw p) but fails a later check |
| gekko-strategies | TEMA (SMA200 safety) | ETH | 5m | 8634 | -83.0% | +54.3% | -5.24 | 0.86 | +3.2 | 0.001 | 0.65 | +2.2 | +2.4 | -17.9 | beats random (raw p) but fails a later check |
| backtesting.py | SmaCross 10/20 (README) | ETH | 1h | 3887 | -30.0% | +88.0% | -0.07 | 0.83 | +3.2 | 0.001 | 0.56 | +2.8 | +1.4 | -0.3 | beats random (raw p) but fails a later check |
| vnpy | DoubleMaStrategy | ETH | 1h | 3878 | +14.8% | +87.8% | 0.56 | 0.83 | +3.1 | 0.001 | 0.56 | +2.8 | +1.4 | +1.5 | beats random (raw p) but fails a later check |
| freqtrade | CofiBitStrategy | ETH | 5m | 5629 | -68.8% | +31.4% | -5.60 | 0.86 | +3.0 | 0.001 | 1.00 | +3.8 | -0.4 | -13.9 | beats random (raw p) but fails a later check |
| gekko-strategies | TEMA (SMA200 safety) | ETH | 1h | 662 | +27.3% | +50.6% | 0.89 | 0.83 | +3.0 | 0.001 | 1.00 | +2.6 | +1.4 | +1.8 | beats random (raw p) but fails a later check |

### NostalgiaForInfinity, the one that looks real

| version (written) | tested on | trades | net CAGR | Sharpe | max DD | basket hold | z vs random | Holm p | alpha t | top-5 days share |
|---|---|---|---|---|---|---|---|---|---|---|
| X7 v17.5 (2025-26) | 2023-01 to 2026-10, in-sample | 149 | +12.0% | 2.12 | -3.2% | +24.1%/yr, Sharpe 0.66, DD -69% | 7.8 | 0.02 | 3.6 | 46% |
| X4 as of 2023-12-30 | 2024-01 to 2026-10 | 35 | +3.8% | 1.61 | -0.3% | -3.2%/yr | 5.0 | 0.24 | 2.6 | 73% |
| X5 as of 2024-12-30 | 2025-01 to 2026-10 | 60 | +13.4% | 0.93 | -10.8% | -31.0%/yr | 3.7 | 0.06 | 1.7 | 49% |
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
  alpha t 3.2, but fails Holm after 876 tests, its 2023-26 half is no better than random (z 0.8), and on BTC
  its Sharpe is below holding.
- **freqtrade's `lookahead_bias/` examples** (strategies kept in the repo as warnings) were run too; in freqtrade's
  backtester they did not look good either (wtc, Zeus, DevilStra, GodStraNew: no edge or too few trades).
- Consistent with past work in this project ([[github-bots]], [[edge-tests]]) and with Sharma ch. 8 (published rules
  decay) and Coqueret & Guida ch. 12 (with ~300 tries, a few t ~3 results are expected by chance).

- **Round 2 bots are the same story.** zenbot's default trend_ema makes +67%/yr gross on ETH 5m and -98% net (17,000
  trades); vnpy's Turtle makes +310%/yr gross on ETH 5m and -96% net. On hourly bars the best were vnpy Turtle ETH
  (+40%/yr, Sharpe 0.82 vs 0.83 for holding, z 2.2) and DoubleMa ETH (z 3.1, below holding after costs). gekko's
  oscillators (RSI, CCI, TSI, UO: buy oversold, sell overbought) are worse than random on crypto (z -1.3 to -2.4).
  binance-trade-bot's coin jumping lost 31%/yr while the 12-coin basket made +15%/yr. Smart-money BOS/CHoCH: no edge.

### Round 3 (2026-11-18): about 40 more repos

Joaquin: "keep trying more bots from github, make an extensive list first". 319 new rows with 10+ trades:

| group | repos (own code run where possible) | result |
|---|---|---|
| freqtrade strategy packs | ssssi/freqtrade_strs (ClucHAnix, E0V1E, archive), nateemma/strategies (81 SimpleStrategies + OversoldReversion), in freqtrade's engine | 259 rows; 1h-1d median -2%/yr net, 5m (CI) median -34%/yr net. **ClucHAnix_5m BTC 5m passes every check** (below). BBMod BTC 5m z 11 but +3%/yr; nateemma TEMACross SPY 5m Sharpe 1.03 vs 0.81, fails Holm |
| gekko community | RSI_BULL_BEAR, RSI_BULL_BEAR_ADX, NEO, BBRSI, TEMA | on 5m (own size) +45-69%/yr gross, -16 to -87% net; on 1h TEMA ETH z 3.0, below holding |
| grid / DCA | 51bitquant grid, opentrader grid and DCA | grid loses to fees on every market; DCA +24-40%/yr on crypto but no better than random timing (it is mostly just being long) |
| pairs / mean reversion | KidQuant ratio z-score, Chan Kalman pair (ETH/BTC, QQQ/SPY) | lose or trade too rarely |
| trend following | PyTrendFollow EWMAC (vol targeted), hackingthemarkets supertrend | EWMAC +4-8%/yr, below holding; supertrend -34% to -95% net on 1m/5m |
| patterns | PatternPy head and shoulders, double top/bottom | worse than random on intraday bars, nothing on daily |
| scalpers / futures bots | alpaca example-scalping (+$0.01 target), conor19w Binance futures (3 strategies, TP 1% / SL 1.5%), CyberPunkMetalHead volatility bot | scalper -72%/yr on SPY/QQQ 1m (16,000+ trades a year); futures bot -8% median; volatility bot +7.8%/yr on the basket with only 46 trades |
| fmz classics | R-Breaker, Dual Thrust | R-Breaker ETH 1h +23%/yr, z 4.6, survives Holm, but the 2023-26 half is weak (z 1.3), alpha t 1.9 and Sharpe 0.74 vs 0.83 for holding; on 5m bars it nets +7%/yr |
| ML / RL | pskrunner14 DQN (trained to 2020, tested 2021+), whchien/ai-trader (18 backtrader strategies) | DQN BTC +20%/yr vs +19% hold, z 2.3; ai-trader daily crypto trend rules (CrossSMA, Momentum, Turtle) beat holding on Sharpe (1.1-1.3 vs 0.8-0.95) but none clears Holm; same family as the live BTC Donchian |
| S&P 500 pickers | freeCodeCamp momentum / HQM (top 51 monthly), myhhub/stock screens (turtle, volume surge, keep_increasing, low_backtrace_increase, breakthrough platform), point-in-time S&P 500 2003-2026, against random picks on the same days | momentum = random; keep_increasing and low_backtrace_increase pick better than random (z 5.8, 5.4) but return +11.5-12.2%/yr vs +12.4% for the equal-weight index and Sharpe 0.49-0.52 vs 0.70: they buy volatile stocks, which raises the average day but not the compounded result |

**ClucHAnix_5m, the one pass.** Buys sharp 5-minute dips (Heikin-Ashi close under the lower Bollinger band, or under the slow EMA
and the band), unless the coin is down too far over the past week (1h rate-of-change filter); exits on a trailing stop
or a Fisher-RSI sell signal. 119 trades 2019-2026, win rate 71%, median
hold about 2 hours, profit spread across years (top 5 trades are 15% of the total). Trades dry up lately (6 in 2023, 8 in
2024, 2 in 2025, 1 in 2026). Written January 2022 and last changed December 2022 (repo history), so the years before 2023
include whatever data its parameters were tuned on.

| | trades | net CAGR | Sharpe | max DD | hold CAGR | hold Sharpe | z vs random | alpha t |
|---|---|---|---|---|---|---|---|---|
| BTC 2019-2026 | 119 | +13.2% | 1.42 | -6.8% | +47.4% | 0.95 | 6.8 (halves 5.0 / 4.8) | 3.4 |
| BTC 2023-2026 (out of sample) | 17 | +5.8% | 1.15 | -4.0% | +49.9% | 1.11 | 4.8 | 2.2 |
| ETH 2019-2026 | 217 | +7.2% | 0.48 | -30.5% | +43.7% | 0.86 | 3.8 | 0.9 |
| ETH 2023-2026 (out of sample) | 44 | +9.7% | 1.07 | -12.8% | +13.2% | 0.51 | 2.4 | 2.2 |

Out-of-sample rows are daily returns rebuilt from freqtrade's trade list (2023+); the z is the second-half random test.

Notes: nateemma's utils import TensorFlow at module level; the indicator strategies never call it, so `stubs/` stands in
(the ML strategies were not run). PatternPy writes strings into a float column, which pandas 3 refuses; that one line is
patched at import, logic untouched. The freeCodeCamp HQM notebook sorts without `inplace`, so as written it buys the
first 51 tickers alphabetically; the intended ranking is what was tested. Grid range (+-10%, 80 levels, monthly), DCA
ladder (5 safety orders at -2%, take profit +2%) and the 20-day hold of the stock screens are assumptions, because those
repos leave them to the user. Some nateemma runs on SPY/QQQ did not finish in the sandbox time limit (66 of 162 done); the 5-minute crypto runs of both packs are in the CI part `packs` (118 of 182 finished in its first 340-minute run; a re-run continues).

### Can ML cut the trades down to the ones that pay? (`meta.py`, 2026-11-18)

Joaquin's question after round 2. Meta-labeling (Lopez de Prado): for 34 fee-killed bots x markets (zenbot, gekko,
vnpy, crypto-trading-bot, backtesting.py on BTC/ETH 1h and 5m/15m, SPY/QQQ), every trade the bot would make is
described by the market at the entry decision (returns over 5 horizons, short/long volatility, RSI, distance from
moving averages, range position, volume, hour, weekday, the bot's own last results) and labelled by whether it made
money after fees. A gradient-boosted classifier trained only on trades closed before each test year picks that
year's trades (walk-forward 2021-2026; threshold chosen inside the training years). Rows are 2021+ only.

| | all trades (2021+) | ML-filtered |
|---|---|---|
| bot x market rows | 68 | 68 |
| trades kept | 100% | ~20% on average (5m bots: ~1%) |
| median net CAGR | -42%/yr | -1.2%/yr |
| rows above zero after fees | 10 of 68 | 21 of 68 |
| Sharpe above buy-and-hold | | 2 (ETH gekko PPO 1h, its 2023-26 half is no better than random) |
| CAGR above buy-and-hold | | 1 |

The filter works mostly by saying no: it stops the bleeding (zenbot sar BTC 5m goes from 30,000 trades and -100%/yr
to 114 trades and +0.9%/yr) but what is left is a bot that is almost always in cash. Two 5m rows now beat random timing
clearly (zenbot sar BTC z 4.1, vnpy DoubleMa ETH z 4.1, both halves positive), which says the filter finds real
moments, but they earn +0.9% and +5.1%/yr with 0.3-0.4% exposure: less than T-bills and far below holding (BTC +18%/yr,
ETH +21%/yr over 2021-26). After Holm across all rows none passes. Comparison rows with all trades:
`results/rows_meta_*.csv`.

ClucHAnix_5m passes, so a paper pot is offered, not started. It would mostly sit in cash (BTC under 1% of the time),
so as a whole pot it earns about T-bill money; it makes more sense as a sleeve that uses idle cash, and only if the
12-coin basket run from 2023 also holds up.

## Files

| file | what |
|---|---|
| `ftrun.py` | runs freqtrade's own CLI offline (static market table instead of an exchange call; Binance answers US runners with 451) |
| `ft_batch.py` | one freqtrade backtest per strategy and market, own timeframe where we have bars |
| `ft_score.py` | replays freqtrade's exported trades on its bars and scores them (CAGR matches freqtrade's total profit) |
| `basket_score.py` | scores a multi-coin run from freqtrade's wallet export |
| `ports.py`, `run_ports.py` | hummingbot, Superalgos, huseinzol05, quant-trading, backtesting.py ports and their runs |
| `ports2.py`, `run_ports2.py` | round 2: gekko, zenbot, vnpy, pycryptobot, crypto-trading-bot, smart-money-concepts, binance-trade-bot |
| `meta.py` | ML trade filter (meta-labeling, walk-forward) on the fee-killed bots |
| `ports3.py`, `run_ports3.py` | round 3: gekko pack, grid/DCA, pairs, EWMAC, supertrend, PatternPy, Alpaca scalper, conor19w futures bot, R-Breaker, Dual Thrust, DQN, volatility bot |
| `run_aitrader.py` | whchien/ai-trader's own backtrader classes on daily bars |
| `sp_pickers.py` | S&P 500 pickers (freeCodeCamp momentum, myhhub screens) vs random point-in-time picks |
| `stubs/` | TensorFlow/Keras stand-ins so nateemma's indicator strategies import |
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
