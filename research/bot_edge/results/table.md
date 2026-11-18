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
