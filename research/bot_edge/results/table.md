Tested strategy x market rows (10+ trades): 1110

- raw p_random < 0.05 (beats random timing before any correction): 146 (13%; chance alone gives ~5%)
- survive Holm across all 1110: 25
- random-timing z >= 3.8 (project-wide bar): 36
- net Sharpe above buy-and-hold: 74
- net CAGR above zero: 450; above buy-and-hold CAGR: 36
- EDGE (all checks): 10

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
| freqtrade | 360 | -2.3% | +3.6% | 53 | 25 | 67% |
| freqtrade (shorts allowed) | 52 | -6.5% | +48.6% | 6 | 12 | 67% |
| freqtrade basket | 4 | -26.5% | n/a | 0 | 1 | 48% |
| freqtrade: nateemma/strategies | 230 | -10.7% | +6.3% | 13 | 4 | 72% |
| freqtrade: ssssi/freqtrade_strs | 29 | -0.9% | +6.3% | 8 | 1 | 65% |
| freqtrade: ssssi/freqtrade_strs basket (out of sample, 2023+) | 1 | +21.7% | n/a | 1 | 1 | 71% |
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
| freqtrade | NOTankAi_15_Cleaned_v2 | BTC | 15m | 1518 | +1575.1% | +2640.7% | 9.25 | 0.82 | +28.9 | 0.000 | 0.01 | +19.7 | +21.6 | +17.5 | look-ahead |
| freqtrade | BB_RTR | BTC | 5m | 21 | +6.0% | +6.7% | 0.87 | 0.82 | +22.0 | 0.000 | 0.00 | +17.3 | +0.0 | +2.2 | beats random (raw p) but fails a later check |
| freqtrade | BB_RPB_TSL | BTC | 5m | 18 | +3.3% | +3.9% | 0.83 | 0.82 | +21.4 | 0.000 | 0.00 | +16.4 | +6.9 | +2.1 | EDGE |
| freqtrade | GeneTrader_gen10_1734895087_6007 | BTC | 5m | 65 | +7.8% | +10.1% | 0.84 | 0.82 | +11.5 | 0.000 | 0.00 | +6.3 | +12.0 | +1.7 | beats random (raw p) but fails a later check |
| freqtrade: ssssi/freqtrade_strs | BBMod | BTC | 5m | 377 | +3.2% | +13.7% | 0.33 | 0.95 | +11.0 | 0.000 | 0.00 | +7.0 | +11.1 | +0.5 | beats random (raw p) but fails a later check |
| freqtrade | ClucHAnix_5m_old | BTC | 5m | 84 | +10.7% | +13.7% | 1.39 | 0.82 | +9.9 | 0.000 | 0.00 | +6.8 | +7.1 | +3.0 | EDGE |
| freqtrade | ClucHAnix_hhll | BTC | 5m | 66 | +4.7% | +6.9% | 0.74 | 0.82 | +9.3 | 0.000 | 0.00 | +9.8 | -1.3 | +2.0 | beats random (raw p) but fails a later check |
| freqtrade | E0V1E_DCA3 | BTC | 5m | 47 | +8.9% | n/a | 0.98 | 0.82 | +9.1 | 0.000 | 0.00 | +6.9 | +3.2 | +2.1 | EDGE |
| freqtrade: ssssi/freqtrade_strs basket (out of sample, 2023+) | ClucHAnix_5m | basket of 12 | 5m | 880 | +21.7% | n/a | 1.64 | 0.66 | +8.8 | 0.000 | 0.05 | +6.5 | +5.6 | +3.0 | beats random (raw p) but fails a later check |
| NostalgiaForInfinity basket (X7, in-sample) | NostalgiaForInfinityX7 | basket of 12 | 5m | 149 | +12.0% | n/a | 2.12 | 0.66 | +7.8 | 0.000 | 0.05 | +4.9 | +6.6 | +3.6 | beats random (raw p) but fails a later check |
| freqtrade | MiniLambo | BTC | 5m | 304 | -1.4% | +8.8% | -0.03 | 0.82 | +7.7 | 0.000 | 0.00 | +3.9 | +8.6 | -0.4 | beats random (raw p) but fails a later check |
| Binance-volatility-trading-bot | +10% jump buy, TP 0.8% trailing 0.4/0.1, SL 5% | basket of 12 | 5m | 46 | +7.8% | +8.1% | 0.50 | 0.54 | +7.3 | 0.000 | 0.21 | +11.4 | +1.0 | +1.0 | beats random (raw p) but fails a later check |
| freqtrade | ClucHAnix_BB_RPB_MOD | BTC | 5m | 228 | +13.6% | +22.3% | 1.08 | 0.82 | +7.3 | 0.000 | 0.00 | +3.8 | +7.8 | +2.1 | EDGE |
| freqtrade | newstrategy53_22 | BTC | 5m | 10 | +10.9% | +11.3% | 1.19 | 0.82 | +7.1 | 0.000 | 0.00 | +5.7 | +0.0 | +2.9 | beats random (raw p) but fails a later check |
| freqtrade: ssssi/freqtrade_strs | ClucHAnix_5m | BTC | 5m | 119 | +13.2% | +16.7% | 1.42 | 0.95 | +6.8 | 0.000 | 0.00 | +5.0 | +4.8 | +3.4 | EDGE |
| freqtrade | ClucHAnix_5m1 | BTC | 5m | 84 | +4.2% | +7.1% | 0.44 | 0.82 | +6.7 | 0.000 | 0.00 | +5.0 | +3.0 | +0.9 | beats random (raw p) but fails a later check |
| freqtrade | NOTankAi_15 | BTC | 15m | 921 | +611.7% | +859.4% | 4.88 | 0.82 | +6.6 | 0.000 | 0.01 | +4.1 | +6.8 | +8.9 | look-ahead |
| freqtrade | NOTankAi_15_Cleaned | BTC | 15m | 921 | +611.7% | +859.4% | 4.88 | 0.82 | +6.6 | 0.000 | 0.01 | +4.1 | +6.8 | +8.9 | look-ahead |
| freqtrade | GeneStrategy_v2 | BTC | 5m | 56 | +16.0% | +18.1% | 1.40 | 0.82 | +6.0 | 0.000 | 0.00 | +3.0 | +6.0 | +2.9 | EDGE |
| myhhub/stock (instock) | keep_increasing (rising MA30, +20%) | S&P 500 (point-in-time) | 1d | 48331 | +11.5% | +13.3% | 0.49 | 0.70 | +5.8 | 0.000 | 0.00 | +7.7 | -0.1 | +0.1 | beats random (raw p) but fails a later check |
| myhhub/stock (instock) | low_backtrace_increase (+60% in 60d, no deep drop) | S&P 500 (point-in-time) | 1d | 4108 | +12.2% | +13.4% | 0.52 | 0.70 | +5.4 | 0.000 | 0.00 | +2.6 | +4.5 | +1.6 | beats random (raw p) but fails a later check |
| freqtrade | CombinedBinHAndClucHyper | BTC | 5m | 296 | +5.6% | +16.2% | 0.92 | 0.82 | +5.0 | 0.000 | 0.01 | +10.2 | +1.1 | +2.4 | EDGE |
| NostalgiaForInfinity basket (out of sample) | NostalgiaForInfinityX4 | basket of 12 | 5m | 35 | +3.8% | n/a | 1.61 | 0.26 | +5.0 | 0.000 | 0.54 | +6.2 | +1.2 | +2.6 | beats random (raw p) but fails a later check |
| freqtrade (shorts allowed) | momentum_rsi | BTC | 5m | 897 | -38.4% | -28.7% | -0.60 | 0.82 | +4.9 | 0.000 | 0.00 | +3.0 | +4.3 | +0.8 | beats random (raw p) but fails a later check |
| freqtrade | CombinedBinHAndCluc2021 | BTC | 5m | 110 | +7.9% | +11.8% | 0.69 | 0.82 | +4.8 | 0.000 | 0.04 | +1.4 | +7.5 | +1.4 | beats random (raw p) but fails a later check |
