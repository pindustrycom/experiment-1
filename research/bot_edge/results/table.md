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

| source | strategy | asset | tf | trades | net CAGR | gross | Sharpe | hold Sharpe | z vs random | p | Holm p | z 2019-22 | z 2023-26 | alpha t | verdict |
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
