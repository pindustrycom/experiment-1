# Kronos candle forecaster, out of sample on Bitcoin and Ether

Source: an Instagram carousel (getintoai, 2026-11-06). Slide 3 says a developer used Claude Code to build a crypto
dashboard that "predicts upcoming 2-hour candlesticks using Kronos, an open-source model trained on 12 billion
candles". The screenshot shows `git clone github.com/shiyu-coder/Kronos`, Kronos-small from Hugging Face, Binance
BTCUSDT 1h, paper mode. Slide 2 (a 19-year-old's Claude Code crypto bot) gives no source, rules or code, and its picture
is an unrelated AI-lab dashboard, so there was nothing to test there.

**Is it real?** Yes. Kronos is the official repo of "Kronos: A Foundation Model for the Language of Financial Markets"
(Shi et al., arXiv 2508.02739, accepted at AAAI 2026, MIT licence). Kronos-small is 24.7M parameters, context 512
candles, pre-trained on 12B+ candles from 45 exchanges. We pinned repo commit `67b630e` and HF weights
`NeoQuasar/Kronos-small@901c26c` (created 2025-06-30, last modified 2025-09-09; see `results/model_info.json`).

## Verdict

**No direction edge, worse volatility forecasts than a 1990s model, and every trading version loses to costs.**
Kronos draws nice-looking candles, but on 14 months the model has never seen it calls the next 1, 2, 4, 8 or 24
hours of BTC and ETH no better than a coin flip (best hit rate 51.6%, all 10 tests p > 0.12, Holm-adjusted p = 1.0).
Its sampled paths are far too narrow: the real 24h move lands outside all 8 sampled paths 53% of the time against 22%
if calibrated. As a volatility forecast it is clearly worse than GARCH and HAR-RV (t 5.6 on BTC). Trading its signal
every 4 hours at 5 bp a side turned BTC's -36% into -49%; the live pot's Donchian rule made +4.8% over the same window.

## Test design (out of sample, no tuning)

- Data: Binance spot hourly klines (data.binance.vision), BTCUSDT and ETHUSDT, to 2026-11-13.
- Test window: **2025-09-01 to 2026-11-13**, after the model's release, so its pre-training cannot contain it.
  Starting at 2025-10-01 instead (after the last weight update date) gives the same numbers (BTC 2h IC 0.030, 24h -0.042).
- Origins: every 4h on BTC (2,629) and every 8h on ETH (1,315). At each origin Kronos sees the previous 512 finished
  hourly candles (OHLCV + quote volume, normalised per window exactly as `KronosPredictor` does) and samples
  **8 paths of the next 24 hours** with the predictor's defaults (T = 1.0, top_p = 0.9). We keep every path rather than
  the averaged one the stock predictor returns. CPU, 20 GitHub Actions shards, about 3 hours (`.github/workflows/kronos.yml`).
- Signal: the mean of the 8 paths at each horizon (what the dashboard shows), plus the share of paths that end up.
- Scoring (`analyze.py`): hit rate vs always-up and vs momentum, AUC, rank IC with Newey-West t, Holm over all 10
  direction tests; next-24h realized variance vs EWMA, GARCH(1,1) and HAR-RV fit only on 2020-01 to 2025-08 (QLIKE,
  as in research/btc_ssm); rank of the realized 24h return among the 8 paths; long/flat and long/short trading
  holding for the next 4h (BTC) or 8h (ETH) at 0 / 5 / 10 / 25 bp a side.

## Results

### Direction (rank IC of the mean path vs the realized return)

| Coin | Horizon | Hit Kronos | Hit always-up | AUC | Rank IC | t | Holm p |
|---|---|---|---|---|---|---|---|
| BTC | 1h | 50.9% | 49.2% | 0.514 | +0.023 | 1.08 | 1.0 |
| BTC | **2h** (the slide's claim) | 51.1% | 49.5% | 0.522 | +0.032 | 1.54 | 1.0 |
| BTC | 4h | 49.8% | 50.0% | 0.504 | +0.010 | 0.53 | 1.0 |
| BTC | 8h | 50.0% | 51.0% | 0.503 | +0.010 | 0.41 | 1.0 |
| BTC | 24h | 47.4% | 50.6% | 0.467 | -0.042 | -1.09 | 1.0 |
| ETH | 1h | 51.6% | 50.7% | 0.512 | -0.010 | -0.36 | 1.0 |
| ETH | 2h | 48.7% | 51.4% | 0.481 | -0.034 | -1.18 | 1.0 |
| ETH | 4h | 48.1% | 51.4% | 0.482 | -0.039 | -1.36 | 1.0 |
| ETH | 8h | 50.3% | 50.8% | 0.507 | +0.019 | 0.66 | 1.0 |
| ETH | 24h | 50.8% | 49.9% | 0.518 | +0.019 | 0.49 | 1.0 |

What the model does: its mean path leans against the last move (rank correlation with the previous same-length
return -0.25 to -0.34) and it calls "down" 54-58% of the time. Short-term reversal on its own has no edge here either
(momentum IC between -0.043 and +0.015), so Kronos is mostly a noisy reversal signal.

### Volatility (next 24h realized variance; lower QLIKE is better, vs GARCH)

| Coin | Kronos | Kronos, level fixed* | EWMA 1d | GARCH(1,1) | HAR-RV |
|---|---|---|---|---|---|
| BTC QLIKE | 0.913 (+0.42, t 5.7) | 0.816 (t 5.0) | 0.462 | 0.488 | **0.383** (t -5.4) |
| BTC log R² | 0.12 | 0.12 | 0.17 | 0.19 | **0.23** |
| ETH QLIKE | 1.230 (+0.79, t 3.7) | 0.981 | 0.522 | 0.445 | **0.421** |
| ETH log R² | 0.13 | 0.13 | 0.19 | 0.18 | **0.23** |

\* rescaled with hindsight so its average level is right; it still loses, so the problem is ranking, not just level.
HAR-RV is again the best day-ahead vol model, as in research/btc_ssm.

Calibration of the 24h paths: realized return lands below all 8 paths 26% (BTC) / 24% (ETH) of the time and above all
8 another 27% / 26%, against 11% each if calibrated (chi-square p ~ 0). Median path spread 1.05% vs median real move 1.14%.

### Trading (hold the next 4h on BTC, 8h on ETH; 2025-09-01 to 2026-11-13)

| Strategy | 0 bp | 5 bp a side | 10 bp | 25 bp (the live pots' cost) |
|---|---|---|---|---|
| BTC hold | -36.3% | -36.3% | -36.4% | -36.4% |
| BTC Kronos long/flat (745 trades/yr) | -20.6% | -49.2% | -67.6% | -91.6% |
| BTC Kronos long/short | -11.6% | -63.9% | -85.3% | -99.0% |
| BTC Donchian 4h (live pot rule) | +6.8% | +4.8% | +2.9% | -2.8% |
| ETH hold | -56.0% | -56.0% | -56.0% | -56.1% |
| ETH Kronos long/flat | -6.3% | -25.7% | -41.1% | -70.7% |
| ETH Kronos long/short | +58.1% | -0.6% | -37.6% | -84.6% |

Against holding the same average exposure, BTC long/flat is t -0.1 before costs and t -2.2 at 5 bp. The ETH long/short
0 bp gain is being short 58% of the time in a market that fell 56%, not skill (its IC is t < 1.4 at every horizon); one
round trip of costs at 5 bp removes all of it.

## Compared with what we already have

- Direction: same as every BTC model so far (state-space, GRU, stochastic vol in research/btc_ssm): no edge.
- Risk: worse than GARCH, HAR and the Student-t GRU; HAR stays the best 24h vol forecast.
- Trading: the BTC Donchian pot beat holding by about 41 points over this window; Kronos loses to costs.
- Edge-test standard (research/edge_tests, PR #50 haircut): nothing here comes close to passing (Holm p = 1.0 on all
  10 direction tests), so Exp2's synthetic vol trade is still the only survivor.

## Reference book check

- Sharma, *The Quantamental Revolution*, ch. 9 (TCA): a signal must clear real costs; at 745 to 1,489 trades a year
  even 5 bp a side wipes out more than any plausible edge here. Ch. 12 (causal critique): check what a black box
  actually responds to; here it is mostly the last move reversed, which on its own has no edge. Ch. 13 (AI in research):
  impressive tooling is not evidence of alpha.
- Coqueret & Guida, ch. 10 and 12 (validation, backtest overfitting): strictly after-release test window, no
  parameter tuning on the test, all horizons and coins reported and Holm-adjusted.

## Limits

Kronos-small only (base is 4x slower on CPU; large is not public), 8 samples per origin, no fine-tuning (the repo
offers it; fine-tuning on the same window would be in-sample), hourly bars only. The dashboard in the post also used
1h bars, so this is the setup it showed.

## Files

- `fetch.py`: hourly Binance klines. `forecast.py`: Kronos sampling, sharded. `analyze.py`: all scoring.
- `results/`: `summary.json`, `direction.csv`, `vol.csv`, `equity_*.csv` (hold vs Kronos at 5 bp), `model_info.json`.
- Raw paths and the exact input candles: release `kronos-results`.
