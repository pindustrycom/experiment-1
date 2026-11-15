# Take-Two (TTWO) around the GTA VI launch (2026-11-15)

Joaquin asked: use the project's tools to look for a volatility edge in Take-Two around GTA VI (launch **Thursday
Nov 19, 2026**), judge the outlook from the GTA V release history and this year's games industry, and give the risk
and return on **500 CAD**. Paper research only; nothing is traded. Data as of the **Friday Nov 13 close** (TTWO
$227.82, USD/CAD 1.4257, so 500 CAD = $351).

## Short answer

- **The options price a huge launch week; history says launches are quiet.** The Nov 20 straddle costs 8.2% of the
  stock (implied vol 74%, a one-week standard deviation of about 10%). Our vol models put a normal week at 40% vol,
  so the options charge about **1.9x a normal week**. In the six past Rockstar launches (GTA IV, RDR, GTA V x3, RDR2)
  the launch week moved **0.1% to 7.3%**, and launch weeks were no wilder than ordinary weeks. If the options were
  right, six launches this calm would happen about **2% of the time** (placebo on TTWO's own weeks; chi-square
  1%). Across all 12 big Take-Two launches the p-value is 0.12.
- **So the edge, if any, is selling that launch week, not buying it.** Today's straddle replayed on past launches lost
  money in 11 of 12 (Rockstar launches: average -77% of the premium). But it is one event, six data points, p 0.02
  against the project's multiple-testing bar of t 3.8 (p about 0.0001), and GTA VI is not GTA V: it is a much larger
  share of a company priced at 5.7x sales, and first sales numbers and reviews land inside the window. **Not an edge we
  can claim.** It also cannot be done with 500 CAD: one straddle costs 2,780 CAD to buy, and selling one needs several
  thousand dollars of margin with unlimited loss.
- **Delays were the real movers, and that risk is gone.** Each delay hit the stock hard (GTA IV -21% over 2 days,
  GTA VI -5% and -9%), reveals gave +5%. The date is locked, pre-load is open.
- **After launch the stock has tended to fade** ("sell the news"): 60 sessions after the six Rockstar launches the
  stock trailed the market by 8.8% on average, 4 of 6 negative (t -1.5, not significant).
- **500 CAD in TTWO vs SPY** (no edge assumed, both priced at the market's expected return; TTWO risk from the
  options): same expected value as SPY, much wider. One year: TTWO median 505 CAD, 5-95% range **241 to 1,030**,
  chance of loss 49%; SPY median 544, range **416 to 690**, chance of loss 29%. Launch week alone: TTWO 417 to 591.
- **Verdict: no edge.** TTWO is a fair-priced, high-risk single-stock bet on GTA VI sales; the only anomaly (rich
  launch-week options) is real-looking but untradeable at this size and too thin to trust. SPY stays the pick.

## Outlook (web, checked 2026-11-15)

- GTA VI: Nov 19, 2026 (PS5, Xbox Series), $79.99 standard edition (as reported); pre-orders opened June 25 (stock +4.6% on the
  date announcement); two delays before (May 2, 2025 and Nov 6, 2025).
- Q2 FY27 results (Nov 9 after the close): net bookings $1.60B, below guidance; FY27 net bookings guidance held at
  $8.0-8.2B (+21%), December quarter guided $3.35-3.50B. Management: pre-orders "far in excess of our initial
  expectations", but no raise. Stock +8.3% to $224.64 on Nov 10, $227.82 on Nov 13. Sensor Tower estimates 9.7M
  pre-sale units (~$942M); JPMorgan assumes about 34M GTA VI copies in FY27. Targets: Jefferies $300, HSBC $282.
- Valuation: 5.7x trailing sales and about 32x next-year earnings, against about 1.4x sales at GTA V's 2013 launch. A
  lot of success is priced in; the stock is -11% this year (log) while SPY is +11%.
- Industry 2026: memory (DRAM) prices pushed console prices up (Switch 2 +$50 to $499.99 from September, Nintendo cut
  its Switch 2 forecast 17%; Sony raised US PS5 prices twice in 12 months and says memory stays expensive into
  FY2027; Microsoft warns on pricing and supply). Fewer and pricier consoles is a headwind for a console-only launch,
  but GTA VI is the kind of title that sells hardware, and PS5/Xbox installed bases are already large.

Sources: TechRadar, Motley Fool (Nov 11), TradingKey (Nov 11), Sherwood News (May 8), IBTimes, Variety, SentiSense.

## Data (`fetch.py`, workflow `ttwo-gta.yml`, release `ttwo-gta-data`)

Tiingo daily OHLC for TTWO, EA, SPY, QQQ, XLC from 1997; Cboe delayed option chain for TTWO (1,110 contracts, 14
expiries, Nov 13 close); SEC EDGAR 8-K item 2.02 filings (86 earnings dates since 2005, after-close releases moved to
the next session); Bank of Canada USD/CAD; Google News headlines (Take-Two, GTA 6, TTWO) for Nov 5-14, scored with the
straddle8_news word list. The Haiku step failed: **the repo's ANTHROPIC_API_KEY returned 401 (invalid key)**, which
also affects any live job that scores news with Haiku. Each workflow run also saves the chain under a dated name, so
rerunning it after the launch gives the implied-vol crush.

## Method (`analysis.py`)

1. **Event study** (27 events, `results/events.csv`): market-model abnormal returns vs SPY (beta from days -260 to
   -20), windows -60..-1, -2..+2, 0..+20, 0..+60, 0..+250, and realized vol before and after.
2. **Vol models** on TTWO 2005-2026 with earnings days removed (the launch and earnings are added back as explicit
   jumps): HAR-RV on Garman-Klass + overnight gap variance, GARCH(1,1)-t, GJR-GARCH-t, log-RV ARMA(1,1) state-space
   (the timescale ladder's day layer without minute data). Out-of-sample 5-day QLIKE 2015-2026, monthly refits: HAR
   best; GJR level with it (DM t 0.4), GARCH worse (t 2.5), trailing 22-day vol much worse (+41%, t 4.1). The
   Student-t GRU and minute ladder from btc_ssm / timescale_ladder need intraday data we do not have for TTWO and
   added little on SPY at the daily level, so they are not used.
3. **Implied vs model** for every expiry (`results/vol_forecasts_vs_implied.csv`), model = average of HAR, GARCH,
   GJR + earnings jumps (TTWO earnings-day sd 8.9%) + the historical launch-window move (rms 3.8%).
4. **Launch test**: past Rockstar launch-week moves (sessions -3..+1, matching Fri close -> Fri expiry around a
   Thursday launch) in units of each period's normal vol, against today's implied/normal ratio of 1.86. Chi-square
   and an empirical placebo (random non-event TTWO weeks, standardised by trailing 60-day vol).
5. **500 CAD**: 40,000 paths, 10-day block bootstrap of joint daily (TTWO, SPY, USD/CAD) returns 2017-2026 (keeps fat
   tails and the stock/currency correlation), demeaned, drift = CAPM (3.7% + beta 0.80 x 5%) for TTWO and 8.7% for SPY.
   *Market view*: TTWO scaled so the simulated straddle is worth exactly its mid price at each expiry. *History view*:
   model vol + one launch jump drawn from the six past launches. Straddles bought at the ask, settled at expiry
   (the straddle8 pot's ask -> bid rule).

## Results

### Implied vs model vol (Nov 13 close)

| Expiry | Sessions | ATM implied | Model + earnings + launch | Gap (pts) |
|---|---|---|---|---|
| Nov 20 | 5 | 73.9% | 47.8% | +26.1 |
| Nov 27 | 9 | 72.2% | 44.5% | +27.7 |
| Dec 18 | 24 | 55.3% | 41.9% | +13.5 |
| Jan 15 | 42 | 48.5% | 41.1% | +7.3 |
| Mar 19 | 85 | 47.6% | 43.4% | +4.1 |
| Dec 2027 | 274 | 45.6% | 43.5% | +2.1 |

Realized: 22-day 46%, 1-year 35%. The long-dated gap is the usual small premium; the extra sits in launch week.

### Past launches (5-session raw move around launch, today's Nov 20 straddle = 8.2% of spot, 8.6% at the ask)

| Launch | Move | Straddle P/L (% of ask) |
|---|---|---|
| GTA IV (2008) | +0.1% | -99% |
| Red Dead Redemption (2010) | -0.3% | -97% |
| GTA V (2013) | -0.6% | -93% |
| GTA V PS4/XB1 (2014) | +2.0% | -77% |
| GTA V PC (2015) | -1.5% | -82% |
| Red Dead Redemption 2 (2018) | -7.3% | -15% |
| Other launches (Max Payne 3, Borderlands 2/3, BioShock Infinite, Mafia III, Civ VII) | 2.6% to 13.1% | 5 of 6 lose; Civ VII +53% (with earnings) |

P-values: moves this small if today's pricing were right: placebo 0.022, chi-square 0.012 (Rockstar, vol units),
0.12 (all 12 launches). Launch weeks vs ordinary weeks: p 0.30 (no calmer, no wilder).

### Around launches (abnormal vs SPY, six Rockstar launches)

| Window | Mean | t | Values |
|---|---|---|---|
| 60 sessions before | +9.0% | 1.2 | +37, +16, -4, +12, -18, +11 |
| 20 sessions before | -0.6% | -0.1 | |
| 20 after | -2.9% | -1.4 | |
| 60 after | -8.8% | -1.5 | -7, -24, -21, +1, +13, -15 |
| 250 after | -10.5% | -0.6 | -88 (2008), +28, -32, +13, +36, -21 |

Now: last 60 sessions +2% abnormal (the post-earnings jump offset the October slide), last 20 +13%.

Delays and reveals (days 0..+1 abnormal): GTA IV delay -21%, GTA V delay -3% (after -7% day 0), RDR2 delay with
earnings -5% (day 0 -12%), GTA VI delays -5% and -9%; GTA V reveal +7%, RDR2 reveal +5%, GTA VI trailer date +8%,
pre-order date +5%. Earnings days since 2005: average move 6.7%, sd 8.9%.

### 500 CAD, value in CAD (median, 5-95%, chance of loss)

| Horizon | TTWO (market risk) | TTWO (history risk) | SPY | Straddle (market) | Straddle (history) |
|---|---|---|---|---|---|
| Launch week (Nov 20) | 504, 417-591, 47% | 504, 457-560, 45% | 502, 484-515, 42% | 367, 30-1,329, 64% | 237, 20-768, 82% |
| Dec 18 | 504, 372-654, 48% | 506, 415-606, 45% | 505, 466-537, 40% | 398, 38-1,193, 60% | 271, 25-813, 78% |
| Mar 19 2027 | 504, 315-773, 49% | 508, 354-706, 47% | 515, 442-583, 36% | 399, 36-1,200, 60% | 308, 28-918, 73% |
| 1 year | 505, 241-1,030, 49% | 510, 275-924, 48% | 544, 416-690, 29% | 377, 36-1,333, 64% | 318, 31-1,067, 72% |

Average value: TTWO and SPY both about 543-553 CAD after a year (same expected return by construction; no edge is
assumed because none was found). Straddle average: about 479 CAD in the market view (you lose the 4-5% bid-ask
spread), 295-397 in the history view. A straddle seller in the history view keeps +35% of the premium on average in
launch week, but loses more than the whole premium 5% of the time; in the market view the seller loses the spread.
Fractional straddles are a paper idealisation: one real Nov 20 straddle is 2,780 CAD, a Dec 18 one 4,660 CAD.

## Checked against the reference books

- Sharma ch. 9 (costs): straddles priced ask -> bid, 4-5% of the premium on the near expiries.
- Sharma ch. 11 (sentiment): headline tone was positive all week (word score positive on 5 of the last 7 days, +0.14
  to +0.71), but the straddle8_news result is on 8% flag days and sentiment "helps only when slowed down"; we do not use
  it as a direction call here.
- Sharma ch. 12 and Coqueret & Guida ch. 14 (causality, non-stationarity): six launches from 2008-2018, when GTA was
  a smaller share of a cheaper company, are weak evidence about 2026.
- Coqueret & Guida ch. 12 (backtest overfitting): p 0.02 on one hypothesis does not survive the project-wide haircut.

## Possible follow-up

Rerun the workflow after the launch (Nov 20 close) to record the realized launch move and the implied-vol crush. That
adds one forward data point to the launch test at no cost. A paper pot is not justified by these results.
