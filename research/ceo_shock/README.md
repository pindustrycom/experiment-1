# Do CEO scandal drops bounce back? (2026-11-23)

Research only, paper only. Nothing is traded.

Joaquin's idea: when a big company drops on a shock headline about its CEO (an affair, a firing, legal trouble,
a sudden exit), the drop is overdone and partly comes back. His example is NVIDIA in November 2026.

## The NVIDIA case (checked against news and prices)

| Date | What happened | NVDA close | NVDA day | SPY day |
|---|---|---|---|---|
| Oct 29 | NYT reports a "close relationship" between Jensen Huang and a Megaspeed executive | 213.66 | -0.8% | -0.4% |
| Nov 8-9 | Board probes a foreign-intelligence angle; Huang confirms an affair | 178.67 | -4.3% | +2.4% |
| Nov 11 | NYT reports a second affair | 173.73 | +2.9% | +1.3% |
| Nov 12 | Huang resigns, CFO Colette Kress interim CEO | 166.19 | -4.3% | -0.4% |
| Nov 19 | | 145.98 | -3.3% | -2.0% |
| Nov 20 | "Plunging 14%" (AI spending fears plus Huang), week -23% (news, our data ends Nov 19) | ~126 | ~-14% | |
| Nov 23 | Board reinstates Huang unanimously after White House talks; stock "roughly 30%" up in the morning | ~163 intraday | ~+30% | |

Sources: Forbes timeline (Nov 23), Yahoo/TheStreet (Nov 23), Tom's Hardware (Nov 23). Prices to Nov 19 from the
data set below; Nov 20 and 23 are from news reports, not closes.

Two things stand out. The board's stated reasons were Huang's candor, U.S. government input that he poses no
national-security risk, and new compliance roles, so "they had to bring him back because of the loss" is only part of
the story. And the market was falling at the same time (SPY -7% from Oct 21 to Nov 19).

What our rule would have done: buy at the Nov 9 close (178.67) or the Nov 12 close (166.19). At ~163 on the
reinstatement morning, both are still a loss (about -9% and -2%). Only buying the Nov 20 low made money, and nobody
knew that was the low or that a reinstatement was coming.

NVIDIA is kept out of every test below (it is the case that gave us the idea).

## The test

- **Events:** 45 key-person shocks at S&P 500 companies, 2001-2025 (`events.csv`): 14 scandals (Hurd, Krzanich,
  Easterbrook, Wynn, Kotick, ...), 9 legal (Musk SEC suit, Skilling, Greenberg, McClendon, ...), 10 sudden exits
  (Witty, Niccol, Gelsinger, Buffett, Altman for MSFT, ...), 6 health/death (Jobs x4, Cantalupo, Thompson),
  3 Musk founder shocks, plus the 2 NVIDIA days. Every date was checked against the price reaction. 3 events have no
  price data (CBS/Moonves x2, Tyco/Kozlowski) and are dropped.
- **Universe and data:** point-in-time S&P 500 membership and adjusted prices (Johnbrick123/sp500-data), so dead
  companies like Enron and Fannie Mae are included.
- **Abnormal return:** stock minus beta x SPY, beta from 3 years ending a month before (Sharma: beta-neutral).
- **Shock filter:** day-0 abnormal drop of 3% or more. 16 events pass.
- **Entry:** day-0 close (the headline is known by then). Returns measured 1, 5, 10, 20 and 60 days after.
- **Baseline:** all 155,904 days since 1996 when an S&P 500 member fell 3%+ abnormally, for any reason.
- **Trade rule:** buy the stock at the day-0 close, hold 20 days, 10 bp round trip, compare with SPY.
- **Stats:** t-test and sign test, Holm correction across 10 tests.

## Results

Abnormal return after the entry close, 16 shock events:

| Days after | Mean | Median | Up share | p | Holm p | vs ordinary drops of same size |
|---|---|---|---|---|---|---|
| 1 | +0.5% | +0.2% | 50% | 0.72 | 1.0 | +0.4% |
| 5 | -0.0% | -0.6% | 44% | 0.97 | 1.0 | -0.7% |
| 10 | +0.4% | -0.7% | 50% | 0.81 | 1.0 | -0.6% |
| 20 | -2.2% | -1.9% | 38% | 0.54 | 1.0 | -4.0% (p 0.25) |
| 60 | +3.2% | +7.0% | 75% | 0.74 | 1.0 | -1.3% |

**No bounce.** On average the stock kept drifting with or below the market. It does a little worse than an ordinary
drop of the same size. The 60-day median is positive (+7%, 12 of 16 up, sign test p 0.08, fails after Holm), but
that is about what ordinary drops of the same size also do.

By type (20 days): scandal -4.8% (4), legal -2.8% (4), exits -0.7% (5), Musk founder shocks -0.2% (3).

The big winners and losers show why it does not work as a rule:
- Wins came when the shock was **reversed**: Tesla +37% in 20 days after Musk settled with the SEC two days later
  (the closest match to NVIDIA). Apple +20% after Jobs' 2009 leave.
- Losses came when the headline was the **first sign of a deeper problem**: Enron -32% in 20 days (bankrupt within 4
  months), AIG -15%, Chesapeake -16%, UnitedHealth after Witty -21% in 60 days.

You can't tell which kind it is on the day of the drop. That is the bet.

**Trade rule** (16 trades, buy at the drop close, hold 20 days, costs included): average -2.5% vs SPY per trade,
1 in 4 beat SPY, p 0.41. A pot sitting in SPY that switched into each shocked stock for 20 days grew 500 to 3,281
since 2001 (7.5%/yr) vs 4,322 for plain SPY (8.7%/yr).

**Ordinary big drops** (side result): outside crisis years (2000-02, 2008-09, 2020) a 3%+ drop has no bounce
(+0.0 to +0.8% over 20 days). The bounces seen in crisis years are the whole market recovering.

## How sure are we

16 events is small. Returns are very spread out (20-day standard deviation ~14%), so this test can only rule out a
big average bounce, about +10% in 20 days or more. A small bounce could exist and we would not see it. But the sign is
the wrong way, and the real money in these cases came from a reversal that nobody can forecast from prices.

Sharma (ch. 11) says news sentiment only adds value when it is slowed down and combined with other signals, not
traded on the headline day, which matches this result.

## Verdict

Not adopted. No paper pot. If we come back to it, the useful version is not "buy the scandal drop" but "buy when the
shock is reversed" (settlement, reinstatement, charges dropped), which is a different and much rarer event.

## Follow-up: buy once the shock is reversed (2026-11-23)

Joaquin asked to test the better version: wait for the shock to be undone, then buy. `reversals.csv` has 19
resolutions at S&P 500 companies 2010-2025, in three kinds:

- **Undone (9):** Musk-SEC settlements (Oct 2018, Apr 2019), Altman joining Microsoft and then reinstated at OpenAI
  (MSFT, Nov 2023), Iger back at Disney, Schultz back at Starbucks, Musk cutting DOGE time, the Musk-Trump feud cooling
  (twice). This is the NVIDIA reinstatement type. NVIDIA itself isn't in the data yet (prices end Nov 19).
- **Seat filled (6):** a credible new CEO after a shock exit (Lip-Bu Tan at Intel, Scharf at Wells Fargo, Apotheker at
  HP, Joly at Best Buy, Mayer at Yahoo, Swan at Intel).
- **Tainted person removed (4):** Wynn resigns, Thompson out at Yahoo, McClendon stripped of chairman, Muilenburg fired.

Same model as above (beta-adjusted, 10 bp round trip, 20-day hold vs SPY). Entry at the day-0 open, because most of these
are announced outside market hours, or at the day-0 close.

| Group | n | Gap at open | 20d after close | Buy at open, 20d vs SPY | Beat SPY |
|---|---|---|---|---|---|
| Undone | 9 | +5.2% | -3.3% | -2.2% | 2 of 9 |
| All resolutions | 19 | +3.7% | -3.0% | -3.1% | 6 of 19 |

Buying at the close instead: -1.8% (undone) and -2.6% (all) vs SPY. Every p-value is above 0.17, and every Holm p is 1.0.

**The reversal is priced in the overnight gap.** On average the stock opens about 5% higher, and after that it drifts
down. Only 2 of the 9 "undone" cases kept going: Tesla after the 2018 SEC settlement (+19% from the open) and Tesla
after Musk cut his DOGE time (+23%). Disney after Iger's return opened +9% and then lost 10% vs SPY in 20 days. Intel
after Lip-Bu Tan opened +14% and lost 10%. For comparison, ordinary 3%+ up days outside crisis years do nothing
afterwards (-0.0% over 20 days), so this group is, if anything, slightly worse than an ordinary good-news day.

So the only money in the NVIDIA-type story is in owning the stock **before** the reversal, which is the shock-day bet
from the first part. That bet loses on average, because you can't know a reversal is coming. NVIDIA is one of the rare
cases where it came. Not adopted.

Files: `reversals.csv`, `reversal.py` (`python reversal.py SP500_DIR results`), `results/reversals_car.csv`,
`results/reversals.json`.

## Files

- `events.csv`: the 45 hand-collected events
- `analysis.py`: the whole study (`python analysis.py SP500_DIR results`)
- `results/events_car.csv`: per-event returns; `results/results.json`: all stats;
  `results/baseline_summary.csv`: ordinary drops by size and crisis year; `results/pot_vs_spy.csv`: the pot curve

Data: `https://github.com/Johnbrick123/sp500-data/releases/download/data/{prices.parquet,membership_intervals.parquet}`
(prices to 2026-11-19).
