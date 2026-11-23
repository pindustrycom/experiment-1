# Alternative data: satellite, Pentagon pizza, memory prices, Iran and oil (2026-11-23)

Paper research only. Joaquin asked why satellite data no longer works for us, whether the "Pentagon pizza index"
means anything, whether RAM prices help with chip stocks, and how the Iran war and oil feed into tech.
Data checks: `oil_memory.py`, run by `.github/workflows/alt-data.yml` on PR #59 (Brent from FRED, ^NDX ^SOX ^GSPC MU
SMH from Yahoo, 1985-10 to 2026-11-23).

## 1. Satellite parking-lot data: real edge, mostly gone for us

- **What it did.** RS Metrics counted cars at 67,078 stores of 44 US retailers (2011-2017). Parking-lot growth
  predicted same-store sales. Shorting the bad-news retailers and buying the good-news ones earned a **4-5% spread in the
  3 days around earnings** (short side -3.1%, long side +1.7%) [Katona, Painter, Patatoukas, Zeng, JFQA 2025].
- **Who won.** Hedge funds, the main buyers, at the expense of individual investors, who bought the stocks the funds
  were shorting [Berkeley Haas, 2019].
- **Why we can't do it now.**
  1. Cost: access ran "hundreds of thousands of dollars per year" before processing [JFQA paper, sec. II.A].
  2. Resolution: counting cars needs ~30-50 cm imagery (Maxar, Airbus, Planet SkySat). The free satellites
     (Sentinel-2, Landsat) see 10-30 m per pixel, so a whole car is a fraction of one pixel.
  3. Crowding: once dozens of funds buy the same feed, the surprise is priced before earnings. The paper's sample ends
     in 2017 and the authors expected the edge to be competed away. Funds have since moved to card-spending panels,
     app downloads and phone location data, which are also paid.
  4. Retail moved online, so a parking lot says less about sales than it did in 2012.
- **What is free.** Sentinel-5P NO2 (factory and traffic pollution) and VIIRS night lights are free but monthly and
  macro (China industrial activity, oil field flaring). They might help a macro view, not a 500 CAD daily stock pot.
  Not worth a test now.

## 2. The Pentagon pizza index: fun, no evidence

- **Origin.** A Washington Domino's franchise owner told reporters in 1990-91 that late-night orders to the Pentagon
  and CIA spiked before the Kuwait invasion and Desert Storm [Fox/LiveNOW Feb 2026; RFE/RL Jul 2025].
- **Today.** The "Pentagon Pizza Report" X account (2024+, 300k+ followers) and pizzint.watch read Google Maps "busy
  hours" of pizzerias near the Pentagon. It went viral when places were busy two nights before Israel struck Iran in
  June 2025 [RFE/RL]. Polymarket links it [LessWrong].
- **Evidence.** The one honest test (LessWrong, Jan 2026) scraped 648 posts and checked pre-registered operations:
  Midnight Hammer (Jun 22 2025) and the Maduro raid (Jan 3 2026) both happened during "utterly unremarkable" pizza
  activity, and the Maduro raid at a low. The Pentagon said the posts "did not align with the events". Spikes are
  remembered, misses are not.
- **Could we trade it?** No. There is no archived history of Google busy-hours to backtest, and even a correct warning
  would not tell us which way to trade: in our table below, the moment of a US strike was often the low for tech
  (Midnight Hammer: Nasdaq 100 +4.2% and Brent -11.9% over the next 5 days).

## 3. RAM / memory prices and Micron

- **Now.** TrendForce expects conventional DRAM contract prices +10-15% in Q4 2026 and NAND +15-20%, with server
  DRAM short because of AI demand, while spot prices have stalled (DDR4 16Gb spot ~$82.7, flat) [Capital Times, 3 Nov
  2026, citing TrendForce]. Micron is +147% year to date but fell 18% from Oct 16 to Nov 16 and sits ~40% below its
  June 25 high of $1,213.56 [Trefis, Nov 17 2026]; that drop came with all chips falling, not memory news.
- **Our check.** The only free long history is the US producer price index for semiconductors (FRED
  PCU334413334413, monthly, 2000-2026). Correlation of 3-month changes with Micron is ~0.15 at the same time and when
  the stock moves first, and fades to ~0 when the price index is the one moving first (3+ months). Holding Micron only
  when the index was rising earned **+9.3%/yr vs +12.5%/yr for just holding** (invested 25% of the time).
  **The stock moves before the public price data, so public memory prices are already in the price.**
- **What could work.** Daily DRAM *spot* prices (DRAMeXchange/TrendForce) are known to turn before contract prices,
  and the full history is paid. TrendForce shows today's spot table free, so we could start collecting it daily (like
  the Deribit and options collectors) and test in ~6-12 months whether spot turns lead Micron vs SMH. Cheap to start.

## 4. Iran war, oil and tech

The 2026 war: Hormuz closure from late Feb 2026 cut ~20% of world oil supply, the IEA's "largest supply disruption in
history"; ceasefire Apr 8; new US strikes Aug 31 [Wikipedia, 2026 Iran war fuel crisis; CNBC]. FRED Brent spot
peaked at $138.2 on Apr 7 and is $92.9 now (+52% this year). Nasdaq 100 is +5.6% this year, the chip index +34%.

| Episode | Brent 20d | Nasdaq 100 20d | Chips (SOX) 20d | Nasdaq 100 60d |
|---|---|---|---|---|
| Iraq invades Kuwait (Aug 1990) | +38.7% | -13.5% | n/a | -20.5% |
| Desert Storm starts (Jan 1991) | -30.6% | +23.3% | n/a | +37.6% |
| Iraq war starts (Mar 2003) | -12.9% | -1.9% | -2.4% | +12.0% |
| Russia invades Ukraine (Feb 2022) | +28.4% | +6.9% | +2.5% | -12.1% |
| Israel strikes Iran (Jun 12 2025) | +0.2% | +4.3% | +7.6% | +8.8% |
| US strikes Iran (Jun 20 2025) | -8.6% | +7.2% | +10.1% | +12.2% |
| Iran war, Hormuz shut (Mar 2 2026) | +57.8% | -8.2% | -12.2% | +19.9% |
| Iran-US ceasefire (Apr 8 2026) | -15.1% | +14.8% | +34.8% | +19.3% |
| New US strikes (Aug 31 2026) | +27.0% | +3.0% | +9.5% | -9.5% |

What the data says:
- **A real supply shock hurts tech for a few weeks, then the bounce is large** once the shooting starts or a ceasefire
  lands (1991, 2003, 2025, Apr 2026). "Sell the build-up, buy the war" shows up again and again, but it is ~10 events.
- **The oil-tech link flipped in 2026.** Daily correlation of Brent and Nasdaq 100 was +0.1 to +0.3 from 2008-2025
  (both rose on good growth news). In 2026 it is **-0.27**: oil now trades as war risk, so oil up means tech down.
- **Oil spike days alone don't predict tech.** 110 days with Brent +5% (de-clumped) since 1988: Nasdaq 100 next
  20 days +0.95% vs +1.07% normal (t -0.1). After a 15%+ 5-day oil jump (27 events) the next day is +0.7% (t 2.2), but
  20 and 60 days are noise; with this many tests that is not an edge (Sharma ch. 8 on multiple testing).
- **Why chips are hit hardest.** Higher oil means higher inflation and rates, which hurt long-duration growth stocks;
  the Gulf also supplies helium and LNG for chip fabs and energy for data centres; and chips carry the most beta.

## What is cheaply testable next

1. Start a daily DRAM spot price collector (free TrendForce table) and test spot turns vs Micron/SMH in 2027.
2. A "war de-escalation" rule (buy QQQ/SMH when Brent falls 10% from a war peak) has ~10 events since 1990; it
   can be written down and tracked forward, but it can't be proven with this little history.
3. Satellite and pizza: no cheap test worth running.

Sources: Katona et al., JFQA 60(2) 2025, "On the capital market consequences of big data: evidence from outer space";
Berkeley Haas newsroom, 28 May 2019; LessWrong "Does Pentagon pizza theory work?" (Jan 2026); RFE/RL, 9 Jul 2025;
Fox/LiveNOW, 23 Feb 2026; Capital Times / TrendForce, 3 Nov 2026; Trefis, 17 Nov 2026; Wikipedia "2026 Iran war
fuel crisis"; FRED DCOILBRENTEU, PCU334413334413.
