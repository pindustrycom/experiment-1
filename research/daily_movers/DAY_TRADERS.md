# Do day traders really make money like this? (2026-10-30)

Joaquin: "There are day traders that get money like this. Do a deep dive research on it."

**Short answer.** A few do, but very few, and you mostly hear from the lucky ones and from people selling
courses. Every study that has looked at *all* the day traders in a country, not just the ones who post wins,
finds the same thing: about 95-97% lose money, and well under 1% earn even a low wage from it, reliably.

## 1. The studies that tracked everyone

| Study | Who | What happened |
|---|---|---|
| Chague, De-Losso & Giovannetti, *Day Trading for a Living?* (2019) | All 19,646 people in Brazil who started day trading index futures, 2013-2015 | **97% lost money.** Of the 1,551 who kept at it 300+ days, 3% made any profit, 17 people (1.1%) made more than minimum wage (US$16/day), 8 (0.5%) more than a bank teller (US$54/day). The best one made US$310/day with swings of US$2,560/day. No sign they got better with practice. |
| Barber, Lee, Liu & Odean, *The Cross-Section of Speculator Skill* (J. Financial Markets 2014) | Every day trader in Taiwan, 1992-2006 (~450,000 a year) | About 20% of the active ones made money in a given year, but **less than 1% did so predictably and reliably** after fees. The top 500 kept earning ~0.38%/day after fees; past losers kept losing. |
| Barber, Lee, Liu, Odean & Zhang, *Do Day Traders Rationally Learn About Their Ability?* | Same Taiwan data | On average day traders lose 0.07%/day before costs and **0.24%/day after costs** (costs more than triple the loss). Profitable traders are consistently about 5% of the group. 75% quit within two years, but losers with 50+ days of experience keep trading almost as often as winners (95% vs 96%). |
| Bryzgalova, Pavlova & Sikorskaya, *Retail Trading in Options* (J. Finance 2023) | All US retail option trades, Nov 2019-Jun 2021 | Retail traders **lost more than $4 billion** after costs. Half their trades were options expiring within a week, with an average bid-ask spread of ~12.6%. |
| EU broker disclosures (CFD warning on every broker site) | 62 brokers, compiled by InvestinGoal | On average **~72% of retail accounts lose money**; the range is 58-89%. |
| Prop-firm "funded trader" challenges | FPFX (~300,000 accounts), The Funded Trader, FTMO | About 5-14% pass the challenge; only about **7% of all accounts ever get a payout** (1-2% at TFT), averaging ~4% of the account. The firms' revenue is mostly the challenge fees (FTMO: $204m revenue in 2023). |

Our own test (`README.md` in this folder) agrees: 13 daily rules on the S&P 500 since 2000, and every
stock-picking rule took $20 to about $0.

## 2. So why do you see people winning?

1. **Luck looks like skill over a year.** `lucky.py`: 100,000 simulated traders with *zero* skill each put
   $20 into a random S&P 500 stock every day for the last year (real prices, 5 bp costs). 32% ended with a
   profit, 884 (0.9%) doubled, 5 turned $20 into $100+. With options or leverage the spread is far wider, so
   the luckiest of millions of traders produce spectacular screenshots, and nobody posts the other 99%.
2. **Survivorship.** The 75% who quit within two years stop posting. The visible crowd is what is left after
   the losers leave.
3. **The money is often in selling the dream.** Courses, signal groups, Discord subscriptions, referral links
   and prop challenges earn whether or not the buyer wins. The prop-firm numbers above show the model: most
   revenue is fees from people who fail.
4. **Real winners exist but look different.** The reliable ~1% in Taiwan trade large size, often provide
   liquidity (they get paid the spread rather than paying it), and are closer to small professional firms. The
   successful professionals today are market makers and high-frequency firms with co-located servers and
   rebates, the other side of retail trades.

## 3. What it means for $20

- Expected outcome of day trading for a beginner: lose money, and faster with options or leverage. The
  studies say practice does not fix it.
- Winning needs an edge bigger than costs. On a $20 account you pay the spread every trade and cannot use the
  strategies that do have small edges (shorting gap-ups, market making).
- If the goal is to grow $20 aggressively, the honest best bet in our data is a leveraged index held with a
  trend rule (3x Nasdaq above its 200-day average), which beat holding the index since 2000 but fell 95% once.
  The safer real-world route is adding money regularly to an index fund; contributions grow a small account far
  faster than any trading edge.

## Sources

- Chague, De-Losso, Giovannetti (2019), Day Trading for a Living? SSRN 3423101; summary at
  https://quantpedia.com/retail-day-trading-is-an-uphill-battle/
- Barber, Lee, Liu, Odean (2014), The cross-section of speculator skill, J. Financial Markets 18:1-24,
  https://ideas.repec.org/a/eee/finmar/v18y2014icp1-24.html
- Barber, Lee, Liu, Odean, Zhang (2017), Do day traders rationally learn about their ability?
  https://faculty.haas.berkeley.edu/odean/papers/day%20traders/day%20trading%20and%20learning%20110217.pdf
- Bryzgalova, Pavlova, Sikorskaya (2023), Retail trading in options and the rise of the big three wholesalers,
  J. Finance 78(6), https://ideas.repec.org/p/cpr/ceprdp/17688.html ; $4bn figure via
  https://www.evidenceinvestor.com/post/how-profitable-is-retail-options-trading
- EU broker loss disclosures: https://investingoal.com/esma-broker-client-success-rate-stats
- Prop firms: https://www.financemagnates.com/forex/only-1-in-20-traders-pass-prop-firm-challenges-reports-the-funded-trader/
  and https://www.tradeinformer.com/prop-weekly/what-ftmo-tells-us-about-prop-firm-finances
