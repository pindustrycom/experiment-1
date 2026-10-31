"""The four rules. Each one turns what is known at a close into target weights (fraction of the pot per ticker,
negative = short) to hold until the next close. Pure functions of their inputs, so the live run and the replay share them.

Long-term (positions held for weeks or months):
  trend    Cross-asset trend following: eight ETFs (US, developed and emerging stocks, long and medium Treasuries, gold,
           commodities, real estate). At each month's last close, long the ETFs that are up over the past 12 months and
           short those that are down, each sized to about 10% yearly volatility, split over the eight; gross capped at 2.
  pead     Post-earnings drift: buy an S&P 500 stock at the first close after it beats the analysts' EPS estimate by 5% or
           more, hold it 60 sessions. 5% of the pot per stock (equal shares once more than 20 are held), the rest in cash.
  insider  Insider cluster buying: buy an S&P 500 stock at the first close after three or more different insiders have
           filed open-market purchases within 30 days, hold it 120 sessions. Sized like pead.
  best_ideas, conviction, pelosi  Copying public filings (13Fs, Pelosi's PTRs): see copy.py.
Short-term (positions held for days):
  btc_donchian  Bitcoin 4-hour breakout: see crypto.py.
  btc_donchian_vol  The same breakout, sized by BTC's 30-day volatility: see crypto.py.
  reversal Weekly reversal: at each week's last close, buy the 20 S&P 500 stocks that fell the most over the past 5
           sessions, 5% each, and hold them for the week.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

TREND_ETFS = ["SPY", "EFA", "EEM", "TLT", "IEF", "GLD", "DBC", "VNQ"]
TREND_LOOKBACK, TREND_VOL_DAYS, TREND_VOL_TARGET, TREND_GROSS = 252, 63, 0.10, 2.0
PEAD_SURPRISE, PEAD_HOLD = 0.05, 60
INSIDER_MIN, INSIDER_WINDOW, INSIDER_HOLD = 3, 30, 120
EVENT_SLOTS = 20                     # pead and insider: 1/20 of the pot per stock until more than 20 are held
REVERSAL_N, REVERSAL_DAYS = 20, 5

LONG_TERM, SHORT_TERM = "Long-term", "Short-term"
# key: (name, category, one-line description)
STRATEGIES = {
    "trend": ("Trend following", LONG_TERM, "Eight ETFs across stocks, bonds, gold and commodities: long what rose over the past "
              "year, short what fell, resized each month to equal risk."),
    "pead": ("Earnings drift", LONG_TERM, "Buys S&P 500 stocks after a clear earnings beat (EPS 5% or more above the estimate) "
             "and holds them for about three months."),
    "insider": ("Insider buying", LONG_TERM, "Buys S&P 500 stocks after three or more of their own insiders bought shares on "
                "the open market within 30 days; holds them for about six months."),
    "best_ideas": ("Best ideas", LONG_TERM, "Holds each of 23 famous funds' biggest stock from its latest 13F filing (Berkshire, "
                   "Scion, Pershing Square...), bought after the filing is public; one equal slice per fund."),
    "conviction": ("Fund conviction", LONG_TERM, "Holds the stocks that long-term funds put 7.5% or more of their book in, "
                   "most-shared first, up to 50; rebuilt 47 days after each quarter from the 13Fs filed by then."),
    "pelosi": ("Pelosi copy", LONG_TERM, "Buys every stock Nancy Pelosi discloses buying (shares or calls, copied with shares) "
               "after the filing is public; sells when she discloses a sale or after a year."),
    "reversal": ("Weekly reversal", SHORT_TERM, "Each Friday buys the 20 S&P 500 stocks that fell the most that week and holds "
                 "them until the next Friday. No stop loss: the twin of the pot below."),
    "reversal_stop": ("Weekly reversal, 5% stop", SHORT_TERM, "The same 20 stocks, but a stock that falls 5% in a day is sold "
                      "and bought back at the close, which caps the pot's daily loss near 5%."),
    "btc_donchian": ("BTC breakout", SHORT_TERM, "Bitcoin on 4-hour bars (Jesse's Donchian example): buys a close above the past "
                     "20 bars' high while above the 200-bar average, sells a close below the past 20 bars' low; trades last "
                     "about 10 days. See crypto.py."),
    "btc_donchian_vol": ("BTC breakout, vol-sized", SHORT_TERM, "The same Bitcoin breakout signal, but the position is sized "
                         "to a steady risk: 2.5% of daily volatility over BTC's past 30 days, up to 2x the pot when BTC is "
                         "calm and less in wild markets. See crypto.py."),
    "straddle8": ("8% day straddles", SHORT_TERM, "Each close, buys at-the-money option straddles (a call and a put) on the 5 "
                  "S&P 500 stocks the time-scale ladder rates most likely to move 8% the next day, either way, and sells them "
                  "at the next close. 20% of the pot goes in, bought at the ask and sold at the bid. See straddle.py."),
    "straddle8_news": ("8% day straddles, news-tilted", SHORT_TERM, "The same 5 flagged stocks and terms as the 8% day "
                       "straddles, but where the day's headlines (read by Claude Haiku before the close) lean one way it holds "
                       "2 options on that side and 1 on the other (2 calls + 1 put on good news, 1 call + 2 puts on bad). "
                       "Tests whether news, which told 8% rises from falls about 60% of the time since 2019, pays. See straddle.py."),
}


def month_end(dates: pd.DatetimeIndex) -> pd.Series:
    """True on the last session of each month, as far as the dates show (the latest date counts only once its month is over,
    so it is decided by the caller)."""
    p = dates.to_period("M")
    return pd.Series(np.r_[p[1:] != p[:-1], False], index=dates)


def week_end(dates: pd.DatetimeIndex) -> pd.Series:
    """True on the last session of each week: Friday, or the last session before a weekend or holiday break."""
    w = dates.to_period("W")
    return pd.Series(np.r_[w[1:] != w[:-1], False], index=dates)


def is_last_session_of_week(d: pd.Timestamp) -> bool:
    """For the latest close, where the next session is not in the data yet: a Friday (holiday Fridays are rare and only
    delay the rebalance to the next week's last session)."""
    return d.weekday() == 4


def is_last_session_of_month(d: pd.Timestamp) -> bool:
    """For the latest close: no weekday left in the month after d (a month ending on a holiday rebalances a session late)."""
    nxt = d + pd.offsets.BDay(1)
    return nxt.month != d.month


def trend_weights(px: pd.DataFrame, d: pd.Timestamp) -> dict[str, float]:
    """px: adjusted closes, one column per ETF. Weights from data up to and including the close of d."""
    h = px.loc[:d].dropna(axis=1, how="all")
    cols = [c for c in TREND_ETFS if c in h.columns and h[c].notna().sum() > TREND_LOOKBACK]
    if not cols:
        return {}
    out = {}
    for c in cols:
        s = h[c].dropna()
        mom = s.iloc[-1] / s.iloc[-1 - TREND_LOOKBACK] - 1
        vol = s.pct_change().iloc[-TREND_VOL_DAYS:].std() * np.sqrt(252)
        if not np.isfinite(vol) or vol <= 0:
            continue
        out[c] = float(np.sign(mom) * TREND_VOL_TARGET / vol / len(TREND_ETFS))
    gross = sum(abs(w) for w in out.values())
    if gross > TREND_GROSS:
        out = {k: w * TREND_GROSS / gross for k, w in out.items()}
    return {k: w for k, w in out.items() if w != 0}


def reversal_weights(px: pd.DataFrame, d: pd.Timestamp, members: list[str]) -> dict[str, float]:
    """The REVERSAL_N members with the worst return over the last REVERSAL_DAYS sessions up to d."""
    h = px.loc[:d, [m for m in members if m in px.columns]]
    if len(h) <= REVERSAL_DAYS:
        return {}
    r = (h.iloc[-1] / h.iloc[-1 - REVERSAL_DAYS] - 1).dropna()
    worst = r.nsmallest(REVERSAL_N)
    return {t: 1.0 / REVERSAL_N for t in worst.index}


def event_weights(holdings: dict[str, int]) -> dict[str, float]:
    """pead and insider: equal money per held stock, at most 1/EVENT_SLOTS each."""
    n = len(holdings)
    return {t: 1.0 / max(n, EVENT_SLOTS) for t in holdings} if n else {}


def earnings_beats(cal: pd.DataFrame) -> pd.DataFrame:
    """Rows of an earnings calendar (symbol, date, epsActual, epsEstimate) that beat the estimate by PEAD_SURPRISE or more."""
    if cal.empty:
        return cal
    c = cal.dropna(subset=["epsActual", "epsEstimate"])
    c = c[c["epsEstimate"].abs() >= 0.01]
    surprise = (c["epsActual"] - c["epsEstimate"]) / c["epsEstimate"].abs()
    return c[surprise >= PEAD_SURPRISE].assign(surprise=surprise)


def insider_clusters(tx: pd.DataFrame, d: pd.Timestamp) -> list[str]:
    """Symbols with INSIDER_MIN or more distinct insiders filing open-market purchases (code P) in the INSIDER_WINDOW days to d.
    tx: symbol, name, transactionCode, change, filingDate."""
    if tx.empty:
        return []
    f = pd.to_datetime(tx["filingDate"], errors="coerce")
    t = tx[(tx["transactionCode"] == "P") & (tx["change"] > 0) & (f <= d) & (f > d - pd.Timedelta(days=INSIDER_WINDOW))]
    n = t.groupby("symbol")["name"].nunique()
    return sorted(n[n >= INSIDER_MIN].index)


def book_return(w: dict[str, float], px: pd.DataFrame, d0: pd.Timestamp, d1: pd.Timestamp) -> float:
    """Return from close d0 to close d1 of holding weights w (a ticker without both prices counts as flat)."""
    if not w:
        return 0.0
    a, b = px.loc[d0], px.loc[d1]
    tot = 0.0
    for t, x in w.items():
        if t in px.columns and pd.notna(a.get(t)) and pd.notna(b.get(t)) and a[t]:
            tot += x * (b[t] / a[t] - 1)
    return float(tot)


def turnover(old: dict[str, float], new: dict[str, float]) -> float:
    return float(sum(abs(new.get(k, 0.0) - old.get(k, 0.0)) for k in set(old) | set(new)))
