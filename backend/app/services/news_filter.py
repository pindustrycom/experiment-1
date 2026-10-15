"""Free, rule-based screening of news stories before they are sent to Claude.

Finnhub tags a story with every ticker it mentions, so most of what arrives is market roundups,
"stocks to watch" lists and stories about other companies. Those are not evidence about either
company in a pair and are dropped here, without an AI call. SEC filings and press releases always
pass: they come from the company itself. A dropped story is marked "filtered" with the reason, so
it stays auditable and counts as handled in news coverage.
"""
from __future__ import annotations

import re

ROUNDUP = re.compile(
    r"\b(top \d+|\d+ (best |top )?(stocks|shares|dividend|companies)|stocks? to (buy|watch|sell)|best stocks|"
    r"market (wrap|recap|update|movers)|stock market today|movers|biggest (gainers|losers)|pre-?market|after-?hours movers|"
    r"morning (brief|call)|week ahead|what to watch|earnings (calendar|this week)|futures (rise|fall|climb|slip)|"
    r"(dow|nasdaq|s&p 500)( futures)? (rises?|falls?|climbs?|slips?|today|closes?|opens?)|etf|mutual fund|portfolio|"
    r"magnificent seven|13f|hedge funds? (buy|sell|bought|sold))\b", re.I)

# Common names that the legal names in reference.py do not cover.
EXTRA_ALIASES = {
    "KO": ["Coca-Cola", "Coke"], "PEP": ["Pepsi", "PepsiCo"], "WMT": ["Walmart", "Wal-Mart"], "TGT": ["Target"],
    "HD": ["Home Depot"], "LOW": ["Lowe's", "Lowes"], "V": ["Visa"], "MA": ["Mastercard"], "XOM": ["Exxon", "ExxonMobil"],
    "CVX": ["Chevron"], "AMD": ["AMD", "Advanced Micro"], "NVDA": ["Nvidia"], "JPM": ["JPMorgan", "JP Morgan", "Chase"],
    "BAC": ["Bank of America", "BofA"], "MS": ["Morgan Stanley"], "GS": ["Goldman"], "MCD": ["McDonald's", "McDonalds"],
    "YUM": ["Yum Brands", "Yum!", "KFC", "Taco Bell", "Pizza Hut"], "FDX": ["FedEx"], "UPS": ["UPS", "United Parcel"],
}
MAX_TICKERS = 6  # stories tagged with more companies than this are roundups


def _names(company) -> list[str]:
    out = [company.name, *(company.aliases or []), *EXTRA_ALIASES.get(company.ticker, [])]
    return [n.strip() for n in out if n and len(n.strip()) >= 3]


def mentions(text: str, company) -> bool:
    if any(re.search(r"(?<![\w])" + re.escape(n) + r"(?![\w])", text, re.I) for n in _names(company)):
        return True
    t = re.escape(company.ticker)
    # bare tickers like V, MA or MS are ordinary words; only count them as $V, (V) or NYSE: V
    return bool(re.search(rf"\${t}\b|\({t}\)|\b(NYSE|NASDAQ|Nasdaq)\s*:\s*{t}\b", text))


def screen(doc, tickers: list[str], companies: list) -> str | None:
    """Return why a story should not go to Claude, or None if it should. `companies` are the pair
    companies the story is tagged with."""
    if doc.doc_type in ("filing", "press_release"):
        return None
    text = f"{doc.title or ''}\n{(doc.content_text or '')[:2000]}"
    if len(set(tickers)) > MAX_TICKERS:
        return f"tagged with {len(set(tickers))} companies: a roundup, not company news"
    if ROUNDUP.search(doc.title or ""):
        return "market roundup or list headline"
    if not any(mentions(text, c) for c in companies):
        return "neither company is named in the headline or summary"
    return None
