"""Documented source-credibility tiers (deterministic; not AI-estimated).

1.00  SEC EDGAR filings (legally accountable primary disclosures)
0.95  Company press releases / investor-relations feeds
0.85  Major financial newswires and outlets
0.60  Other identifiable publishers
0.50  Unknown publisher
"""
from __future__ import annotations

MAJOR = {
    "reuters", "associated press", "ap", "bloomberg", "wall street journal", "wsj", "financial times", "ft",
    "cnbc", "dow jones", "marketwatch", "barron's", "barrons", "the new york times", "nikkei", "fortune",
    "business wire", "pr newswire", "globenewswire", "seeking alpha", "yahoo finance", "investing.com",
}


def reliability_for(doc_type: str, publisher: str | None) -> float:
    if doc_type == "filing":
        return 1.0
    if doc_type == "press_release":
        return 0.95
    if not publisher:
        return 0.5
    p = publisher.strip().lower()
    for m in MAJOR:
        if p == m or p.startswith(m + " ") or m in p.split(" - "):
            return 0.85
    return 0.6
