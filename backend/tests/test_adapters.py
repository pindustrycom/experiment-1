"""Provider adapters parse the documented response formats (sample payloads below are SIMULATED,
shaped like each provider's public API). No network calls are made."""
from __future__ import annotations

from datetime import UTC, date, datetime

import pytest

from app.data_sources.base import ProviderError
from app.data_sources.news import parse_alphavantage, parse_feed, parse_finnhub
from app.data_sources.prices import parse_stooq, parse_tiingo
from app.data_sources.sec import html_to_text, parse_company_facts, parse_recent_filings

START, END = datetime(2026, 9, 1, tzinfo=UTC), datetime(2026, 9, 30, tzinfo=UTC)


def test_tiingo_adjusted_prices():
    bars = parse_tiingo([
        {"date": "2026-09-14T00:00:00.000Z", "close": 61.2, "adjClose": 60.9, "volume": 1200, "divCash": 0.0, "splitFactor": 1.0},
        {"date": "2026-09-15T00:00:00.000Z", "close": 61.0, "adjClose": 61.0, "volume": 900, "divCash": 0.51, "splitFactor": 1.0},
    ])
    assert [b.date for b in bars] == [date(2026, 9, 14), date(2026, 9, 15)]
    assert bars[0].adj_close == 60.9 and bars[1].dividend == 0.51
    with pytest.raises(ProviderError):
        parse_tiingo({"detail": "Invalid token"})
    with pytest.raises(ProviderError):
        parse_tiingo([{"date": "2026-09-14", "close": "x"}])


def test_stooq_csv():
    bars = parse_stooq("Date,Open,High,Low,Close,Volume\n2026-09-14,60,62,59,61.5,1000\n2026-09-15,61,61,60,N/D,0\n")
    assert len(bars) == 1 and bars[0].close == 61.5
    with pytest.raises(ProviderError):
        parse_stooq("No data")


def test_finnhub_company_news_filters_window_and_requires_url():
    docs = parse_finnhub([
        {"category": "company", "datetime": 1789560000, "headline": "PepsiCo recalls snacks", "id": 1, "related": "PEP",
         "source": "Reuters", "summary": "PepsiCo is recalling...", "url": "https://example.com/a"},
        {"category": "company", "datetime": 1789560000, "headline": "No url", "id": 2, "related": "PEP", "url": ""},
        {"category": "company", "datetime": 1600000000, "headline": "Too old", "id": 3, "related": "PEP", "url": "https://example.com/old"},
    ], "PEP", START, END)
    assert [d.url for d in docs] == ["https://example.com/a"]
    assert docs[0].publisher == "Reuters" and docs[0].tickers == ["PEP"]
    with pytest.raises(ProviderError):
        parse_finnhub({"error": "API limit reached"}, "PEP", START, END)


def test_alphavantage_feed_and_quota_message():
    docs = parse_alphavantage({"feed": [{
        "title": "Coca-Cola raises guidance", "url": "https://example.com/ko", "time_published": "20260916T130000",
        "source": "Benzinga", "summary": "The company raised...", "ticker_sentiment": [{"ticker": "KO"}, {"ticker": "PEP"}],
    }]}, "KO")
    assert docs[0].published_at == datetime(2026, 9, 16, 13, tzinfo=UTC) and docs[0].tickers == ["KO", "PEP"]
    with pytest.raises(ProviderError, match="rate limit"):
        parse_alphavantage({"Information": "rate limit is 25 requests per day"}, "KO")


def test_rss_feed_parsing():
    xml = """<?xml version="1.0"?><rss version="2.0"><channel><title>IR News</title>
      <item><title>KO declares dividend</title><link>https://example.com/ko-div</link>
        <pubDate>Tue, 16 Sep 2026 12:00:00 GMT</pubDate><description>&lt;p&gt;Board declared &lt;b&gt;dividend&lt;/b&gt;&lt;/p&gt;</description></item>
      <item><title>Old</title><link>https://example.com/old</link><pubDate>Tue, 16 Sep 2025 12:00:00 GMT</pubDate></item>
    </channel></rss>"""
    docs = parse_feed(xml, "KO", "press_release", START, END)
    assert len(docs) == 1 and docs[0].publisher == "IR News" and docs[0].doc_type == "press_release"
    assert docs[0].content_text == "Board declared dividend"


def test_sec_recent_filings_and_items():
    sub = {"filings": {"recent": {
        "accessionNumber": ["0000021344-26-000050", "0000021344-26-000049"],
        "form": ["8-K", "10-Q"],
        "filingDate": ["2026-09-16", "2026-07-25"],
        "acceptanceDateTime": ["2026-09-16T16:05:12.000Z", "2026-07-25T10:00:00.000Z"],
        "items": ["5.02,9.01", ""],
        "primaryDocument": ["ko-8k.htm", "ko-10q.htm"],
    }}}
    docs = parse_recent_filings("KO", "21344", sub, date(2026, 9, 1))
    assert len(docs) == 1
    d = docs[0]
    assert d.filing_items == ["5.02", "9.01"] and d.accession == "0000021344-26-000050"
    assert d.published_at == datetime(2026, 9, 16, 16, 5, 12, tzinfo=UTC)
    assert d.url == "https://www.sec.gov/Archives/edgar/data/21344/000002134426000050/ko-8k.htm"


def test_sec_company_facts_keep_filing_dates_for_point_in_time_use():
    payload = {"facts": {"us-gaap": {
        "Revenues": {"units": {"USD": [
            {"start": "2026-04-01", "end": "2026-06-30", "val": 12.5e9, "accn": "a1", "fy": 2026, "fp": "Q2", "form": "10-Q", "filed": "2026-07-25"},
            {"start": "2025-04-01", "end": "2025-06-30", "val": 11.9e9, "accn": "a0", "fy": 2025, "fp": "Q2", "form": "10-Q", "filed": "2025-07-24"},
        ]}},
    }}}
    facts = parse_company_facts(payload)
    rev = [f for f in facts if f.metric == "revenue"]
    assert {f.filed_at for f in rev} == {date(2026, 7, 25), date(2025, 7, 24)}
    assert all(f.period_start is not None for f in rev)


def test_html_to_text_strips_scripts_and_tags():
    text = html_to_text("<html><head><title>x</title></head><script>alert(1)</script><p>Item 5.02</p><p>CEO resigns</p></html>")
    assert "alert" not in text and "<" not in text
    assert "Item 5.02" in text and "CEO resigns" in text
