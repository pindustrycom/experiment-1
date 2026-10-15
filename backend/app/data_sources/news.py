"""News adapters: Finnhub company news, Alpha Vantage NEWS_SENTIMENT, generic RSS/Atom feeds
(financial news feeds and company investor-relations feeds configured in Settings)."""
from __future__ import annotations

from datetime import UTC, datetime
from email.utils import parsedate_to_datetime

import feedparser

from app.core.config import get_settings
from app.data_sources.base import ProviderError, ProviderNotConfigured, RawDocument
from app.data_sources.http import request
from app.data_sources.sec import html_to_text


class FinnhubNews:
    name = "finnhub"

    def configured(self) -> bool:
        return bool(get_settings().secret("finnhub_api_key"))

    def fetch(self, ticker: str, start: datetime, end: datetime) -> list[RawDocument]:
        key = get_settings().secret("finnhub_api_key")
        if not key:
            raise ProviderNotConfigured(self.name, "FINNHUB_API_KEY")
        r = request(self.name, "GET", "https://finnhub.io/api/v1/company-news",
                    params={"symbol": ticker, "from": start.date().isoformat(), "to": end.date().isoformat()},
                    headers={"X-Finnhub-Token": key}, operation="company_news")
        return parse_finnhub(r.json(), ticker, start, end)


def parse_finnhub(payload, ticker: str, start: datetime, end: datetime) -> list[RawDocument]:
    if not isinstance(payload, list):
        raise ProviderError("finnhub", f"unexpected payload {str(payload)[:200]}", retryable=False)
    out = []
    for it in payload:
        try:
            ts = datetime.fromtimestamp(int(it["datetime"]), tz=UTC)
        except (KeyError, TypeError, ValueError):
            continue
        if not (start <= ts <= end) or not it.get("url") or not it.get("headline"):
            continue
        related = [t.strip().upper() for t in str(it.get("related") or ticker).split(",") if t.strip()]
        out.append(RawDocument(
            provider="finnhub", doc_type="news", url=it["url"], title=it["headline"].strip(), published_at=ts,
            publisher=it.get("source"), content_text=(it.get("summary") or "").strip() or None,
            tickers=sorted(set(related) | {ticker}), raw={"id": it.get("id"), "category": it.get("category")},
        ))
    return out


class AlphaVantageNews:
    name = "alphavantage"

    def configured(self) -> bool:
        return bool(get_settings().secret("alphavantage_api_key"))

    def fetch(self, ticker: str, start: datetime, end: datetime) -> list[RawDocument]:
        key = get_settings().secret("alphavantage_api_key")
        if not key:
            raise ProviderNotConfigured(self.name, "ALPHAVANTAGE_API_KEY")
        r = request(self.name, "GET", "https://www.alphavantage.co/query", params={
            "function": "NEWS_SENTIMENT", "tickers": ticker, "time_from": start.strftime("%Y%m%dT%H%M"),
            "time_to": end.strftime("%Y%m%dT%H%M"), "limit": 200, "sort": "LATEST", "apikey": key,
        }, operation="news_sentiment")
        return parse_alphavantage(r.json(), ticker)


def parse_alphavantage(payload: dict, ticker: str) -> list[RawDocument]:
    if "feed" not in payload:
        msg = payload.get("Information") or payload.get("Note") or payload.get("Error Message") or str(payload)[:200]
        raise ProviderError("alphavantage", f"no feed: {msg}", retryable=False)
    out = []
    for it in payload["feed"]:
        try:
            ts = datetime.strptime(it["time_published"], "%Y%m%dT%H%M%S").replace(tzinfo=UTC)
        except (KeyError, ValueError):
            continue
        tick = sorted({x.get("ticker", "").upper() for x in it.get("ticker_sentiment", []) if x.get("ticker")} | {ticker})
        out.append(RawDocument(
            provider="alphavantage", doc_type="news", url=it["url"], title=it.get("title", "").strip(), published_at=ts,
            publisher=it.get("source"), content_text=(it.get("summary") or "").strip() or None, tickers=tick,
            raw={"source_domain": it.get("source_domain"), "topics": it.get("topics", [])},
        ))
    return out


class RssFeeds:
    """Generic RSS/Atom adapter. Feeds come from settings: url templates containing {ticker}
    (financial news feeds) and per-ticker investor-relations feed URLs (doc_type press_release)."""

    name = "rss"

    def __init__(self, templates: list[str] | None = None, company_feeds: dict[str, list[str]] | None = None):
        self.templates = templates or []
        self.company_feeds = company_feeds or {}

    def configured(self) -> bool:
        return bool(self.templates or self.company_feeds)

    def feeds_for(self, ticker: str) -> list[tuple[str, str]]:
        return [(t.replace("{ticker}", ticker), "rss") for t in self.templates] + [
            (u, "press_release") for u in self.company_feeds.get(ticker, [])
        ]

    def fetch(self, ticker: str, start: datetime, end: datetime) -> list[RawDocument]:
        feeds = self.feeds_for(ticker)
        if not feeds:
            raise ProviderNotConfigured(self.name, f"no feeds configured for {ticker}")
        out, errors = [], []
        for url, doc_type in feeds:
            try:
                r = request(self.name, "GET", url, headers={"User-Agent": get_settings().sec_user_agent or "SplittabilityMonitor/1.0"}, operation="feed")
                out.extend(parse_feed(r.content, ticker, doc_type, start, end))
            except ProviderError as e:
                errors.append(str(e))
        if errors and not out and len(errors) == len(feeds):
            raise ProviderError(self.name, "; ".join(errors)[:500])
        return out


def _entry_time(e) -> datetime | None:
    for k in ("published_parsed", "updated_parsed"):
        if e.get(k):
            return datetime(*e[k][:6], tzinfo=UTC)
    for k in ("published", "updated"):
        if e.get(k):
            try:
                return parsedate_to_datetime(e[k]).astimezone(UTC)
            except (TypeError, ValueError):
                pass
    return None


def parse_feed(content: bytes | str, ticker: str, doc_type: str, start: datetime, end: datetime) -> list[RawDocument]:
    parsed = feedparser.parse(content)
    publisher = parsed.feed.get("title")
    out = []
    for e in parsed.entries:
        ts = _entry_time(e)
        if ts is None or not (start <= ts <= end) or not e.get("link"):
            continue
        summary = html_to_text(e.get("summary", "")) if e.get("summary") else None
        out.append(RawDocument(provider="rss", doc_type=doc_type, url=e["link"], title=(e.get("title") or "").strip(),
                               published_at=ts, publisher=publisher, content_text=summary, tickers=[ticker]))
    return out
