"""Market-price adapters.

Treatment of dividends and corporate actions
--------------------------------------------
* Tiingo (primary): `adjClose` is adjusted for splits and cash dividends (dividends reinvested
  on the ex-date), i.e. a total-return series. `divCash` and `splitFactor` are stored per bar.
  is_total_return = True.
* Stooq (no-key fallback): closes are split-adjusted; dividend treatment is not documented by
  the provider, so the series is stored with is_total_return = False and every result that uses
  it is labelled price-return only.
* Demo: synthetic, labelled is_demo.
A pair is always scored from a single provider for both legs so adjustment bases match.
"""
from __future__ import annotations

import csv
import io
from datetime import date

from app.core.config import get_settings
from app.data_sources.base import PriceBar, ProviderError, ProviderNotConfigured
from app.data_sources.http import request


class TiingoPrices:
    name = "tiingo"
    is_total_return = True

    def configured(self) -> bool:
        return bool(get_settings().secret("tiingo_api_key"))

    def fetch_daily(self, ticker: str, start: date, end: date) -> list[PriceBar]:
        key = get_settings().secret("tiingo_api_key")
        if not key:
            raise ProviderNotConfigured(self.name, "TIINGO_API_KEY")
        resp = request(
            self.name, "GET", f"https://api.tiingo.com/tiingo/daily/{ticker.replace('.', '-').lower()}/prices",
            params={"startDate": start.isoformat(), "endDate": end.isoformat(), "format": "json"},
            headers={"Authorization": f"Token {key}", "Content-Type": "application/json"},
            operation="daily_prices",
        )
        return parse_tiingo(resp.json())


def parse_tiingo(payload) -> list[PriceBar]:
    if not isinstance(payload, list):
        raise ProviderError("tiingo", f"unexpected payload: {str(payload)[:200]}", retryable=False)
    bars = []
    for row in payload:
        try:
            bars.append(PriceBar(
                date=date.fromisoformat(row["date"][:10]),
                close=float(row["close"]),
                adj_close=float(row["adjClose"]),
                volume=float(row.get("volume") or 0),
                dividend=float(row.get("divCash") or 0),
                split_factor=float(row.get("splitFactor") or 1),
            ))
        except (KeyError, TypeError, ValueError) as e:
            raise ProviderError("tiingo", f"malformed bar {row!r}: {e}", retryable=False) from e
    return bars


class StooqPrices:
    name = "stooq"
    is_total_return = False

    def configured(self) -> bool:
        return get_settings().enable_stooq

    def fetch_daily(self, ticker: str, start: date, end: date) -> list[PriceBar]:
        if not self.configured():
            raise ProviderNotConfigured(self.name, "ENABLE_STOOQ=false")
        resp = request(
            self.name, "GET", "https://stooq.com/q/d/l/",
            params={"s": f"{ticker.replace('.', '-').lower()}.us", "i": "d", "d1": start.strftime("%Y%m%d"), "d2": end.strftime("%Y%m%d")},
            operation="daily_prices",
        )
        return parse_stooq(resp.text)


def parse_stooq(text: str) -> list[PriceBar]:
    if not text.startswith("Date,"):
        raise ProviderError("stooq", f"unexpected response: {text[:120]!r}", retryable=False)
    bars = []
    for row in csv.DictReader(io.StringIO(text)):
        try:
            c = float(row["Close"])
        except (KeyError, ValueError):
            continue
        bars.append(PriceBar(date=date.fromisoformat(row["Date"]), close=c, adj_close=c, volume=float(row.get("Volume") or 0)))
    return bars


def configured_price_providers() -> list:
    return [p for p in (TiingoPrices(), StooqPrices()) if p.configured()]
