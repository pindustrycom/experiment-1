"""Price ingestion and point-in-time loading."""
from __future__ import annotations

import logging
from datetime import UTC, date, datetime, timedelta

import pandas as pd
from sqlalchemy import func, select
from sqlalchemy.dialects.postgresql import insert
from sqlalchemy.orm import Session

from app.data_sources.base import PriceBar, ProviderError
from app.data_sources.prices import StooqPrices, TiingoPrices
from app.models import Company, PriceObservation
from app.research.market_calendar import last_completed_session
from app.scoring.pit import truncate_market
from app.scoring.types import MarketData

log = logging.getLogger(__name__)
PROVIDERS = {"tiingo": TiingoPrices, "stooq": StooqPrices}
PRIORITY = ["tiingo", "stooq", "demo"]  # total-return providers first


def upsert_bars(db: Session, company_id: int, provider: str, bars: list[PriceBar], is_total_return: bool, is_demo: bool = False) -> int:
    if not bars:
        return 0
    rows = [dict(company_id=company_id, date=b.date, close=b.close, adj_close=b.adj_close, volume=b.volume,
                 dividend=b.dividend, split_factor=b.split_factor, provider=provider, is_total_return=is_total_return,
                 is_demo=is_demo) for b in bars]
    total = 0
    for i in range(0, len(rows), 2000):
        chunk = rows[i:i + 2000]
        stmt = insert(PriceObservation).values(chunk)
        # Total-return providers re-adjust history after each dividend, so refresh existing rows.
        stmt = stmt.on_conflict_do_update(
            constraint="uq_price_company_date_provider",
            set_={"close": stmt.excluded.close, "adj_close": stmt.excluded.adj_close, "volume": stmt.excluded.volume,
                  "dividend": stmt.excluded.dividend, "split_factor": stmt.excluded.split_factor, "ingested_at": func.now()},
        )
        db.execute(stmt)
        total += len(chunk)
    return total


def ingest_prices(db: Session, provider_names: list[str], history_start: date, now: datetime | None = None) -> dict:
    now = now or datetime.now(UTC)
    end = last_completed_session(now)
    providers = [PROVIDERS[n]() for n in provider_names if n in PROVIDERS]
    providers = [p for p in providers if p.configured()]
    if not providers:
        raise ProviderError("prices", "no market-data provider configured (set TIINGO_API_KEY or enable Stooq)", retryable=False)
    companies = db.scalars(select(Company).order_by(Company.ticker)).all()
    report: dict = {"end": str(end), "companies": {}, "errors": []}
    for c in companies:
        last = {p: d for p, d in db.execute(
            select(PriceObservation.provider, func.max(PriceObservation.date)).where(PriceObservation.company_id == c.id).group_by(PriceObservation.provider)).all()}
        done = False
        for p in providers:
            # Re-fetch a short overlap so late corrections and dividend re-adjustments are captured;
            # for total-return series re-fetch the full history after a dividend (cheap: one request).
            start = history_start if last.get(p.name) is None else last[p.name] - timedelta(days=10)
            if last.get(p.name) and last[p.name] >= end:
                report["companies"][c.ticker] = {"provider": p.name, "status": "up_to_date"}
                done = True
                break
            try:
                bars = p.fetch_daily(c.ticker, start, end)
                if p.is_total_return and any(b.dividend or b.split_factor != 1 for b in bars) and last.get(p.name):
                    bars = p.fetch_daily(c.ticker, history_start, end)
                n = upsert_bars(db, c.id, p.name, bars, p.is_total_return)
                db.commit()
                report["companies"][c.ticker] = {"provider": p.name, "bars": n, "last": str(bars[-1].date) if bars else None}
                done = True
                break
            except ProviderError as e:
                db.rollback()
                report["errors"].append(f"{c.ticker}/{p.name}: {e}")
                log.warning("price fetch failed", extra={"ticker": c.ticker, "provider": p.name, "error": str(e)})
        if not done:
            report["companies"][c.ticker] = {"status": "failed"}
    if companies and all(v.get("status") == "failed" for v in report["companies"].values()):
        # Nothing at all came back: report the run as failed instead of a silent "success".
        raise ProviderError("prices", f"every price request failed; first error: {report['errors'][0] if report['errors'] else 'unknown'}",
                            retryable=False)
    return report


def load_series(db: Session, company_id: int, provider: str | None = None, demo: bool | None = None) -> tuple[pd.Series, str | None, bool]:
    """Adjusted-close series from the best available provider for one company."""
    q = select(PriceObservation.provider, func.count(), func.max(PriceObservation.date)).where(PriceObservation.company_id == company_id)
    if demo is not None:
        q = q.where(PriceObservation.is_demo.is_(demo))
    avail = {p: (n, d) for p, n, d in db.execute(q.group_by(PriceObservation.provider)).all()}
    if provider is None:
        provider = next((p for p in PRIORITY if p in avail), next(iter(avail), None))
    if provider is None or provider not in avail:
        return pd.Series(dtype=float), None, False
    rows = db.execute(select(PriceObservation.date, PriceObservation.adj_close, PriceObservation.is_total_return)
                      .where(PriceObservation.company_id == company_id, PriceObservation.provider == provider)
                      .order_by(PriceObservation.date)).all()
    s = pd.Series([r[1] for r in rows], index=pd.to_datetime([r[0] for r in rows]), dtype=float)
    return s, provider, bool(rows and all(r[2] for r in rows))


def pair_market_data(db: Session, company_a: Company, company_b: Company, cutoff: datetime, benchmark: Company | None = None,
                     demo: bool | None = None) -> MarketData:
    """Both legs from one common provider (matching adjustment basis), truncated at the cutoff."""
    def providers(cid):
        q = select(PriceObservation.provider).where(PriceObservation.company_id == cid)
        if demo is not None:
            q = q.where(PriceObservation.is_demo.is_(demo))
        return set(db.scalars(q.distinct()).all())
    common = providers(company_a.id) & providers(company_b.id)
    prov = next((p for p in PRIORITY if p in common), next(iter(common), None))
    if prov is None:
        return MarketData(prices=pd.DataFrame(columns=["a", "b"]), benchmark=None, is_total_return=False, provider="none")
    sa, _, tra = load_series(db, company_a.id, prov)
    sb, _, trb = load_series(db, company_b.id, prov)
    df = pd.concat([sa.rename("a"), sb.rename("b")], axis=1, join="inner").dropna()
    df = truncate_market(df, cutoff)
    bench = None
    if benchmark is not None:
        sm, _, _ = load_series(db, benchmark.id, prov)
        if len(sm):
            bench = truncate_market(sm, cutoff)
    return MarketData(prices=df, benchmark=bench, is_total_return=tra and trb, provider=prov)


def last_price_update(db: Session) -> datetime | None:
    return db.scalar(select(func.max(PriceObservation.ingested_at)))
