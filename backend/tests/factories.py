"""Test fixtures. All data here is SIMULATED."""
from __future__ import annotations

from datetime import UTC, date, datetime, timedelta

import numpy as np
import pandas as pd

from app.research.market_calendar import trading_days_between
from app.scoring.types import CompanyProfile, EventEvidence, Fact, MarketData, NewsCoverage, PairInputs


def correlated_prices(n=800, rho=0.8, vol=0.012, drift_a=0.0, drift_b=0.0, seed=1, end=date(2026, 9, 30), break_at=None, rho_after=None):
    rng = np.random.default_rng(seed)
    days = trading_days_between(end - timedelta(days=int(n * 1.6)), end)[-n:]
    z1, z2 = rng.standard_normal(n), rng.standard_normal(n)
    r = np.full(n, rho)
    if break_at is not None:
        r[break_at:] = rho_after
    ra = vol * z1 + drift_a
    rb = vol * (r * z1 + np.sqrt(1 - r**2) * z2) + drift_b
    pa = 50 * np.exp(np.cumsum(ra))
    pb = 80 * np.exp(np.cumsum(rb))
    idx = pd.to_datetime(days)
    return pd.DataFrame({"a": pa, "b": pb}, index=idx)


def market(prices, bench=True, seed=3):
    b = None
    if bench:
        rng = np.random.default_rng(seed)
        b = pd.Series(400 * np.exp(np.cumsum(0.01 * rng.standard_normal(len(prices)))), index=prices.index)
    return MarketData(prices=prices, benchmark=b, is_total_return=True, provider="test")


def event(i=1, thread=None, direction="PEP_disadvantaged", scope="company_specific", severity=0.7, asym=0.9, persistence=90,
          published=None, ingested=None, verified=1.0, material=True, etype="product_recall", publishers=1, official=False):
    published = published or datetime(2026, 9, 20, tzinfo=UTC)
    return EventEvidence(
        id=i, thread_key=thread or f"t{i}", event_type=etype, scope=scope, relative_direction=direction,
        affected_tickers=["PEP"], severity=severity, asymmetry=asym, persistence_days=persistence,
        source_reliability=0.9, extraction_confidence=0.8, evidence_verified_ratio=verified,
        published_at=published, ingested_at=ingested or published + timedelta(hours=1), is_material=material,
        distinct_publishers=publishers, official_source=official, summary="simulated event",
    )


def coverage(ok=7, relevant=5, analyzed=5, last=None):
    return NewsCoverage(ok, ok, 7, relevant, analyzed, last or datetime(2026, 9, 30, 20, tzinfo=UTC))


def facts(ticker, rev_now, rev_prev, op_now, op_prev, eps_now, eps_prev, filed=date(2026, 8, 1)):
    out = []
    for metric, now, prev in (("revenue", rev_now, rev_prev), ("operating_income", op_now, op_prev), ("eps_diluted", eps_now, eps_prev)):
        out.append(Fact(metric, date(2026, 6, 30), 91, now, filed, f"{ticker}-1"))
        out.append(Fact(metric, date(2025, 6, 30), 91, prev, date(2025, 8, 1), f"{ticker}-0"))
    out.append(Fact("revenue", date(2025, 12, 31), 365, rev_now * 4, date(2026, 2, 10), f"{ticker}-k"))
    return out


def pair_inputs(prices=None, events=None, cov="default", fa=None, fb=None, cutoff=None, sic=("2086", "2086")):
    prices = prices if prices is not None else correlated_prices()
    return PairInputs(
        ticker_a="KO", ticker_b="PEP",
        cutoff=cutoff or datetime(2026, 9, 30, 22, tzinfo=UTC),
        market=market(prices),
        events=events,
        coverage=coverage() if cov == "default" else cov,
        profile_a=CompanyProfile("KO", sic[0], fa or []),
        profile_b=CompanyProfile("PEP", sic[1], fb or []),
    )
