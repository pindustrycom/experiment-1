"""DEMO MODE ONLY: deterministic, clearly-labelled SIMULATED data so the dashboard is usable before
any API key is configured. Every row written here has is_demo = true / provider 'demo', URLs use
the reserved .invalid domain, and the UI shows a persistent SIMULATED DATA banner.
Production mode refuses to run this module."""
from __future__ import annotations

import hashlib
import logging
from datetime import UTC, date, datetime, timedelta

import numpy as np
import pandas as pd
from sqlalchemy import func, select, text
from sqlalchemy.orm import Session

from app.core.config import get_settings
from app.data_sources.base import FactRecord, PriceBar, RawDocument
from app.models import Company, NewsFetchLog, Pair, PriceObservation
from app.research.market_calendar import last_completed_session, trading_days_between
from app.schemas.events import EventInterpretation
from app.services import backtest_service, fundamentals_service, news_service, predictions_service, prices_service, scoring_service
from app.services.reference import seed_reference_data

log = logging.getLogger(__name__)
START = date(2009, 1, 2)
HORIZON_END = date(2032, 12, 31)

SECTOR_OF = {
    "KO": "bev", "PEP": "bev", "WMT": "retail", "TGT": "retail", "HD": "home", "LOW": "home", "V": "pay", "MA": "pay",
    "XOM": "oil", "CVX": "oil", "AMD": "semi", "NVDA": "semi", "JPM": "bank", "BAC": "bank", "MS": "ib", "GS": "ib",
    "MCD": "rest", "YUM": "rest", "FDX": "parcel", "UPS": "parcel",
}
# (start date, daily drift) regimes that make some simulated pairs split apart
REGIMES = {
    "PEP": [(date(2024, 1, 2), -0.00045, 0.45)], "KO": [(date(2024, 1, 2), 0.00030, 0.9)],
    "NVDA": [(date(2023, 1, 3), 0.0020, 0.6)],
    "TGT": [(date(2025, 2, 3), -0.0007, 0.6)], "UPS": [(date(2025, 6, 2), -0.0006, 0.7)],
}


def _seed(name: str) -> int:
    return int(hashlib.sha256(name.encode()).hexdigest()[:8], 16)


def _days() -> list[date]:
    return trading_days_between(START, HORIZON_END)


def simulate_paths(tickers: list[str]) -> dict[str, np.ndarray]:
    days = _days()
    n = len(days)
    rng = np.random.default_rng(42)
    market = 0.0003 + 0.010 * rng.standard_normal(n)
    sectors = {s: 0.008 * np.random.default_rng(_seed("sector" + s)).standard_normal(n) for s in set(SECTOR_OF.values())}
    out = {}
    didx = np.array([d.toordinal() for d in days])
    for t in tickers:
        r = np.random.default_rng(_seed(t))
        beta = 0.7 + 0.6 * r.random()
        idio = (0.006 + 0.006 * r.random()) * r.standard_normal(n)
        sec = sectors.get(SECTOR_OF.get(t, ""), np.zeros(n))
        load = np.ones(n)
        drift = np.full(n, 0.0001 * r.standard_normal())
        for start, dr, ld in REGIMES.get(t, []):
            m = didx >= start.toordinal()
            drift[m] = dr
            load[m] = ld
        rets = (market if t != "SPY" else market) * (1.0 if t == "SPY" else beta) + (0 if t == "SPY" else 1) * load * sec + (0.002 * r.standard_normal(n) if t == "SPY" else idio) + drift
        out[t] = (20 + 180 * r.random()) * np.exp(np.cumsum(rets))
    return out


def _require_demo() -> None:
    if not get_settings().is_demo:
        raise RuntimeError("demo data can only be generated when APP_MODE=demo")


def insert_prices(db: Session, until: date) -> int:
    companies = db.scalars(select(Company)).all()
    paths = simulate_paths([c.ticker for c in companies])
    days = _days()
    total = 0
    for c in companies:
        last = db.scalar(select(func.max(PriceObservation.date)).where(PriceObservation.company_id == c.id, PriceObservation.provider == "demo"))
        bars = [PriceBar(d, float(p), float(p), 1e6) for d, p in zip(days, paths[c.ticker], strict=False) if d <= until and (last is None or d > last)]
        total += prices_service.upsert_bars(db, c.id, "demo", bars, is_total_return=True, is_demo=True)
    return total


def insert_fundamentals(db: Session, until: date) -> None:
    for c in db.scalars(select(Company).where(Company.is_benchmark.is_(False))).all():
        r = np.random.default_rng(_seed("fund" + c.ticker))
        base, g, margin = 2e9 + 20e9 * r.random(), 0.01 + 0.015 * r.random(), 0.1 + 0.2 * r.random()
        shares = 1e9 * (0.5 + r.random())
        facts = []
        for k, q in enumerate(pd.date_range("2012-03-31", until, freq="QE")):
            q_end = q.date()
            shock = -0.03 if (c.ticker == "PEP" and q_end >= date(2024, 3, 1)) else 0.0
            rev = base * np.exp((g + shock) * k + 0.02 * r.standard_normal())
            m = margin + (shock * 2) + 0.01 * r.standard_normal()
            filed = q_end + timedelta(days=40)
            if filed > until:
                continue
            start = q_end - timedelta(days=90)
            acc = f"DEMO-{c.ticker}-{q_end:%Y%m}"
            for metric, val, unit in (("revenue", rev, "USD"), ("operating_income", rev * m, "USD"), ("eps_diluted", rev * m * 0.75 / shares, "USD/shares")):
                facts.append(FactRecord(metric, start, q_end, float(val), unit, filed, acc, "10-Q", q_end.year, f"Q{(q_end.month - 1) // 3 + 1}", "demo"))
            if q_end.month == 12 and q_end + timedelta(days=55) <= until:
                facts.append(FactRecord("revenue", date(q_end.year, 1, 1), q_end, float(rev * 4), "USD", q_end + timedelta(days=55),
                                        f"DEMO-{c.ticker}-{q_end:%Y}K", "10-K", q_end.year, "FY", "demo"))
        fundamentals_service.upsert_facts(db, c.id, facts, is_demo=True)


DEMO_STORIES = [
    ("KO", "PEP", date(2024, 2, 9), "PEP", "product_recall", "b_disadvantaged", "company_specific", 0.6, 0.85, 120,
     "PepsiCo unit announces voluntary recall of several snack lines after a supplier contamination finding",
     "The company said the voluntary recall covers several snack product lines distributed nationally. It expects the recall to reduce quarterly segment revenue and said production at the affected supplier has been suspended."),
    ("KO", "PEP", date(2024, 10, 8), "PEP", "guidance_change", "b_disadvantaged", "company_specific", 0.55, 0.8, 180,
     "PepsiCo lowers full-year organic revenue outlook citing weaker North American snack volumes",
     "PepsiCo lowered its full-year organic revenue growth outlook. Management cited weaker North American snack volumes and said the recall continued to weigh on results."),
    ("KO", "PEP", date(2025, 2, 11), "KO", "earnings_surprise", "a_advantaged", "company_specific", 0.45, 0.7, 90,
     "Coca-Cola reports quarterly organic revenue growth above expectations on pricing and volume",
     "Coca-Cola reported organic revenue growth above analyst expectations, driven by both pricing and unit case volume growth across regions."),
    ("KO", "PEP", date(2025, 6, 3), "KO", "industry_shock", "symmetric", "industry_wide", 0.4, 0.15, 60,
     "Aluminium can costs rise for beverage makers after tariff change",
     "Beverage makers face higher aluminium can costs after a tariff change. Analysts said the cost increase affects the entire beverage industry."),
    ("AMD", "NVDA", date(2023, 5, 24), "NVDA", "guidance_change", "b_advantaged", "company_specific", 0.85, 0.85, 365,
     "Nvidia issues revenue forecast far above estimates on data-center AI demand",
     "Nvidia forecast second-quarter revenue far above analyst estimates, citing surging demand for its data-center chips used in artificial intelligence."),
    ("WMT", "TGT", date(2025, 3, 4), "TGT", "consumer_demand_change", "b_disadvantaged", "company_specific", 0.5, 0.75, 120,
     "Target warns of profit pressure as discretionary spending weakens",
     "Target warned of first-quarter profit pressure as shoppers pulled back on discretionary purchases, while it said essentials sales held up."),
    ("FDX", "UPS", date(2025, 4, 29), "UPS", "major_partnership", "b_disadvantaged", "company_specific", 0.5, 0.8, 365,
     "UPS to cut deliveries for its largest customer by half over 18 months",
     "UPS said it reached an agreement to lower delivery volume for its largest customer by more than 50% by the second half of 2026 and announced workforce reductions."),
    ("XOM", "CVX", date(2025, 9, 10), "XOM", "industry_shock", "symmetric", "industry_wide", 0.5, 0.1, 90,
     "Oil prices fall as OPEC+ raises output targets",
     "Crude oil prices fell after OPEC+ agreed to raise output targets, a move analysts said would weigh on earnings across integrated oil producers."),
]


# Recent stories dated relative to the seeding day (days ago) so a fresh demo shows live-looking activity.
RECENT_DEMO_STORIES = [
    ("KO", "PEP", 18, "PEP", "executive_change", "unclear", "company_specific", 0.5, 0.8, 365,
     "PepsiCo names new chief executive as current CEO steps down",
     "PepsiCo said its chief executive will step down at the end of the quarter and named a successor from outside the company. The board said it expects a review of the snack and beverage portfolio."),
    ("HD", "LOW", 6, "LOW", "acquisition_merger", "unclear", "company_specific", 0.45, 0.7, 365,
     "Lowe's completes acquisition of a professional building-supply distributor",
     "Lowe's completed its acquisition of a professional building-supply distributor, expanding its business with contractors. The company said the deal adds about 8% to annual revenue."),
    ("V", "MA", 12, "V", "regulatory_change", "symmetric", "industry_wide", 0.5, 0.1, 180,
     "Card-network fee legislation advances in Senate committee",
     "A bill that would require large banks to offer an alternative card network for credit transactions advanced in a Senate committee. Analysts said the bill would affect both major card networks."),
    ("JPM", "BAC", 3, "BAC", "earnings_surprise", "b_disadvantaged", "company_specific", 0.35, 0.7, 60,
     "Bank of America trading revenue falls short of estimates",
     "Bank of America reported quarterly trading revenue below analyst estimates, while net interest income was in line with its guidance."),
]


def insert_news(db: Session, until: datetime, include_recent: bool = False) -> None:
    companies = db.scalars(select(Company)).all()
    by_t = {c.ticker: c for c in companies}
    # daily successful "demo" fetches so news coverage is measurable
    have = {(r[0], r[1]) for r in db.execute(select(NewsFetchLog.company_id, func.date(NewsFetchLog.fetched_at)).where(NewsFetchLog.provider == "demo")).all()}
    d = date(2014, 1, 1)
    rows = []
    while d <= until.date():
        for c in companies:
            if c.is_benchmark or (c.id, d) in have:
                continue
            ts = datetime(d.year, d.month, d.day, 14, tzinfo=UTC)
            if ts <= until:
                rows.append(dict(provider="demo", company_id=c.id, window_start=ts - timedelta(hours=1), window_end=ts, fetched_at=ts,
                                 success=True, documents_found=0, is_demo=True))
        d += timedelta(days=1)
    if rows:
        db.execute(NewsFetchLog.__table__.insert(), rows)
    stories = list(DEMO_STORIES)
    if include_recent:
        stories += [(ta, tb, (until - timedelta(days=ago)).date(), *rest) for ta, tb, ago, *rest in RECENT_DEMO_STORIES]
    for ta, tb, day, subj, etype, direction, scope, sev, asym, persist, title, body in stories:
        pub = datetime(day.year, day.month, day.day, 13, 30, tzinfo=UTC)
        if pub > until:
            continue
        pair = db.scalar(select(Pair).where(Pair.pair_key == "|".join(sorted([ta, tb]))))
        if pair is None:
            continue
        raw = RawDocument(provider="demo", doc_type="news", url=f"https://news.demo.invalid/{subj.lower()}/{day.isoformat()}-{etype}",
                          title=f"[SIMULATED] {title}", published_at=pub, publisher="Demo Newswire (simulated)", content_text=body, tickers=[subj])
        sd, created = news_service.store_document(db, raw, [by_t[subj]] + [by_t[x] for x in (ta, tb) if x != subj], is_demo=True)
        if not created or sd is None:
            continue
        sd.ingested_at = pub + timedelta(minutes=20)
        art = sd.article
        art.first_ingested_at = pub + timedelta(minutes=20)
        side_a = pair.company_a.ticker == subj
        dmap = {"b_disadvantaged": "b_disadvantaged" if not side_a else "a_disadvantaged", "a_advantaged": "a_advantaged" if side_a else "b_advantaged",
                "b_advantaged": "b_advantaged" if not side_a else "a_advantaged", "symmetric": "symmetric", "unclear": "unclear"}
        interp = EventInterpretation(
            is_relevant=True, is_material=True, event_type=etype, affected_sides=["a" if side_a else "b"] if scope == "company_specific" else ["a", "b"],
            scope=scope, relative_direction=dmap[direction], severity=sev, asymmetry=asym, expected_persistence_days=persist,
            extraction_confidence=0.8, is_new_information=True, related_prior_event_id=None,
            summary=f"SIMULATED: {body.split('. ')[0]}.", company_exposure=[], evidence=[{"quote": body.split(". ")[0], "supports": "main fact"}],
            shared_industry_effects=None if scope == "company_specific" else "affects both companies", uncertainties=["simulated demo item"],
        )
        news_service.persist_event(db, art, sd, pair, interp, "demo-fixture (simulated, no AI call)", {"demo": True}, is_demo=True,
                                   analyzed_at=pub + timedelta(minutes=30))
        art.analysis_status = "analyzed"
    db.flush()


def seed(db: Session, backfill_snapshots: bool = True) -> dict:
    _require_demo()
    seed_reference_data(db)
    backtest_service.seed_reported_benchmark(db)
    db.commit()
    now = datetime.now(UTC)
    last = last_completed_session(now)
    n = insert_prices(db, last)
    insert_fundamentals(db, last)
    insert_news(db, now, include_recent=True)
    db.commit()
    out = {"prices": n}
    if backfill_snapshots:
        # Simulated forward-test history: monthly snapshots from mid-2022, produced with the same
        # code path as live snapshots at those (past) cutoffs, then resolved against simulated prices.
        d = date(2022, 7, 1)
        while d < last:
            cutoff = datetime(d.year, d.month, d.day, 22, 0, tzinfo=UTC)
            predictions_service.snapshot_predictions(db, now=cutoff)
            d = date(d.year + (d.month == 12), d.month % 12 + 1, 1)
        for k in range(10, 0, -1):  # recent daily-ish history for charts
            cutoff = now - timedelta(days=3 * k)
            scoring_service.score_all(db, trigger="backfill", cutoff=cutoff)
        out["outcomes"] = predictions_service.evaluate_outcomes(db, now)
        run = backtest_service.run_walk_forward(db, train=(2015, 2019), embargo=(2020, 2021), test=(2022, 2024), n_boot=200)
        out["backtest_run"] = run.id
    out["scores"] = scoring_service.score_all(db, trigger="scheduled")
    db.commit()
    return out


def refresh(db: Session) -> dict:
    """Extends simulated prices to the latest completed session and records a simulated news fetch."""
    _require_demo()
    now = datetime.now(UTC)
    n = insert_prices(db, last_completed_session(now))
    insert_news(db, now)
    db.commit()
    return {"new_price_rows": n}


def is_seeded(db: Session) -> bool:
    return bool(db.scalar(select(func.count()).select_from(PriceObservation).where(PriceObservation.provider == "demo")))


def demo_data_present(db: Session) -> bool:
    """True if any SIMULATED rows exist. Production mode refuses to run on such a database rather
    than silently mixing simulated and real data."""
    from app.models import ExtractedEvent, IndexScore, NewsArticle, PredictionSnapshot
    for m in (PriceObservation, NewsArticle, ExtractedEvent, IndexScore, PredictionSnapshot):
        if db.scalar(select(m.id).where(m.is_demo.is_(True)).limit(1)) is not None:
            return True
    return False


def wipe(db: Session) -> None:
    """Remove ALL research data (prices, news, scores, predictions). Used once when converting a demo
    database to production; the CLI refuses if any non-demo data exists."""
    for t in ["realized_outcomes", "prediction_snapshots", "si_component_calculations", "index_scores", "alerts", "extracted_events",
              "document_companies", "source_documents", "news_articles", "news_fetch_log", "company_fundamentals", "price_observations", "backtest_runs"]:
        db.execute(text(f"TRUNCATE {t} CASCADE"))
    db.commit()
