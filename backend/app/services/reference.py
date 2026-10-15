"""Companies, pairs, watchlists and model-version registry."""
from __future__ import annotations

import logging
import re
from datetime import UTC, datetime

from sqlalchemy import select
from sqlalchemy.orm import Session

from app.models import Company, ModelVersion, Pair, Watchlist, WatchlistPair
from app.scoring.params import si_v1_definition, si_v2_definition

log = logging.getLogger(__name__)

# CIKs are seeded for convenience and re-verified against SEC company_tickers.json by the
# reference-data job whenever SEC access is configured.
SEED_COMPANIES = [
    ("KO", "The Coca-Cola Company", "0000021344", "Consumer Staples", "Beverages"),
    ("PEP", "PepsiCo, Inc.", "0000077476", "Consumer Staples", "Beverages & Snacks"),
    ("WMT", "Walmart Inc.", "0000104169", "Consumer Staples", "Mass Retail"),
    ("TGT", "Target Corporation", "0000027419", "Consumer Discretionary", "Mass Retail"),
    ("HD", "The Home Depot, Inc.", "0000354950", "Consumer Discretionary", "Home Improvement Retail"),
    ("LOW", "Lowe's Companies, Inc.", "0000060667", "Consumer Discretionary", "Home Improvement Retail"),
    ("V", "Visa Inc.", "0001403161", "Financials", "Payment Networks"),
    ("MA", "Mastercard Incorporated", "0001141391", "Financials", "Payment Networks"),
    ("XOM", "Exxon Mobil Corporation", "0000034088", "Energy", "Integrated Oil & Gas"),
    ("CVX", "Chevron Corporation", "0000093410", "Energy", "Integrated Oil & Gas"),
    ("AMD", "Advanced Micro Devices, Inc.", "0000002488", "Information Technology", "Semiconductors"),
    ("NVDA", "NVIDIA Corporation", "0001045810", "Information Technology", "Semiconductors"),
    ("JPM", "JPMorgan Chase & Co.", "0000019617", "Financials", "Diversified Banks"),
    ("BAC", "Bank of America Corporation", "0000070858", "Financials", "Diversified Banks"),
    ("MS", "Morgan Stanley", "0000895421", "Financials", "Investment Banking & Brokerage"),
    ("GS", "The Goldman Sachs Group, Inc.", "0000886982", "Financials", "Investment Banking & Brokerage"),
    ("MCD", "McDonald's Corporation", "0000063908", "Consumer Discretionary", "Restaurants"),
    ("YUM", "Yum! Brands, Inc.", "0001041061", "Consumer Discretionary", "Restaurants"),
    ("FDX", "FedEx Corporation", "0001048911", "Industrials", "Air Freight & Logistics"),
    ("UPS", "United Parcel Service, Inc.", "0001090727", "Industrials", "Air Freight & Logistics"),
]
BENCHMARK = ("SPY", "SPDR S&P 500 ETF Trust (benchmark)")

SEED_PAIRS = [
    ("KO", "PEP", "Beverages"), ("WMT", "TGT", "Mass Retail"), ("HD", "LOW", "Home Improvement Retail"),
    ("V", "MA", "Payment Networks"), ("XOM", "CVX", "Integrated Oil & Gas"), ("AMD", "NVDA", "Semiconductors"),
    ("JPM", "BAC", "Diversified Banks"), ("MS", "GS", "Investment Banking & Brokerage"),
    ("MCD", "YUM", "Restaurants"), ("FDX", "UPS", "Air Freight & Logistics"),
    ("LOW", "HD", "Home Improvement Retail"),  # reversed duplicate of HD/LOW: deduplicated on seed
]

TICKER_RE = re.compile(r"^[A-Z][A-Z0-9.\-]{0,9}$")


class PairExists(Exception):
    def __init__(self, pair: Pair):
        super().__init__(f"pair already exists as {pair.slug}")
        self.pair = pair


def pair_key(a: str, b: str) -> str:
    return "|".join(sorted([a.upper(), b.upper()]))


def get_or_create_company(db: Session, ticker: str, name: str | None = None, **kw) -> Company:
    ticker = ticker.upper().strip()
    if not TICKER_RE.match(ticker):
        raise ValueError(f"invalid ticker '{ticker}'")
    c = db.scalar(select(Company).where(Company.ticker == ticker))
    if c is None:
        c = Company(ticker=ticker, name=name or ticker, aliases=[], **kw)
        db.add(c)
        db.flush()
    return c


def create_pair(db: Session, ta: str, tb: str, industry: str | None = None, **kw) -> Pair:
    ta, tb = ta.upper().strip(), tb.upper().strip()
    if ta == tb:
        raise ValueError("a pair needs two different tickers")
    existing = db.scalar(select(Pair).where(Pair.pair_key == pair_key(ta, tb)))
    if existing:
        raise PairExists(existing)
    a, b = get_or_create_company(db, ta), get_or_create_company(db, tb)
    p = Pair(slug=f"{ta}-{tb}", pair_key=pair_key(ta, tb), company_a_id=a.id, company_b_id=b.id,
             industry=industry or (a.industry if a.industry == b.industry else None), **kw)
    db.add(p)
    db.flush()
    return p


def seed_reference_data(db: Session) -> dict:
    created_c = created_p = skipped = 0
    for t, name, cik, sector, industry in SEED_COMPANIES:
        if db.scalar(select(Company).where(Company.ticker == t)) is None:
            db.add(Company(ticker=t, name=name, cik=cik, sector=sector, industry=industry, aliases=_aliases(name)))
            created_c += 1
    if db.scalar(select(Company).where(Company.ticker == BENCHMARK[0])) is None:
        db.add(Company(ticker=BENCHMARK[0], name=BENCHMARK[1], is_benchmark=True, aliases=[]))
    db.flush()
    for ta, tb, ind in SEED_PAIRS:
        try:
            create_pair(db, ta, tb, ind)
            created_p += 1
        except PairExists as e:
            skipped += 1
            if e.pair.company_a.ticker != ta:  # reversed duplicate in the requested list (e.g. LOW/HD vs HD/LOW)
                log.info("seed pair %s/%s is the same pair as %s; kept once", ta, tb, e.pair.slug)
    wl = db.scalar(select(Watchlist).where(Watchlist.name == "Core pairs"))
    if wl is None:
        wl = Watchlist(name="Core pairs", description="Initial research watchlist", is_default=True)
        db.add(wl)
        db.flush()
        for p in db.scalars(select(Pair)).all():
            db.add(WatchlistPair(watchlist_id=wl.id, pair_id=p.id))
    ensure_model_versions(db)
    return {"companies_created": created_c, "pairs_created": created_p, "pairs_already_present": skipped}


def _aliases(name: str) -> list[str]:
    base = re.sub(r"[,.]|\b(Inc|Corporation|Corp|Company|Co|Incorporated|Holdings|Group|The|plc)\b", " ", name)
    base = re.sub(r"\s+", " ", base).strip()
    return [base] if base and base != name else []


def ensure_model_versions(db: Session, weights: dict | None = None, params: dict | None = None) -> tuple[ModelVersion, ModelVersion]:
    """Register (idempotently) the SI-v1 definition and the SI-v2 definition for the current weights,
    and mark the latter active. Older versions are kept forever."""
    v1 = _upsert_version(db, si_v1_definition())
    v2 = _upsert_version(db, si_v2_definition(weights, params))
    for mv in db.scalars(select(ModelVersion).where(ModelVersion.index_name == "si_v2", ModelVersion.id != v2.id, ModelVersion.is_active.is_(True))):
        mv.is_active = False
    v1.is_active = v2.is_active = True
    db.flush()
    return v1, v2


def _upsert_version(db: Session, d: dict) -> ModelVersion:
    mv = db.scalar(select(ModelVersion).where(ModelVersion.index_name == d["index_name"], ModelVersion.content_hash == d["content_hash"]))
    if mv is None:
        mv = ModelVersion(index_name=d["index_name"], version=d["version"], weights=d["weights"], params=d["params"],
                          description=d["description"], content_hash=d["content_hash"], is_active=False,
                          created_at=datetime.now(UTC))
        db.add(mv)
        db.flush()
    return mv
