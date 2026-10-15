from __future__ import annotations

import logging
from datetime import date

from sqlalchemy import select
from sqlalchemy.dialects.postgresql import insert
from sqlalchemy.orm import Session

from app.data_sources.base import ProviderError, ProviderNotConfigured
from app.data_sources.sec import SecEdgar, parse_company_facts
from app.models import Company, CompanyFundamental
from app.scoring.types import CompanyProfile, Fact

log = logging.getLogger(__name__)


def sync_reference(db: Session) -> dict:
    """Verify CIKs against SEC company_tickers.json and refresh SIC codes."""
    sec = SecEdgar()
    if not sec.configured():
        raise ProviderNotConfigured("sec", "SEC_USER_AGENT")
    tmap = sec.ticker_map()
    out = {"updated": [], "missing": [], "errors": []}
    for c in db.scalars(select(Company).where(Company.is_benchmark.is_(False))).all():
        ref = tmap.get(c.ticker)
        if ref is None:
            out["missing"].append(c.ticker)
            c.listing_status = "not_in_sec_ticker_list"
            continue
        if c.cik != ref.cik:
            out["updated"].append(f"{c.ticker}: CIK {c.cik} -> {ref.cik}")
            c.cik = ref.cik
        c.listing_status = "listed"
        try:
            sub = sec.submissions(c.cik)
            c.sic_code, c.sic_description = sub.get("sic") or None, sub.get("sicDescription") or None
        except ProviderError as e:
            out["errors"].append(f"{c.ticker}: {e}")
    db.commit()
    return out


def ingest_fundamentals(db: Session) -> dict:
    sec = SecEdgar()
    if not sec.configured():
        raise ProviderNotConfigured("sec", "SEC_USER_AGENT")
    out = {"companies": {}, "errors": []}
    for c in db.scalars(select(Company).where(Company.is_benchmark.is_(False), Company.cik.is_not(None))).all():
        try:
            facts = parse_company_facts(sec.company_facts(c.cik))
        except ProviderError as e:
            out["errors"].append(f"{c.ticker}: {e}")
            continue
        n = upsert_facts(db, c.id, facts)
        db.commit()
        out["companies"][c.ticker] = {"facts": len(facts), "inserted": n}
    return out


def upsert_facts(db: Session, company_id: int, facts, is_demo: bool = False) -> int:
    rows = [dict(company_id=company_id, metric=f.metric, period_start=f.period_start, period_end=f.period_end,
                 duration_days=(f.period_end - f.period_start).days if f.period_start else None,
                 fiscal_period=f.fiscal_period, fiscal_year=f.fiscal_year, form=f.form, value=f.value, unit=f.unit,
                 filed_at=f.filed_at, accession=f.accession, provider="demo" if is_demo else "sec_companyfacts",
                 is_demo=is_demo, raw={"tag": f.tag}) for f in facts]
    n = 0
    for i in range(0, len(rows), 2000):
        stmt = insert(CompanyFundamental).values(rows[i:i + 2000]).on_conflict_do_nothing(constraint="uq_fundamental_fact")
        n += db.execute(stmt).rowcount or 0
    return n


def company_profile(db: Session, c: Company, cutoff: date) -> CompanyProfile:
    rows = db.scalars(select(CompanyFundamental).where(CompanyFundamental.company_id == c.id, CompanyFundamental.filed_at <= cutoff)).all()
    return CompanyProfile(
        ticker=c.ticker, sic_code=c.sic_code,
        facts=[Fact(r.metric, r.period_end, r.duration_days, r.value, r.filed_at, r.accession, r.form) for r in rows],
    )
