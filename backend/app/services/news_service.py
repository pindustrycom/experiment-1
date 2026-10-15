"""News ingestion (facts), clustering (dedup) and Claude analysis (interpretations)."""
from __future__ import annotations

import logging
import re
from datetime import UTC, datetime, timedelta

from sqlalchemy import func, select
from sqlalchemy.orm import Session

from app.agents.news_analyst import (
    PROMPT_VERSION,
    SCHEMA_VERSION,
    NewsAnalyst,
    build_user_message,
    estimate_cost,
    verify_evidence,
)
from app.core.config import get_settings
from app.data_sources.base import ProviderError, RawDocument
from app.data_sources.news import AlphaVantageNews, FinnhubNews, RssFeeds
from app.data_sources.reliability import reliability_for
from app.data_sources.sec import ROUTINE_ITEMS, SecEdgar
from app.models import Company, DocumentCompany, ExtractedEvent, NewsArticle, NewsFetchLog, Pair, ProviderUsage, SourceDocument
from app.schemas.events import direction_to_ticker_form
from app.scoring.types import EventEvidence, NewsCoverage
from app.services import dedup, settings_service, usage

log = logging.getLogger(__name__)


class SecNewsAdapter:
    """Adapts SEC 8-K filings to the news-provider interface."""

    name = "sec"

    def __init__(self, ciks: dict[str, str]):
        self.sec, self.ciks = SecEdgar(), ciks

    def configured(self) -> bool:
        return self.sec.configured()

    def fetch(self, ticker: str, start: datetime, end: datetime) -> list[RawDocument]:
        cik = self.ciks.get(ticker)
        if not cik:
            raise ProviderError("sec", f"no CIK for {ticker}", retryable=False)
        docs, _ = self.sec.recent_filings(ticker, cik, start.date())
        return [d for d in docs if start <= d.published_at <= end]


def monitored_companies(db: Session) -> list[Company]:
    ids = set()
    for p in db.scalars(select(Pair).where(Pair.status == "active")).all():
        ids |= {p.company_a_id, p.company_b_id}
    return db.scalars(select(Company).where(Company.id.in_(ids)).order_by(Company.ticker)).all() if ids else []


def build_providers(db: Session, cfg: dict) -> list:
    companies = monitored_companies(db)
    prov = []
    enabled = cfg["providers"]
    if enabled.get("sec"):
        prov.append(SecNewsAdapter({c.ticker: c.cik for c in companies if c.cik}))
    if enabled.get("finnhub"):
        prov.append(FinnhubNews())
    if enabled.get("alphavantage"):
        prov.append(AlphaVantageNews())
    if enabled.get("rss"):
        prov.append(RssFeeds(cfg.get("rss_templates"), cfg.get("company_feeds")))
    return [p for p in prov if p.configured()]


def mentioned_companies(doc: RawDocument, companies: list[Company]) -> list[Company]:
    text = f"{doc.title}\n{doc.content_text or ''}"
    out = []
    for c in companies:
        if c.ticker in doc.tickers:
            out.append(c)
            continue
        names = [c.name] + list(c.aliases or [])
        if any(len(n) > 3 and re.search(r"\b" + re.escape(n) + r"\b", text, re.I) for n in names):
            out.append(c)
    return out


def store_document(db: Session, doc: RawDocument, companies: list[Company], is_demo: bool = False) -> tuple[SourceDocument | None, bool]:
    """Insert a source document and attach it to a story cluster. Returns (doc, created)."""
    uh = dedup.url_hash(doc.url)
    existing = db.scalar(select(SourceDocument).where(SourceDocument.url_hash == uh))
    if existing:
        if existing.article:
            existing.article.last_seen_at = datetime.now(UTC)
        return existing, False
    mentioned = mentioned_companies(doc, companies)
    if not mentioned:
        return None, False
    tickers = sorted({c.ticker for c in mentioned})
    ch = dedup.content_hash(doc.content_text)
    sd = SourceDocument(
        provider=doc.provider, doc_type=doc.doc_type, url=doc.url, url_hash=uh, title=doc.title[:2000], publisher=doc.publisher,
        published_at=doc.published_at, content_text=doc.content_text, content_hash=ch, tickers=tickers,
        filing_form=doc.filing_form, filing_items=doc.filing_items, accession=doc.accession,
        source_reliability=reliability_for(doc.doc_type, doc.publisher), is_demo=is_demo, raw=doc.raw,
    )
    sd.article = assign_cluster(db, sd)
    db.add(sd)
    db.flush()
    for c in mentioned:
        db.add(DocumentCompany(document_id=sd.id, company_id=c.id))
    return sd, True


def assign_cluster(db: Session, sd: SourceDocument) -> NewsArticle:
    window = (sd.published_at - timedelta(days=3), sd.published_at + timedelta(days=3))
    if sd.doc_type != "filing":
        cands = db.scalars(select(NewsArticle).where(NewsArticle.first_published_at.between(*window))).all()
        for art in cands:
            if not set(art.tickers) & set(sd.tickers):
                continue
            docs = art.documents
            if any(d.doc_type == "filing" for d in docs):
                continue
            if (sd.content_hash and any(d.content_hash == sd.content_hash for d in docs)) or any(dedup.same_story(sd.title, d.title) for d in docs):
                art.document_count += 1
                pubs = {d.publisher for d in docs} | {sd.publisher}
                art.distinct_publishers = len({p for p in pubs if p})
                art.tickers = sorted(set(art.tickers) | set(sd.tickers))
                art.last_seen_at = datetime.now(UTC)
                if art.analysis_status in ("analyzed", "skipped") and sd.source_reliability > max(d.source_reliability for d in docs):
                    art.analysis_status = "pending"  # a more authoritative source arrived: re-analyse
                return art
    art = NewsArticle(
        cluster_key=dedup.cluster_key_for(sd.title + (sd.accession or ""), sd.published_at.date().isoformat(), sd.tickers),
        headline=sd.title, first_published_at=sd.published_at, document_count=1, distinct_publishers=1,
        tickers=list(sd.tickers), is_demo=sd.is_demo,
    )
    db.add(art)
    db.flush()
    return art


def ingest_news(db: Session, now: datetime | None = None) -> dict:
    now = now or datetime.now(UTC)
    cfg = settings_service.get_section(db, "news")
    providers = build_providers(db, cfg)
    companies = monitored_companies(db)
    if not providers:
        raise ProviderError("news", "no news provider configured (SEC_USER_AGENT, FINNHUB_API_KEY, ALPHAVANTAGE_API_KEY or RSS feeds)", retryable=False)
    start = now - timedelta(hours=int(cfg["lookback_hours"]))
    report = {"providers": [p.name for p in providers], "new_documents": 0, "duplicates": 0, "errors": []}
    for c in companies:
        for p in providers:
            try:
                docs = p.fetch(c.ticker, start, now)
                new = 0
                for d in docs:
                    sd, created = store_document(db, d, companies)
                    new += created
                    report["duplicates"] += (sd is not None and not created)
                db.add(NewsFetchLog(provider=p.name, company_id=c.id, window_start=start, window_end=now, success=True, documents_found=len(docs)))
                db.commit()
                report["new_documents"] += new
            except ProviderError as e:
                db.rollback()
                db.add(NewsFetchLog(provider=p.name, company_id=c.id, window_start=start, window_end=now, success=False, error=str(e)[:1000]))
                db.commit()
                report["errors"].append(f"{c.ticker}/{p.name}: {e}")
    return report


def _primary_document(art: NewsArticle) -> SourceDocument:
    return max(art.documents, key=lambda d: (d.source_reliability, len(d.content_text or "")))


def _prior_events(db: Session, pair_id: int, before: datetime) -> list[dict]:
    rows = db.scalars(select(ExtractedEvent).where(ExtractedEvent.pair_id == pair_id, ExtractedEvent.is_material.is_(True),
                                                   ExtractedEvent.published_at <= before)
                      .order_by(ExtractedEvent.published_at.desc()).limit(25)).all()
    return [{"id": e.id, "date": e.published_at.date().isoformat(), "type": e.event_type, "direction": e.relative_direction, "summary": e.summary[:240]} for e in reversed(rows)]


def pairs_for_article(db: Session, art: NewsArticle) -> list[Pair]:
    tick = set(art.tickers)
    return [p for p in db.scalars(select(Pair).where(Pair.status == "active")).all() if {p.company_a.ticker, p.company_b.ticker} & tick]


def analyze_pending(db: Session, analyst: NewsAnalyst | None = None, now: datetime | None = None) -> dict:
    now = now or datetime.now(UTC)
    s = get_settings()
    llm = settings_service.get_section(db, "llm")
    budget = settings_service.get_section(db, "budget")
    pending = db.scalars(select(NewsArticle).where(NewsArticle.analysis_status.in_(("pending", "budget_blocked", "not_configured", "failed")))
                         .order_by(NewsArticle.first_published_at.desc()).limit(500)).all()
    report = {"analyzed": 0, "skipped": 0, "failed": 0, "budget_blocked": 0, "not_configured": 0, "events": 0, "material_pairs": []}
    material_pairs: set[int] = set()
    if analyst is None and s.secret("anthropic_api_key"):
        analyst = NewsAnalyst(model=llm["model"], effort=llm["effort"], pricing=llm["pricing"], refusal_fallback=llm["refusal_fallback"])
    calls = 0
    for art in pending:
        doc = _primary_document(art)
        pairs = pairs_for_article(db, art)
        if not pairs:
            art.analysis_status = "skipped"
            art.analysis_error = "no active pair mentions these companies"
            report["skipped"] += 1
            continue
        if doc.doc_type == "filing" and doc.filing_items and set(doc.filing_items) <= ROUTINE_ITEMS and not llm["analyze_routine_filings"]:
            art.analysis_status = "skipped"
            art.analysis_error = "routine filing items only"
            report["skipped"] += 1
            continue
        if analyst is None:
            art.analysis_status = "not_configured"
            art.analysis_error = "ANTHROPIC_API_KEY not set"
            report["not_configured"] += 1
            continue
        failed = blocked = False
        for pair in pairs:
            if db.scalar(select(ExtractedEvent.id).where(ExtractedEvent.article_id == art.id, ExtractedEvent.pair_id == pair.id,
                                                       ExtractedEvent.prompt_version == PROMPT_VERSION)):
                continue
            if calls >= int(llm["max_documents_per_run"]):
                blocked = True
                break
            pinfo = {"ticker_a": pair.company_a.ticker, "name_a": pair.company_a.name, "ticker_b": pair.company_b.ticker,
                     "name_b": pair.company_b.name, "industry": pair.industry}
            ddict = {"title": doc.title, "publisher": doc.publisher, "doc_type": doc.doc_type, "filing_form": doc.filing_form,
                     "published_at": doc.published_at.isoformat(), "url": doc.url, "content_text": doc.content_text}
            est = estimate_cost(analyst.model, build_user_message(pinfo, ddict, []), llm["pricing"], analyst.max_output_tokens)
            spent = usage.ai_spend_month_to_date(db, now)
            if spent + est > float(budget["monthly_ai_limit_usd"]):
                art.analysis_status = "budget_blocked"
                art.analysis_error = f"monthly AI budget ${budget['monthly_ai_limit_usd']:.2f} would be exceeded (spent ${spent:.2f})"
                report["budget_blocked"] += 1
                blocked = True
                break
            calls += 1
            try:
                res = analyst.analyze(pinfo, ddict, _prior_events(db, pair.id, now))
            except Exception as e:  # SDK/API errors, refusals, schema-validation errors
                db.add(ProviderUsage(provider="anthropic", operation="analyze_document", success=False, model=analyst.model,
                                     error=f"{type(e).__name__}: {str(e)[:500]}"))
                art.analysis_status = "failed"
                art.analysis_error = f"{type(e).__name__}: {str(e)[:500]}"
                report["failed"] += 1
                failed = True
                break
            db.add(ProviderUsage(provider="anthropic", operation="analyze_document", success=True, model=res.model,
                                 input_tokens=res.input_tokens, output_tokens=res.output_tokens, cache_read_tokens=res.cache_read_tokens,
                                 cache_write_tokens=res.cache_write_tokens, cost_usd=res.cost_usd, latency_ms=res.latency_ms))
            ev = persist_event(db, art, doc, pair, res.interpretation, res.model, res.raw)
            report["events"] += 1
            if ev.is_material:
                material_pairs.add(pair.id)
        if not failed and not blocked:
            art.analysis_status = "analyzed"
            art.analysis_error = None
            report["analyzed"] += 1
        db.commit()
    report["material_pairs"] = sorted(material_pairs)
    return report


def persist_event(db: Session, art: NewsArticle, doc: SourceDocument, pair: Pair, interp, model: str, raw: dict, is_demo: bool = False,
                  analyzed_at: datetime | None = None) -> ExtractedEvent:
    ta, tb = pair.company_a.ticker, pair.company_b.ticker
    source_text = "\n".join(f"{d.title}\n{d.content_text or ''}" for d in art.documents)
    evidence, ratio = verify_evidence([q.model_dump() for q in interp.evidence], source_text)
    related = None
    if interp.related_prior_event_id:
        related = db.get(ExtractedEvent, interp.related_prior_event_id)
        if related is not None and related.pair_id != pair.id:
            related = None
    thread = related.thread_key if related else f"p{pair.id}-a{art.id}"
    sides = {"a": ta, "b": tb}
    ev = ExtractedEvent(
        article_id=art.id, pair_id=pair.id, document_id=doc.id, is_material=bool(interp.is_relevant and interp.is_material and interp.evidence),
        event_type=interp.event_type, affected_tickers=[sides[s] for s in interp.affected_sides], scope=interp.scope,
        relative_direction=direction_to_ticker_form(interp.relative_direction, ta, tb), severity=interp.severity,
        asymmetry=interp.asymmetry, expected_persistence_days=interp.expected_persistence_days,
        extraction_confidence=interp.extraction_confidence, source_reliability=max(d.source_reliability for d in art.documents),
        is_new_information=interp.is_new_information, related_event_id=related.id if related else None, thread_key=thread,
        summary=interp.summary, company_exposure={sides[x.side]: x.model_dump() for x in interp.company_exposure},
        evidence=evidence, evidence_verified_ratio=ratio, uncertainties=interp.uncertainties
        + ([f"shared industry effects: {interp.shared_industry_effects}"] if interp.shared_industry_effects else []),
        source_urls=[d.url for d in art.documents], published_at=art.first_published_at, model_name=model,
        prompt_version=PROMPT_VERSION, schema_version=SCHEMA_VERSION, raw_response=raw, is_demo=is_demo,
    )
    if analyzed_at:
        ev.analyzed_at = analyzed_at
    db.add(ev)
    db.flush()
    return ev


def load_events(db: Session, pair: Pair, cutoff: datetime, lookback_days: int = 365) -> list[EventEvidence]:
    rows = db.execute(
        select(ExtractedEvent, NewsArticle).join(NewsArticle, NewsArticle.id == ExtractedEvent.article_id)
        .where(ExtractedEvent.pair_id == pair.id, ExtractedEvent.published_at <= cutoff, ExtractedEvent.analyzed_at <= cutoff,
               ExtractedEvent.published_at >= cutoff - timedelta(days=lookback_days))
    ).all()
    out = []
    for e, art in rows:
        official = any(d.doc_type in ("filing", "press_release") for d in art.documents)
        out.append(EventEvidence(
            id=e.id, thread_key=e.thread_key, event_type=e.event_type, scope=e.scope, relative_direction=e.relative_direction,
            affected_tickers=e.affected_tickers, severity=e.severity, asymmetry=e.asymmetry, persistence_days=e.expected_persistence_days,
            source_reliability=e.source_reliability, extraction_confidence=e.extraction_confidence,
            evidence_verified_ratio=e.evidence_verified_ratio, published_at=e.published_at,
            ingested_at=max(e.analyzed_at, min(d.ingested_at for d in art.documents)), is_material=e.is_material,
            distinct_publishers=art.distinct_publishers, official_source=official, summary=e.summary,
            document_ids=[d.id for d in art.documents],
        ))
    return out


def load_coverage(db: Session, pair: Pair, cutoff: datetime, days: int = 7, lookback_days: int = 180) -> NewsCoverage | None:
    start = cutoff - timedelta(days=days)

    def ok_days(cid: int) -> int:
        return int(db.scalar(select(func.count(func.distinct(func.date(NewsFetchLog.fetched_at)))).where(
            NewsFetchLog.company_id == cid, NewsFetchLog.success.is_(True), NewsFetchLog.fetched_at > start, NewsFetchLog.fetched_at <= cutoff)) or 0)

    any_fetch = db.scalar(select(func.count()).select_from(NewsFetchLog).where(
        NewsFetchLog.company_id.in_([pair.company_a_id, pair.company_b_id]), NewsFetchLog.fetched_at <= cutoff))
    if not any_fetch:
        return None
    ta, tb = pair.company_a.ticker, pair.company_b.ticker
    arts = db.scalars(select(NewsArticle).where(NewsArticle.first_published_at >= cutoff - timedelta(days=lookback_days),
                                                NewsArticle.first_ingested_at <= cutoff)).all()
    rel = [a for a in arts if {ta, tb} & set(a.tickers)]
    analyzed = sum(1 for a in rel if a.analysis_status in ("analyzed", "skipped"))
    last = db.scalar(select(func.max(NewsFetchLog.fetched_at)).where(NewsFetchLog.company_id.in_([pair.company_a_id, pair.company_b_id]),
                                                                     NewsFetchLog.success.is_(True), NewsFetchLog.fetched_at <= cutoff))
    return NewsCoverage(ok_days(pair.company_a_id), ok_days(pair.company_b_id), days, len(rel), analyzed, last)


def last_news_update(db: Session) -> datetime | None:
    return db.scalar(select(func.max(NewsFetchLog.fetched_at)).where(NewsFetchLog.success.is_(True)))


__all__ = ["ingest_news", "analyze_pending", "load_events", "load_coverage", "store_document", "persist_event"]
