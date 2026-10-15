"""Builds point-in-time inputs from the database, runs SI-v1/SI-v2 and persists append-only scores."""
from __future__ import annotations

import hashlib
import json
import logging
import time
from datetime import UTC, date, datetime, timedelta

from sqlalchemy import select
from sqlalchemy.dialects.postgresql import insert
from sqlalchemy.orm import Session

from app.core.config import get_settings
from app.models import Alert, Company, ComponentCalculation, IndexScore, ModelVersion, Pair
from app.research.market_calendar import last_completed_session, trading_days_between
from app.scoring.params import SI_V2_PARAMS
from app.scoring.pit import close_timestamp
from app.scoring.si_v1 import compute_si_v1
from app.scoring.si_v2 import NAMES, compute_si_v2
from app.scoring.types import PairInputs
from app.services import fundamentals_service, news_service, prices_service, settings_service
from app.services.reference import ensure_model_versions

log = logging.getLogger(__name__)


def current_params(db: Session) -> tuple[dict, dict]:
    sc = settings_service.get_section(db, "scoring")
    params = json.loads(json.dumps(SI_V2_PARAMS))
    params["min_coverage"] = float(sc["min_coverage"])
    return sc["si_v2_weights"], params


def build_inputs(db: Session, pair: Pair, cutoff: datetime) -> PairInputs:
    s = get_settings()
    bench = db.scalar(select(Company).where(Company.is_benchmark.is_(True)))
    market = prices_service.pair_market_data(db, pair.company_a, pair.company_b, cutoff, bench, demo=None if s.is_demo else False)
    cov = news_service.load_coverage(db, pair, cutoff)
    events = news_service.load_events(db, pair, cutoff) if cov is not None else None
    return PairInputs(
        ticker_a=pair.company_a.ticker, ticker_b=pair.company_b.ticker, cutoff=cutoff, market=market, events=events, coverage=cov,
        profile_a=fundamentals_service.company_profile(db, pair.company_a, cutoff.date()),
        profile_b=fundamentals_service.company_profile(db, pair.company_b, cutoff.date()),
    )


def inputs_digest(inp: PairInputs) -> str:
    blob = {
        "last_price": str(inp.market.last_date), "n_prices": len(inp.market.prices), "provider": inp.market.provider,
        "events": sorted((e.id, e.ingested_at.isoformat()) for e in (inp.events or [])),
        "coverage": None if inp.coverage is None else [inp.coverage.fetch_days_ok_a, inp.coverage.fetch_days_ok_b, inp.coverage.relevant_clusters, inp.coverage.analyzed_clusters],
        "facts": sorted({f.accession for f in inp.profile_a.facts} | {f.accession for f in inp.profile_b.facts}),
        "sic": [inp.profile_a.sic_code, inp.profile_b.sic_code], "day": inp.cutoff.date().isoformat(),
    }
    return hashlib.sha256(json.dumps(blob, sort_keys=True).encode()).hexdigest()


def _store(db: Session, pair: Pair, mv: ModelVersion, cutoff: datetime, digest: str, trigger: str, is_demo: bool, **fields) -> IndexScore | None:
    comps = fields.pop("components")
    stmt = insert(IndexScore).values(pair_id=pair.id, model_version_id=mv.id, index_name=mv.index_name, as_of=cutoff,
                                     inputs_digest=digest, trigger=trigger, is_demo=is_demo, **fields)
    stmt = stmt.on_conflict_do_nothing(constraint="uq_score_identity").returning(IndexScore.id)
    sid = db.execute(stmt).scalar()
    if sid is None:
        return None
    for code, c, w in comps:
        db.add(ComponentCalculation(score_id=sid, component=code, value=c.value, available=c.available, weight=w,
                                    inputs=_jsonable(c.inputs), explanation=c.explanation, sources=_jsonable(c.sources)))
    db.flush()
    return db.get(IndexScore, sid)


def _jsonable(x):
    return json.loads(json.dumps(x, default=lambda o: o.isoformat() if hasattr(o, "isoformat") else float(o) if hasattr(o, "__float__") else str(o)))


def latest_score(db: Session, pair_id: int, index_name: str, before: datetime | None = None) -> IndexScore | None:
    q = select(IndexScore).where(IndexScore.pair_id == pair_id, IndexScore.index_name == index_name)
    if before is not None:
        q = q.where(IndexScore.as_of <= before)
    return db.scalar(q.order_by(IndexScore.as_of.desc(), IndexScore.id.desc()).limit(1))


def score_pair(db: Session, pair: Pair, cutoff: datetime | None = None, trigger: str = "scheduled") -> dict:
    cutoff = cutoff or datetime.now(UTC)
    weights, params = current_params(db)
    v1mv, v2mv = ensure_model_versions(db, weights, params)
    inp = build_inputs(db, pair, cutoff)
    digest = inputs_digest(inp)
    is_demo = inp.market.provider == "demo"
    r1 = compute_si_v1(inp.market)
    r2 = compute_si_v2(inp, weights, params)
    tr_note = "" if inp.market.is_total_return else " Prices are price-return only (no dividend adjustment available)."
    s1 = _store(db, pair, v1mv, cutoff, digest, trigger, is_demo, value=r1.value,
                status="complete" if r1.value is not None else "unavailable", coverage=1.0 if r1.value is not None else 0.0,
                direction="not_applicable", direction_detail={}, confidence=None, confidence_breakdown={"baselines": _jsonable(r1.baselines)},
                explanation=r1.explanation + tr_note, supporting_event_ids=[],
                components=[(k, c, v1mv.weights[k]) for k, c in r1.components.items()])
    s2 = _store(db, pair, v2mv, cutoff, digest, trigger, is_demo, value=r2.value, status=r2.status, coverage=r2.coverage,
                direction=r2.direction, direction_detail=_jsonable(r2.direction_detail), confidence=r2.confidence,
                confidence_breakdown=_jsonable(r2.confidence_breakdown), explanation=r2.explanation + tr_note,
                supporting_event_ids=r2.supporting_event_ids,
                components=[(k, r2.components[k], r2.weights[k]) for k in NAMES])
    if s2 is not None and trigger != "backfill":  # reconstructed history never raises alerts
        check_alerts(db, pair, s2)
    db.commit()
    return {"pair": pair.slug, "si_v1": r1.value, "si_v2": r2.value, "si_v2_status": r2.status, "stored": s2 is not None or s1 is not None}


def check_alerts(db: Session, pair: Pair, score: IndexScore) -> None:
    if score.value is None:
        return
    prev = db.scalar(select(IndexScore).where(IndexScore.pair_id == pair.id, IndexScore.index_name == "si_v2", IndexScore.value.is_not(None),
                                             IndexScore.as_of <= score.as_of - timedelta(days=7)).order_by(IndexScore.as_of.desc()).limit(1))
    day = score.as_of.date().isoformat()

    def add(kind, sev, msg):
        stmt = insert(Alert).values(pair_id=pair.id, kind=kind, severity=sev, message=msg, payload={"score_id": score.id, "value": score.value},
                                    dedup_key=f"{pair.id}:{kind}:{day}").on_conflict_do_nothing(index_elements=["dedup_key"])
        db.execute(stmt)

    last = db.scalar(select(IndexScore).where(IndexScore.pair_id == pair.id, IndexScore.index_name == "si_v2", IndexScore.value.is_not(None),
                                             IndexScore.id != score.id).order_by(IndexScore.as_of.desc()).limit(1))
    if score.value >= pair.alert_threshold and (last is None or (last.value or 0) < pair.alert_threshold):
        add("threshold_crossed", "critical" if score.value >= pair.alert_threshold + 10 else "warning",
            f"{pair.slug} SI-v2 {score.value:.1f} crossed alert threshold {pair.alert_threshold:.0f}")
    if prev is not None and prev.value is not None and score.value - prev.value >= pair.alert_change_threshold:
        add("rapid_increase", "warning", f"{pair.slug} SI-v2 rose {score.value - prev.value:+.1f} points over 7 days")


def score_all(db: Session, trigger: str = "scheduled", pair_ids: list[int] | None = None, cutoff: datetime | None = None) -> dict:
    q = select(Pair).where(Pair.status == "active")
    if pair_ids:
        q = q.where(Pair.id.in_(pair_ids))
    out = {"scored": [], "errors": []}
    for p in db.scalars(q.order_by(Pair.slug)).all():
        try:
            out["scored"].append(score_pair(db, p, cutoff, trigger))
        except Exception as e:  # one bad pair must not stop the others
            db.rollback()
            log.exception("scoring failed", extra={"pair": p.slug})
            out["errors"].append(f"{p.slug}: {type(e).__name__}: {e}")
    return out


def backfill_history(db: Session, start: date, end: date | None = None, max_seconds: float | None = None) -> dict:
    """Reconstruct weekly score history (last session of each week, scored after its close) from
    stored data, using the same point-in-time inputs as a live run: prices, filings and analysed news
    known at that cutoff. Rows are marked trigger="backfill"; they are research history, never
    predictions, and never raise alerts. Weeks that already have a score for a pair are skipped, so
    the job is incremental and safe to re-run."""
    started = time.monotonic()
    end = end or last_completed_session(datetime.now(UTC))
    weeks: dict[tuple[int, int], date] = {}
    for d in trading_days_between(start, end):
        weeks[d.isocalendar()[:2]] = d  # last session of each ISO week
    pairs = db.scalars(select(Pair).where(Pair.status == "active").order_by(Pair.slug)).all()
    have = {(pid, t.date().isocalendar()[:2]) for pid, t in db.execute(
        select(IndexScore.pair_id, IndexScore.as_of).where(IndexScore.index_name == "si_v1")).all()}
    out = {"weeks": len(weeks), "scored": 0, "skipped": 0, "errors": 0, "first_errors": [], "complete": True}
    # newest weeks first, all pairs together, so a time-limited run fills every chart evenly
    for wk, d in sorted(weeks.items(), reverse=True):
        for p in pairs:
            if (p.id, wk) in have:
                out["skipped"] += 1
                continue
            if max_seconds is not None and time.monotonic() - started > max_seconds:
                out["complete"] = False
                return out
            try:
                score_pair(db, p, close_timestamp(d) + timedelta(hours=2), trigger="backfill")
                out["scored"] += 1
            except Exception as e:  # keep going; one bad week must not stop the history
                db.rollback()
                out["errors"] += 1
                if len(out["first_errors"]) < 5:
                    out["first_errors"].append(f"{p.slug} {d}: {type(e).__name__}: {e}")
    return out

