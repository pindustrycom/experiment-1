from __future__ import annotations

from datetime import UTC, datetime, timedelta

from fastapi import APIRouter, Depends
from sqlalchemy import func, select, text
from sqlalchemy.orm import Session

from app.api.deps import current_user, require_admin
from app.api.serializers import latest_scores
from app.core.config import get_settings
from app.core.db import get_db
from app.models import Alert, ExtractedEvent, IndexScore, JobRun, NewsArticle, Pair, ProviderUsage, User
from app.services import demo_service, news_service, prices_service, settings_service, usage

router = APIRouter(tags=["system"])
VERSION = "1.0.0"


@router.get("/health")
def health(db: Session = Depends(get_db)):
    """Liveness/readiness for Docker and the reverse proxy (public, no data)."""
    db.execute(text("select 1"))
    return {"status": "ok", "version": VERSION, "mode": get_settings().app_mode}


def provider_config() -> dict:
    s = get_settings()
    return {
        "anthropic": bool(s.secret("anthropic_api_key")), "tiingo": bool(s.secret("tiingo_api_key")), "stooq": s.enable_stooq,
        "finnhub": bool(s.secret("finnhub_api_key")), "alphavantage": bool(s.secret("alphavantage_api_key")), "sec": bool(s.sec_user_agent),
    }


def worker_status(db: Session) -> dict:
    last_hb = db.scalar(select(func.max(func.coalesce(JobRun.heartbeat_at, JobRun.finished_at, JobRun.started_at))))
    running = db.scalars(select(JobRun).where(JobRun.status == "running")).all()
    queued = db.scalar(select(func.count()).select_from(JobRun).where(JobRun.status == "queued"))
    paused = settings_service.get_section(db, "bot")["paused"]
    alive = last_hb is not None and datetime.now(UTC) - last_hb < timedelta(hours=2)
    state = "paused" if paused else ("running" if running else ("idle" if alive else "unknown"))
    return {"state": state, "paused": paused, "last_activity": last_hb, "running_jobs": [r.job_name for r in running], "queued": queued}


@router.get("/system/status")
def system_status(db: Session = Depends(get_db), _: User = Depends(current_user)):
    s = get_settings()
    now = datetime.now(UTC)
    pairs = db.scalars(select(Pair).where(Pair.status == "active")).all()
    latest = latest_scores(db, [p.id for p in pairs])
    v2 = [latest[(p.id, "si_v2")].value for p in pairs if (p.id, "si_v2") in latest and latest[(p.id, "si_v2")].value is not None]
    v1 = [latest[(p.id, "si_v1")].value for p in pairs if (p.id, "si_v1") in latest and latest[(p.id, "si_v1")].value is not None]
    high = sum(1 for p in pairs if (p.id, "si_v2") in latest and (latest[(p.id, "si_v2")].value or 0) >= p.alert_threshold)
    since = now - timedelta(days=7)
    sig_events = db.scalar(select(func.count()).select_from(ExtractedEvent).where(ExtractedEvent.is_material.is_(True), ExtractedEvent.analyzed_at >= since,
                                                                                 ExtractedEvent.severity >= 0.4))
    budget = settings_service.get_section(db, "budget")
    spent = usage.ai_spend_month_to_date(db, now)
    ai = db.execute(select(func.coalesce(func.sum(ProviderUsage.input_tokens), 0), func.coalesce(func.sum(ProviderUsage.output_tokens), 0), func.count())
                    .where(ProviderUsage.provider == "anthropic", ProviderUsage.ts >= usage.month_start(now))).one()
    last_news, last_price = news_service.last_news_update(db), prices_service.last_price_update(db)
    llm = settings_service.get_section(db, "llm")
    return {
        "mode": s.app_mode, "version": VERSION, "now": now,
        "pairs": {"monitored": len(pairs), "high_risk": high, "avg_si_v2": round(sum(v2) / len(v2), 2) if v2 else None,
                  "avg_si_v1": round(sum(v1) / len(v1), 2) if v1 else None, "scored_si_v2": len(v2)},
        "significant_events_7d": sig_events,
        "bot": worker_status(db),
        "last_news_update": last_news, "last_price_update": last_price,
        "news_age_hours": None if last_news is None else round((now - last_news).total_seconds() / 3600, 1),
        "price_age_hours": None if last_price is None else round((now - last_price).total_seconds() / 3600, 1),
        "ai": {"model": llm["model"] or s.claude_model, "effort": llm["effort"] or s.claude_effort, "month_cost_usd": round(spent, 4),
               "monthly_limit_usd": budget["monthly_ai_limit_usd"], "input_tokens": int(ai[0]), "output_tokens": int(ai[1]), "calls": int(ai[2]),
               "configured": bool(s.secret("anthropic_api_key"))},
        "providers": provider_config(),
        "demo_data_present": (not s.is_demo) and demo_service.demo_data_present(db),
        "pending_analysis": db.scalar(select(func.count()).select_from(NewsArticle).where(NewsArticle.analysis_status.in_(("pending", "not_configured", "budget_blocked", "failed")))),
        "open_alerts": db.scalar(select(func.count()).select_from(Alert).where(Alert.acknowledged_at.is_(None))),
    }


@router.get("/system/top-movers")
def top_movers(db: Session = Depends(get_db), _: User = Depends(current_user), limit: int = 5):
    from app.api.serializers import pair_out
    pairs = pair_out(db, db.scalars(select(Pair).where(Pair.status == "active")).all())
    movers = [p for p in pairs if p.si_v2 and p.si_v2.change_7d is not None]
    movers.sort(key=lambda p: -(p.si_v2.change_7d or 0))
    return [{"slug": p.slug, "si_v2": p.si_v2.value, "change_7d": p.si_v2.change_7d, "direction": p.direction} for p in movers[:limit]]


@router.get("/system/score-distribution")
def score_distribution(db: Session = Depends(get_db), _: User = Depends(current_user), days: int = 365):
    """Histogram of stored SI-v1 and SI-v2 values (all pairs) plus monthly averages."""
    since = datetime.now(UTC) - timedelta(days=days)
    rows = db.execute(select(IndexScore.index_name, IndexScore.value, IndexScore.as_of).where(IndexScore.value.is_not(None), IndexScore.as_of >= since)).all()
    bins = [{"bin": f"{b}-{b + 10}", "si_v1": 0, "si_v2": 0} for b in range(0, 100, 10)]
    monthly: dict[str, dict] = {}
    for name, v, ts in rows:
        bins[min(int(v // 10), 9)][name] += 1
        m = monthly.setdefault(ts.strftime("%Y-%m"), {"month": ts.strftime("%Y-%m"), "si_v1": [], "si_v2": []})
        m[name].append(v)
    series = [{"month": k, "si_v1": round(sum(v["si_v1"]) / len(v["si_v1"]), 2) if v["si_v1"] else None,
               "si_v2": round(sum(v["si_v2"]) / len(v["si_v2"]), 2) if v["si_v2"] else None} for k, v in sorted(monthly.items())]
    return {"histogram": bins, "monthly_mean": series, "n": len(rows)}


@router.get("/usage")
def usage_report(db: Session = Depends(get_db), _: User = Depends(current_user), days: int = 30):
    since = datetime.now(UTC) - timedelta(days=days)
    daily = db.execute(select(func.date(ProviderUsage.ts), func.coalesce(func.sum(ProviderUsage.cost_usd), 0.0), func.coalesce(func.sum(ProviderUsage.input_tokens), 0),
                              func.coalesce(func.sum(ProviderUsage.output_tokens), 0), func.count())
                       .where(ProviderUsage.provider == "anthropic", ProviderUsage.ts >= since).group_by(func.date(ProviderUsage.ts)).order_by(func.date(ProviderUsage.ts))).all()
    errors = db.scalars(select(ProviderUsage).where(ProviderUsage.success.is_(False), ProviderUsage.ts >= since).order_by(ProviderUsage.ts.desc()).limit(50)).all()
    return {
        "providers": usage.usage_summary(db, since),
        "ai_daily": [{"date": str(d), "cost_usd": round(float(c), 4), "input_tokens": int(i), "output_tokens": int(o), "calls": n} for d, c, i, o, n in daily],
        "recent_errors": [{"ts": e.ts, "provider": e.provider, "operation": e.operation, "status_code": e.status_code, "error": e.error} for e in errors],
    }


@router.get("/alerts")
def list_alerts(db: Session = Depends(get_db), _: User = Depends(current_user), open_only: bool = True, limit: int = 100):
    q = select(Alert, Pair.slug).outerjoin(Pair, Pair.id == Alert.pair_id)
    if open_only:
        q = q.where(Alert.acknowledged_at.is_(None))
    return [{"id": a.id, "pair": slug, "kind": a.kind, "severity": a.severity, "message": a.message, "created_at": a.created_at,
             "acknowledged_at": a.acknowledged_at} for a, slug in db.execute(q.order_by(Alert.created_at.desc()).limit(limit)).all()]


@router.post("/alerts/{alert_id}/ack")
def ack_alert(alert_id: int, db: Session = Depends(get_db), _: User = Depends(require_admin)):
    a = db.get(Alert, alert_id)
    if a is None:
        from fastapi import HTTPException
        raise HTTPException(404, "alert not found")
    a.acknowledged_at = datetime.now(UTC)
    db.commit()
    return {"ok": True}
