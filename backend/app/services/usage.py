from __future__ import annotations

from datetime import UTC, datetime

from sqlalchemy import func, select
from sqlalchemy.orm import Session

from app.core.db import session_scope
from app.data_sources import http
from app.models import ProviderUsage


def db_usage_sink(**kw) -> None:
    with session_scope() as db:
        db.add(ProviderUsage(**kw))


def install() -> None:
    http.set_usage_sink(db_usage_sink)


def month_start(now: datetime | None = None) -> datetime:
    now = now or datetime.now(UTC)
    return datetime(now.year, now.month, 1, tzinfo=UTC)


def ai_spend_month_to_date(db: Session, now: datetime | None = None) -> float:
    return float(db.scalar(select(func.coalesce(func.sum(ProviderUsage.cost_usd), 0.0)).where(
        ProviderUsage.provider == "anthropic", ProviderUsage.ts >= month_start(now))) or 0.0)


def usage_summary(db: Session, since: datetime) -> list[dict]:
    rows = db.execute(
        select(
            ProviderUsage.provider,
            func.count(),
            func.count().filter(ProviderUsage.success.is_(True)),
            func.coalesce(func.sum(ProviderUsage.input_tokens), 0),
            func.coalesce(func.sum(ProviderUsage.output_tokens), 0),
            func.coalesce(func.sum(ProviderUsage.cost_usd), 0.0),
            func.max(ProviderUsage.ts),
            func.max(ProviderUsage.ts).filter(ProviderUsage.success.is_(True)),
            func.avg(ProviderUsage.latency_ms),
        ).where(ProviderUsage.ts >= since).group_by(ProviderUsage.provider)
    ).all()
    return [
        {"provider": r[0], "requests": r[1], "successes": r[2], "input_tokens": int(r[3]), "output_tokens": int(r[4]),
         "cost_usd": round(float(r[5]), 4), "last_call": r[6], "last_success": r[7], "avg_latency_ms": None if r[8] is None else int(r[8])}
        for r in rows
    ]
