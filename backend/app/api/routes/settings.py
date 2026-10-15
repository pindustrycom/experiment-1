from __future__ import annotations

import time

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy import select
from sqlalchemy.orm import Session

from app.api.deps import current_user, require_admin
from app.api.routes.system import provider_config
from app.core.config import get_settings
from app.core.db import get_db
from app.models import ModelVersion, User
from app.schemas.api import SettingsUpdate
from app.services import settings_service
from app.services.reference import ensure_model_versions

router = APIRouter(prefix="/settings", tags=["settings"])


@router.get("")
def get_settings_all(db: Session = Depends(get_db), _: User = Depends(current_user)):
    s = get_settings()
    return {
        "sections": settings_service.get_all(db),
        "environment": {"mode": s.app_mode, "claude_model_default": s.claude_model, "claude_effort_default": s.claude_effort,
                        "providers_configured": provider_config(), "scheduler_timezone": s.scheduler_timezone,
                        "note": "API keys are configured as server environment variables and are never sent to the browser."},
    }


@router.put("")
def put_settings(body: SettingsUpdate, db: Session = Depends(get_db), user: User = Depends(require_admin)):
    try:
        value = settings_service.put_section(db, body.section, body.value, user.id)
        if body.section == "scoring":
            from app.services.scoring_service import current_params
            w, p = current_params(db)
            ensure_model_versions(db, w, p)  # new weights => new immutable SI-v2 model version
    except ValueError as e:
        db.rollback()
        raise HTTPException(422, str(e)) from e
    db.commit()
    return {"section": body.section, "value": value}


@router.get("/model-versions")
def model_versions(db: Session = Depends(get_db), _: User = Depends(current_user)):
    return [{"id": m.id, "index_name": m.index_name, "version": m.version, "weights": m.weights, "is_active": m.is_active,
             "created_at": m.created_at, "description": m.description, "params": m.params}
            for m in db.scalars(select(ModelVersion).order_by(ModelVersion.created_at.desc())).all()]


@router.post("/health-check")
def health_check(_: User = Depends(require_admin)):
    """Live, minimal calls to each configured provider (results are not stored as data)."""
    from app.data_sources.base import ProviderError
    from app.data_sources.http import request
    s = get_settings()
    out = {}

    def probe(name, fn):
        t0 = time.monotonic()
        try:
            fn()
            out[name] = {"ok": True, "latency_ms": int((time.monotonic() - t0) * 1000)}
        except ProviderError as e:
            out[name] = {"ok": False, "error": str(e)[:300]}
        except Exception as e:  # SDK errors
            out[name] = {"ok": False, "error": f"{type(e).__name__}: {str(e)[:300]}"}

    if s.secret("tiingo_api_key"):
        probe("tiingo", lambda: request("tiingo", "GET", "https://api.tiingo.com/tiingo/daily/spy", headers={"Authorization": f"Token {s.secret('tiingo_api_key')}"}, operation="health", max_attempts=1))
    if s.secret("finnhub_api_key"):
        probe("finnhub", lambda: request("finnhub", "GET", "https://finnhub.io/api/v1/quote", params={"symbol": "SPY"},
                                         headers={"X-Finnhub-Token": s.secret("finnhub_api_key")}, operation="health", max_attempts=1))
    if s.sec_user_agent:
        probe("sec", lambda: request("sec", "GET", "https://data.sec.gov/submissions/CIK0000021344.json", headers={"User-Agent": s.sec_user_agent}, operation="health", max_attempts=1))
    if s.enable_stooq:
        def stooq():
            r = request("stooq", "GET", "https://stooq.com/q/d/l/", params={"s": "spy.us", "i": "d"}, operation="health", max_attempts=1)
            if not r.text.startswith("Date,"):
                raise ProviderError("stooq", f"unexpected response {r.text[:80]!r}")
        probe("stooq", stooq)
    if s.secret("alphavantage_api_key"):
        out["alphavantage"] = {"ok": None, "note": "not probed automatically to preserve the small free-tier quota"}
    if s.secret("anthropic_api_key"):
        import anthropic

        def claude():
            anthropic.Anthropic(api_key=s.secret("anthropic_api_key"), timeout=20, max_retries=0).models.retrieve(s.claude_model)
        probe("anthropic", claude)
    if not out:
        out["note"] = {"ok": None, "error": "no providers configured"}
    return out
