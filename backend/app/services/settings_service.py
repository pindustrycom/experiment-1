"""Runtime research settings stored in app_settings (JSON per section). Secrets never live here."""
from __future__ import annotations

import math
from copy import deepcopy

from sqlalchemy import select
from sqlalchemy.orm import Session

from app.agents.pricing import DEFAULT_PRICING
from app.models import AppSetting
from app.scoring.params import SI_V2_DEFAULT_WEIGHTS, normalize_weights

DEFAULTS: dict = {
    "bot": {"paused": False},
    "scoring": {"si_v2_weights": {k: round(v * 100, 4) for k, v in SI_V2_DEFAULT_WEIGHTS.items()}, "min_coverage": 0.60},
    "llm": {
        "model": None,  # None -> CLAUDE_MODEL env var
        "effort": None,  # None -> CLAUDE_EFFORT env var
        "max_documents_per_run": 40,
        "refusal_fallback": True,
        "analyze_routine_filings": False,
        "pricing": DEFAULT_PRICING,
    },
    "budget": {"monthly_ai_limit_usd": 50.0, "warn_at_fraction": 0.8},
    "news": {
        "providers": {"sec": True, "finnhub": True, "alphavantage": True, "rss": True},
        "lookback_hours": 72,
        "rss_templates": [],
        "company_feeds": {},
    },
    "prices": {"providers": ["tiingo", "stooq"], "history_start": "2009-01-01", "benchmark": "SPY"},
    "alerts": {"default_threshold": 70.0, "default_change_threshold": 10.0, "confidence_floor": 0.3},
    "predictions": {
        "horizons": [30, 90, 180, 252, 504],
        "base_threshold": 0.20,
        "base_horizon": 504,
        "threshold_scaling": "sqrt",
        "threshold_overrides": {},
        "sensitivity_multipliers": [0.5, 0.75, 1.0, 1.25, 1.5],
    },
}

SECTIONS = tuple(DEFAULTS)


def _merge(base: dict, override: dict) -> dict:
    out = deepcopy(base)
    for k, v in (override or {}).items():
        if isinstance(v, dict) and isinstance(out.get(k), dict) and k not in ("pricing", "company_feeds", "threshold_overrides"):
            out[k] = _merge(out[k], v)
        else:
            out[k] = v
    return out


def get_all(db: Session) -> dict:
    stored = {r.key: r.value for r in db.scalars(select(AppSetting)).all()}
    return {s: _merge(DEFAULTS[s], stored.get(s) or {}) for s in SECTIONS}


def get_section(db: Session, section: str) -> dict:
    row = db.get(AppSetting, section)
    return _merge(DEFAULTS[section], (row.value if row else None) or {})


def validate(section: str, value: dict) -> dict:
    if section not in DEFAULTS:
        raise ValueError(f"unknown settings section '{section}'")
    merged = _merge(DEFAULTS[section], value)
    if section == "scoring":
        normalize_weights(merged["si_v2_weights"])
        if not 0 < float(merged["min_coverage"]) <= 1:
            raise ValueError("min_coverage must be in (0, 1]")
    elif section == "budget":
        if float(merged["monthly_ai_limit_usd"]) < 0:
            raise ValueError("monthly_ai_limit_usd must be >= 0")
    elif section == "llm":
        if merged["effort"] not in (None, "low", "medium", "high", "xhigh", "max"):
            raise ValueError("effort must be low|medium|high|xhigh|max")
        if not 1 <= int(merged["max_documents_per_run"]) <= 1000:
            raise ValueError("max_documents_per_run must be 1..1000")
        if merged["model"] is not None and not str(merged["model"]).startswith("claude-"):
            raise ValueError("model must be a Claude model id")
    elif section == "predictions":
        hs = merged["horizons"]
        if not hs or any(int(h) <= 0 or int(h) > 1260 for h in hs):
            raise ValueError("horizons must be positive trading-day counts <= 1260")
        if float(merged["base_threshold"]) <= 0:
            raise ValueError("base_threshold must be > 0")
        if merged["threshold_scaling"] not in ("sqrt", "constant"):
            raise ValueError("threshold_scaling must be sqrt|constant")
    elif section == "news":
        for t in merged["rss_templates"]:
            if not str(t).startswith(("https://", "http://")):
                raise ValueError("feed URLs must be http(s)")
        for urls in merged["company_feeds"].values():
            for u in urls:
                if not str(u).startswith(("https://", "http://")):
                    raise ValueError("feed URLs must be http(s)")
    elif section == "alerts":
        if not 0 <= float(merged["default_threshold"]) <= 100:
            raise ValueError("default_threshold must be 0..100")
    return merged


def put_section(db: Session, section: str, value: dict, user_id: int | None) -> dict:
    merged = validate(section, value)
    row = db.get(AppSetting, section)
    if row is None:
        db.add(AppSetting(key=section, value=merged, updated_by=user_id))
    else:
        row.value = merged
        row.updated_by = user_id
    db.flush()
    return merged


def divergence_threshold(horizon: int, pred: dict) -> float:
    """Documented target definition: the historical 504-session threshold (0.20) is kept for the
    two-year horizon; shorter horizons scale by sqrt(H/504) (random-walk dispersion of the
    log-return gap) unless an explicit override is configured."""
    ov = pred.get("threshold_overrides", {}).get(str(horizon))
    if ov is not None:
        return float(ov)
    base, bh = float(pred["base_threshold"]), int(pred["base_horizon"])
    if pred["threshold_scaling"] == "constant":
        return base
    return round(base * math.sqrt(horizon / bh), 4)


def target_definition(horizon: int, pred: dict) -> tuple[str, dict]:
    thr = divergence_threshold(horizon, pred)
    version = f"div1-H{horizon}-T{thr:.4f}"
    return version, {
        "measure": "D(t,H) = |ln(PA(t+H)/PA(t)) - ln(PB(t+H)/PB(t))| on total-return adjusted closes",
        "horizon_trading_days": horizon,
        "threshold": thr,
        "event": "D >= threshold",
        "t": "close of the last completed NYSE session at the prediction cutoff",
        "t_plus_H": "close of the H-th subsequent NYSE session",
        "threshold_rule": "override" if str(horizon) in pred.get("threshold_overrides", {}) else pred["threshold_scaling"],
    }
