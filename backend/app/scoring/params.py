"""Versioned index definitions. A definition's content hash identifies it; any change to a weight
or parameter yields a new model version, so historical scores are never re-interpreted."""
from __future__ import annotations

import hashlib
import json
from copy import deepcopy

SI_V1_VERSION = "1.0"
SI_V1_DEFINITION = {
    "index_name": "si_v1",
    "version": SI_V1_VERSION,
    "weights": {"C": 25, "D": 15, "R": 25},
    "params": {
        "returns": "daily log total returns (adjusted close)",
        "long_window": 1260,
        "short_window": 252,
        "max_missing_closes": 10,
        "min_returns": 1250,
        "C": "clip((c_anchor - rho_short) / c_anchor, 0, 1)",
        "c_anchor": 0.90,
        "D": "clip((rho_long - rho_short) / d_scale, 0, 1)",
        "d_scale": 0.30,
        "R": "clip(|100*(PA_t/PA_t-252 - 1) - 100*(PB_t/PB_t-252 - 1)| / r_scale_pp, 0, 1)",
        "r_window": 252,
        "r_scale_pp": 50.0,
        "formula": "(25C + 15D + 25R) * 100 / 65",
    },
    "description": (
        "Market-only benchmark, ported from the original run_experiment.py (index_components and "
        "calculate_rows): rho_short is the 1-year (252-session) and rho_long the 5-year (1260-session) "
        "correlation of daily log returns; R is the 1-year simple total-return gap in percentage points."
    ),
}

SI_V2_DEFAULT_WEIGHTS = {"N": 0.35, "F": 0.25, "M": 0.20, "P": 0.10, "S": 0.10}

SI_V2_PARAMS = {
    "schema": "2.0",
    "min_coverage": 0.60,
    "required_components": ["M"],
    "N": {
        "lookback_days": 180,
        "kappa": 0.5,
        "min_evidence_verified_ratio": 0.5,
        "persistence_floor_days": 7,
        "persistence_cap_days": 365,
        "shared_scope_asymmetry_cap": 0.3,
        "coverage_window_days": 7,
        "min_fetch_days": 3,
        "min_analyzed_ratio": 0.8,
    },
    "P": {
        "kappa": 0.5,
        "full_persistence_days": 365,
        "structural_event_types": [
            "acquisition_merger", "executive_change", "restructuring_layoffs", "regulatory_change",
            "market_share_change", "lawsuit_investigation",
        ],
        "structural_floor": 0.5,
    },
    "M": {
        "long_window": 504,
        "short_window": 63,
        "corr_scale": 0.5,
        "formation_window": 252,
        "zscore_window": 63,
        "z_scale": 3.0,
        "coint_window": 252,
        "p_low": 0.05,
        "p_high": 0.50,
        "min_subfeatures": 2,
    },
    "F": {
        "max_staleness_days": 200,
        "revenue_growth_scale": 0.20,
        "op_margin_change_scale": 0.05,
        "eps_growth_scale": 0.50,
        "min_subfeatures": 2,
    },
    "S": {
        "beta_window": 504,
        "beta_scale": 0.5,
        "vol_window": 252,
        "size_scale_log": 2.302585,
        "min_subfeatures": 2,
    },
    "direction": {"news_weight": 0.6, "fundamental_weight": 0.4, "neutral_band": 0.15, "conflict_level": 0.2},
}


def definition_hash(weights: dict, params: dict) -> str:
    blob = json.dumps({"w": weights, "p": params}, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(blob.encode()).hexdigest()


def si_v2_definition(weights: dict | None = None, params: dict | None = None) -> dict:
    w = normalize_weights(weights or SI_V2_DEFAULT_WEIGHTS)
    p = deepcopy(params or SI_V2_PARAMS)
    h = definition_hash(w, p)
    return {
        "index_name": "si_v2",
        "version": f"2.0+{h[:10]}",
        "weights": w,
        "params": p,
        "content_hash": h,
        "description": "SI-v2 = 100 * (wN*N + wF*F + wM*M + wP*P + wS*S); weights are research hypotheses.",
    }


def si_v1_definition() -> dict:
    d = deepcopy(SI_V1_DEFINITION)
    d["content_hash"] = definition_hash(d["weights"], d["params"])
    return d


def normalize_weights(weights: dict) -> dict:
    keys = ["N", "F", "M", "P", "S"]
    missing = [k for k in keys if k not in weights]
    if missing:
        raise ValueError(f"missing weights: {missing}")
    vals = {k: float(weights[k]) for k in keys}
    if any(v < 0 for v in vals.values()):
        raise ValueError("weights must be non-negative")
    total = sum(vals.values())
    if total <= 0:
        raise ValueError("weights must sum to a positive number")
    # Accept either fractions (sum 1) or points (sum 100); store fractions rounded for stable hashes.
    return {k: round(v / total, 6) for k, v in vals.items()}
