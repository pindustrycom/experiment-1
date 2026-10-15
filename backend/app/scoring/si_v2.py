"""SI-v2 = 100 * (wN*N + wF*F + wM*M + wP*P + wS*S), plus directional bias and evidence confidence.

Missing components are excluded and the score is re-normalised over the available weight only
when (a) available weight >= min_coverage and (b) every required component is present; the
result is then labelled 'provisional'. Otherwise the score is 'unavailable' (null), never zero.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from datetime import UTC

from app.research.stats import clip01
from app.scoring.components import (
    eligible_events,
    score_fundamentals,
    score_market,
    score_news,
    score_persistence,
    score_structure,
    thread_maxima,
)
from app.scoring.params import si_v2_definition
from app.scoring.types import ComponentResult, PairInputs

ORDER = ["N", "F", "M", "P", "S"]
NAMES = {
    "N": "Asymmetric news risk",
    "F": "Fundamental divergence",
    "M": "Market relationship deterioration",
    "P": "Event persistence",
    "S": "Structural exposure differences",
}


@dataclass
class SIv2Result:
    value: float | None
    status: str
    coverage: float
    components: dict[str, ComponentResult]
    weights: dict[str, float]
    direction: str
    direction_detail: dict
    confidence: float | None
    confidence_breakdown: dict
    explanation: str
    supporting_event_ids: list[int] = field(default_factory=list)


def combine(components: dict[str, ComponentResult], weights: dict[str, float], min_coverage: float, required: list[str]) -> tuple[float | None, str, float]:
    avail = {k: c for k, c in components.items() if c.available and c.value is not None}
    coverage = sum(weights[k] for k in avail)
    if not avail:
        return None, "unavailable", 0.0
    if any(r not in avail for r in required) or coverage + 1e-9 < min_coverage:
        return None, "unavailable", round(coverage, 4)
    value = 100 * sum(weights[k] * avail[k].value for k in avail) / coverage  # type: ignore[operator]
    status = "complete" if len(avail) == len(weights) else "provisional"
    return round(value, 4), status, round(coverage, 4)


def directional_bias(components: dict[str, ComponentResult], p: dict, ticker_a: str, ticker_b: str, kappa: float) -> tuple[str, dict]:
    n, f = components.get("N"), components.get("F")
    news = None
    if n and n.available and n.signed is not None:
        # signed mass > 0 means developments favour A
        news = math.tanh(n.signed / kappa)
    fund = None
    if f and f.available and f.signed is not None:
        fund = f.signed
    detail = {"news_signal": news, "fundamental_signal": fund, "basis": "documented asymmetric developments and reported fundamentals; recent price moves are not used"}
    if news is None and fund is None:
        return "uncertain", {**detail, "reason": "no news or fundamental evidence available"}
    if news is not None and fund is not None and abs(news) >= p["conflict_level"] and abs(fund) >= p["conflict_level"] and news * fund < 0:
        return "uncertain", {**detail, "reason": "news and fundamentals point in opposite directions"}
    parts, wsum = 0.0, 0.0
    if news is not None:
        parts += p["news_weight"] * news
        wsum += p["news_weight"]
    if fund is not None:
        parts += p["fundamental_weight"] * fund
        wsum += p["fundamental_weight"]
    combined = parts / wsum
    detail["combined"] = combined
    if abs(combined) < p["neutral_band"]:
        return "neutral", detail
    return ("a_advantaged" if combined > 0 else "b_advantaged"), detail


def evidence_confidence(inp: PairInputs, components: dict[str, ComponentResult], coverage: float, pn: dict) -> tuple[float | None, dict]:
    parts: dict[str, float | None] = {}
    weights = {"source_credibility": 0.2, "news_coverage": 0.15, "source_agreement": 0.15, "data_completeness": 0.2, "ai_certainty": 0.15, "freshness": 0.15}
    counted = []
    if inp.events is not None:
        counted = [(ev, w) for ev, w, _ in thread_maxima(eligible_events(inp.events, inp.cutoff, pn["lookback_days"]), inp.cutoff, pn) if w > 0]
    tw = sum(w for _, w in counted)
    if counted and tw > 0:
        parts["source_credibility"] = sum(ev.source_reliability * w for ev, w in counted) / tw
        parts["source_agreement"] = sum((1.0 if (ev.distinct_publishers >= 2 or ev.official_source) else 0.0) * w for ev, w in counted) / tw
        parts["ai_certainty"] = sum(ev.extraction_confidence * ev.evidence_verified_ratio * w for ev, w in counted) / tw
    else:
        parts["source_credibility"] = parts["source_agreement"] = parts["ai_certainty"] = None
    cov = inp.coverage
    parts["news_coverage"] = None if cov is None else clip01(min(cov.fetch_days_ok_a, cov.fetch_days_ok_b) / max(cov.days, 1))
    parts["data_completeness"] = coverage
    fresh = []
    last = inp.market.last_date
    if last is not None:
        age_days = max(0, (inp.cutoff.date() - last).days)
        fresh.append(math.exp(-max(0, age_days - 3) / 3))  # weekends/holidays tolerated
    if cov is not None and cov.last_fetch_at is not None:
        hrs = max(0.0, (inp.cutoff - cov.last_fetch_at).total_seconds() / 3600)
        fresh.append(math.exp(-max(0.0, hrs - 2) / 24))
    parts["freshness"] = sum(fresh) / len(fresh) if fresh else None
    avail = {k: v for k, v in parts.items() if v is not None}
    if not avail:
        return None, {"parts": parts}
    conf = sum(weights[k] * v for k, v in avail.items()) / sum(weights[k] for k in avail)
    return round(clip01(conf), 4), {"parts": parts, "weights": weights, "note": "Evidence quality, not a probability of divergence or profit."}


def compute_si_v2(inp: PairInputs, weights: dict | None = None, params: dict | None = None) -> SIv2Result:
    d = si_v2_definition(weights, params)
    w, p = d["weights"], d["params"]
    cutoff = inp.cutoff if inp.cutoff.tzinfo else inp.cutoff.replace(tzinfo=UTC)
    cdate = cutoff.date()
    comps = {
        "N": score_news(inp.events, inp.coverage, cutoff, inp.ticker_a, inp.ticker_b, p["N"]),
        "F": score_fundamentals(inp.profile_a, inp.profile_b, cdate, p["F"]),
        "M": score_market(inp.market, p["M"]),
        "P": score_persistence(inp.events, inp.coverage, cutoff, p["N"], p["P"]),
        "S": score_structure(inp.market, inp.profile_a, inp.profile_b, cdate, p["F"], p["S"]),
    }
    value, status, coverage = combine(comps, w, p["min_coverage"], p["required_components"])
    direction, ddetail = directional_bias(comps, p["direction"], inp.ticker_a, inp.ticker_b, p["N"]["kappa"])
    conf, cbreak = evidence_confidence(inp, comps, coverage, p["N"])
    expl = explain(value, status, coverage, comps, w, direction, inp.ticker_a, inp.ticker_b)
    support = sorted({int(s) for k in ("N", "P") for s in comps[k].sources})
    return SIv2Result(value, status, coverage, comps, w, direction, ddetail, conf, cbreak, expl, support)


def direction_label(direction: str, ta: str, tb: str) -> str:
    return {"a_advantaged": f"{ta} relatively advantaged", "b_advantaged": f"{tb} relatively advantaged", "neutral": "Neutral", "uncertain": "Uncertain"}[direction]


def explain(value, status, coverage, comps, w, direction, ta, tb) -> str:
    if value is None:
        missing = [NAMES[k] for k in ORDER if not comps[k].available]
        return f"SI-v2 unavailable (coverage {coverage:.0%}); missing: {', '.join(missing)}."
    contrib = sorted(((w[k] * comps[k].value, k) for k in ORDER if comps[k].available), reverse=True)
    lead = ", ".join(f"{NAMES[k]} {comps[k].value:.2f}" for _, k in contrib[:2])
    s = f"SI-v2 {value:.1f} ({status}, {coverage:.0%} of weight available). Main drivers: {lead}. "
    s += f"Bias: {direction_label(direction, ta, tb)}."
    if comps["N"].available:
        s += " " + comps["N"].explanation
    gaps = [NAMES[k] for k in ORDER if not comps[k].available]
    if gaps:
        s += f" Not scored: {', '.join(gaps)}."
    return s
