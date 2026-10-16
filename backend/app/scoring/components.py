"""SI-v2 component scorers. Each returns a value in [0,1] or marks itself unavailable with a
reason; missing data is never silently treated as zero risk."""
from __future__ import annotations

import math
from datetime import date, datetime, timedelta

import numpy as np
import pandas as pd

from app.research.stats import (
    align,
    annualized_vol,
    beta_to_benchmark,
    clip01,
    engle_granger_pvalue,
    log_returns,
    out_of_sample_zscores,
    window_corr,
)
from app.scoring.types import CompanyProfile, ComponentResult, EventEvidence, Fact, MarketData, NewsCoverage

# --------------------------------------------------------------------------- news (N, P)

SHARED_SCOPES = {"both_companies", "industry_wide"}


def event_side_sign(ev: EventEvidence, ticker_a: str, ticker_b: str) -> int:
    """+1 if the event favours A relative to B, -1 if it favours B, 0 if symmetric/unclear."""
    d = (ev.relative_direction or "").upper()
    if d in (f"{ticker_a}_ADVANTAGED", f"{ticker_b}_DISADVANTAGED"):
        return 1
    if d in (f"{ticker_b}_ADVANTAGED", f"{ticker_a}_DISADVANTAGED"):
        return -1
    return 0


def effective_asymmetry(ev: EventEvidence, cap: float) -> float:
    """Deterministic guard on the AI estimate: shared/industry-wide events cannot carry more than
    `cap` asymmetry, and a symmetric or unclear direction with a shared scope carries none beyond it."""
    a = clip01(ev.asymmetry)
    if ev.scope in SHARED_SCOPES:
        a = min(a, cap)
    if ev.scope == "unrelated":
        a = 0.0
    return a


def event_weight(ev: EventEvidence, cutoff: datetime, p: dict) -> tuple[float, dict]:
    if not ev.is_material:
        return 0.0, {"reason": "not material"}
    if ev.evidence_verified_ratio < p["min_evidence_verified_ratio"]:
        return 0.0, {"reason": "quoted evidence not verified in source text"}
    age_days = max(0.0, (cutoff - ev.published_at).total_seconds() / 86400)
    tau = min(max(ev.persistence_days, p["persistence_floor_days"]), p["persistence_cap_days"])
    decay = math.exp(-age_days / tau)
    asym = effective_asymmetry(ev, p["shared_scope_asymmetry_cap"])
    w = clip01(ev.severity) * asym * clip01(ev.source_reliability) * clip01(ev.extraction_confidence) * decay
    return w, {"age_days": round(age_days, 2), "tau_days": tau, "decay": round(decay, 4), "asymmetry_used": asym}


def eligible_events(events: list[EventEvidence], cutoff: datetime, lookback_days: int) -> list[EventEvidence]:
    """Point-in-time filter: an event counts only if it was both published AND ingested before the
    cutoff (late-ingested news cannot leak into earlier scores)."""
    start = cutoff - timedelta(days=lookback_days)
    return [e for e in events if e.published_at <= cutoff and e.ingested_at <= cutoff and e.published_at >= start]


def thread_maxima(events: list[EventEvidence], cutoff: datetime, p: dict) -> list[tuple[EventEvidence, float, dict]]:
    """Repeated coverage of one development (same thread_key) counts once: the strongest item."""
    best: dict[str, tuple[EventEvidence, float, dict]] = {}
    for ev in events:
        w, info = event_weight(ev, cutoff, p)
        cur = best.get(ev.thread_key)
        if cur is None or w > cur[1]:
            best[ev.thread_key] = (ev, w, info)
    return sorted(best.values(), key=lambda t: -t[1])


def news_availability(cov: NewsCoverage | None, p: dict) -> str | None:
    if cov is None:
        return "no news provider has run for this pair"
    if min(cov.fetch_days_ok_a, cov.fetch_days_ok_b) < p["min_fetch_days"]:
        return (
            f"news collection succeeded on only {min(cov.fetch_days_ok_a, cov.fetch_days_ok_b)} of the last "
            f"{cov.days} days for one company (need {p['min_fetch_days']})"
        )
    if cov.relevant_clusters > 0 and cov.analyzed_clusters / cov.relevant_clusters < p["min_analyzed_ratio"]:
        return f"only {cov.analyzed_clusters}/{cov.relevant_clusters} relevant stories analysed (AI not configured, failing or over budget)"
    return None


def score_news(events: list[EventEvidence] | None, cov: NewsCoverage | None, cutoff: datetime, tick_a: str, tick_b: str, p: dict) -> ComponentResult:
    reason = news_availability(cov, p)
    if events is None or reason:
        return ComponentResult.unavailable("N", reason or "no event data")
    elig = eligible_events(events, cutoff, p["lookback_days"])
    threads = thread_maxima(elig, cutoff, p)
    total = sum(w for _, w, _ in threads)
    net = sum(event_side_sign(ev, tick_a, tick_b) * w for ev, w, _ in threads)
    value = 1 - math.exp(-total / p["kappa"])
    counted = [(ev, w, info) for ev, w, info in threads if w > 0]
    if counted:
        top = counted[0][0]
        expl = (
            f"{len(counted)} asymmetric development(s) in the last {p['lookback_days']} days "
            f"(weighted mass {total:.2f}); strongest: {top.event_type.replace('_', ' ')} — {top.summary[:160]}"
        )
    else:
        expl = f"No material asymmetric developments found in {len(elig)} analysed item(s) over {p['lookback_days']} days (measured, coverage OK)."
    return ComponentResult(
        "N", clip01(value), True, expl,
        inputs={"mass": total, "net_signed_mass": net, "threads": len(threads), "events_considered": len(elig),
                "contributions": [{"event_id": ev.id, "thread": ev.thread_key, "weight": round(w, 4), **info} for ev, w, info in threads[:20]]},
        sources=[ev.id for ev, w, _ in counted],
        signed=net,
    )


def score_persistence(events: list[EventEvidence] | None, cov: NewsCoverage | None, cutoff: datetime, pn: dict, pp: dict) -> ComponentResult:
    reason = news_availability(cov, pn)
    if events is None or reason:
        return ComponentResult.unavailable("P", reason or "no event data")
    threads = thread_maxima(eligible_events(events, cutoff, pn["lookback_days"]), cutoff, pn)
    mass, parts = 0.0, []
    for ev, w, _ in threads:
        if w <= 0:
            continue
        pi = clip01(ev.persistence_days / pp["full_persistence_days"])
        if ev.event_type in pp["structural_event_types"]:
            pi = max(pi, pp["structural_floor"])
        mass += w * pi
        parts.append({"event_id": ev.id, "persistence_days": ev.persistence_days, "pi": round(pi, 3), "w": round(w, 4)})
    value = 1 - math.exp(-mass / pp["kappa"])
    expl = (
        f"Persistence-weighted asymmetric mass {mass:.2f} across {len(parts)} development(s)"
        if parts else "No persistent asymmetric developments in the lookback (measured)."
    )
    return ComponentResult("P", clip01(value), True, expl, inputs={"mass": mass, "parts": parts[:20]}, sources=[x["event_id"] for x in parts])


# --------------------------------------------------------------------------- market (M)

def score_market(market: MarketData, p: dict) -> ComponentResult:
    prices = market.prices
    rets = log_returns(prices)
    subs: dict[str, float] = {}
    info: dict = {"is_total_return": market.is_total_return, "provider": market.provider, "last_date": str(market.last_date)}

    rl, rs = window_corr(rets, p["long_window"]), window_corr(rets, p["short_window"])
    if rl is not None and rs is not None:
        subs["corr_deterioration"] = clip01((rl - rs) / p["corr_scale"])
        info.update(rho_long=rl, rho_short=rs)

    fit, z = out_of_sample_zscores(prices, p["formation_window"], p["zscore_window"])
    if fit is not None and len(z):
        zl = float(z.iloc[-1])
        subs["spread_dislocation"] = clip01(abs(zl) / p["z_scale"])
        info.update(z_last=zl, hedge_beta=fit.beta, formation_start=str(fit.formation_start.date()), formation_end=str(fit.formation_end.date()))

    if len(prices) >= p["coint_window"]:
        pv = engle_granger_pvalue(prices.tail(p["coint_window"]))
        if pv is not None:
            subs["cointegration_breakdown"] = clip01((pv - p["p_low"]) / (p["p_high"] - p["p_low"]))
            info["coint_pvalue"] = pv

    info["subfeatures"] = subs
    if len(subs) < p["min_subfeatures"]:
        return ComponentResult.unavailable("M", f"only {len(subs)} market sub-features computable ({len(prices)} aligned closes)", **info)
    v = float(np.mean(list(subs.values())))
    bits = []
    if "rho_long" in info:
        bits.append(f"corr {info['rho_long']:.2f}→{info['rho_short']:.2f}")
    if "z_last" in info:
        bits.append(f"out-of-sample spread z {info['z_last']:+.2f}")
    if "coint_pvalue" in info:
        bits.append(f"Engle-Granger p={info['coint_pvalue']:.2f}")
    return ComponentResult("M", clip01(v), True, "; ".join(bits), inputs=info)


# --------------------------------------------------------------------------- fundamentals (F)

def _latest_pit(facts: list[Fact], metric: str, cutoff: date, max_stale: int) -> tuple[Fact, Fact] | None:
    """Latest period known at `cutoff` and the comparable period one year earlier (same duration),
    using only facts filed on/before the cutoff; restatements filed before the cutoff supersede."""
    known: dict[tuple[date, int | None], Fact] = {}
    for f in facts:
        if f.metric != metric or f.filed_at > cutoff:
            continue
        key = (f.period_end, f.duration_days)
        if key not in known or f.filed_at > known[key].filed_at:
            known[key] = f
    if not known:
        return None
    latest = max(known.values(), key=lambda f: (f.period_end, -(f.duration_days or 0)))
    if (cutoff - latest.filed_at).days > max_stale:
        return None
    target = latest.period_end - timedelta(days=365)
    # 52/53-week calendars (e.g. Coca-Cola) make the same fiscal quarter a few days longer or shorter year to year
    prior = [f for (pe, dur), f in known.items() if _same_length(dur, latest.duration_days) and abs((pe - target).days) <= 20]
    if not prior:
        return None
    return latest, min(prior, key=lambda f: abs((f.period_end - target).days))


def _same_length(d1: int | None, d2: int | None) -> bool:
    if d1 is None or d2 is None:
        return d1 == d2
    return abs(d1 - d2) <= max(7, 0.05 * d2)


def _growth(cur: float, prev: float) -> float | None:
    if prev == 0:
        return None
    return max(-1.0, min(1.0, (cur - prev) / abs(prev)))


def fundamental_metrics(profile: CompanyProfile, cutoff: date, p: dict) -> dict:
    out: dict = {}
    rev = _latest_pit(profile.facts, "revenue", cutoff, p["max_staleness_days"])
    if rev:
        g = _growth(rev[0].value, rev[1].value)
        if g is not None:
            out["revenue_growth"] = g
            out["revenue_period"] = str(rev[0].period_end)
            out["revenue_accession"] = rev[0].accession
        opi = _latest_pit(profile.facts, "operating_income", cutoff, p["max_staleness_days"])
        if opi and opi[0].period_end == rev[0].period_end and opi[1].period_end == rev[1].period_end and rev[0].value and rev[1].value:
            out["op_margin_change"] = opi[0].value / rev[0].value - opi[1].value / rev[1].value
            out["op_margin"] = opi[0].value / rev[0].value
    eps = _latest_pit(profile.facts, "eps_diluted", cutoff, p["max_staleness_days"])
    if eps:
        g = _growth(eps[0].value, eps[1].value)
        if g is not None:
            out["eps_growth"] = g
    return out


def score_fundamentals(pa: CompanyProfile, pb: CompanyProfile, cutoff: date, p: dict) -> ComponentResult:
    ma, mb = fundamental_metrics(pa, cutoff, p), fundamental_metrics(pb, cutoff, p)
    subs, signed = {}, []
    spec = [("revenue_growth", "revenue_growth_scale"), ("op_margin_change", "op_margin_change_scale"), ("eps_growth", "eps_growth_scale")]
    for key, scale in spec:
        if key in ma and key in mb:
            diff = ma[key] - mb[key]
            subs[key] = clip01(abs(diff) / p[scale])
            signed.append(max(-1.0, min(1.0, diff / p[scale])))
    inputs = {"a": ma, "b": mb, "subfeatures": subs}
    if len(subs) < p["min_subfeatures"]:
        return ComponentResult.unavailable(
            "F", f"only {len(subs)} comparable point-in-time fundamental metrics (need {p['min_subfeatures']}; SEC facts filed within {p['max_staleness_days']} days)", **inputs
        )
    v = float(np.mean(list(subs.values())))
    bits = []
    if "revenue_growth" in subs:
        bits.append(f"revenue growth {pa.ticker} {ma['revenue_growth']:+.1%} vs {pb.ticker} {mb['revenue_growth']:+.1%}")
    if "op_margin_change" in subs:
        bits.append(f"op-margin change {ma['op_margin_change']:+.1%} vs {mb['op_margin_change']:+.1%}")
    if "eps_growth" in subs:
        bits.append(f"EPS growth {ma['eps_growth']:+.1%} vs {mb['eps_growth']:+.1%}")
    return ComponentResult(
        "F", clip01(v), True, "; ".join(bits), inputs=inputs,
        sources=[x for x in (ma.get("revenue_accession"), mb.get("revenue_accession")) if x],
        signed=float(np.mean(signed)),
    )


# --------------------------------------------------------------------------- structure (S)

def sic_distance(a: str | None, b: str | None) -> float | None:
    if not a or not b:
        return None
    if a == b:
        return 0.0
    if a[:3] == b[:3]:
        return 1 / 3
    if a[:2] == b[:2]:
        return 2 / 3
    return 1.0


def score_structure(market: MarketData, pa: CompanyProfile, pb: CompanyProfile, cutoff: date, pf: dict, p: dict) -> ComponentResult:
    subs, info = {}, {}
    sd = sic_distance(pa.sic_code, pb.sic_code)
    if sd is not None:
        subs["sic_distance"] = sd
        info["sic"] = [pa.sic_code, pb.sic_code]
    prices = market.prices
    if market.benchmark is not None and len(prices) > 60:
        rets = log_returns(prices)
        bench = np.log(market.benchmark).diff().dropna()
        ba = beta_to_benchmark(rets["a"], bench, p["beta_window"])
        bb = beta_to_benchmark(rets["b"], bench, p["beta_window"])
        if ba is not None and bb is not None:
            subs["beta_gap"] = clip01(abs(ba - bb) / p["beta_scale"])
            info.update(beta_a=ba, beta_b=bb)
    if len(prices) > 60:
        rets = log_returns(prices)
        va, vb = annualized_vol(rets["a"], p["vol_window"]), annualized_vol(rets["b"], p["vol_window"])
        if va and vb:
            subs["volatility_ratio"] = clip01(abs(math.log(va / vb)) / math.log(2))
            info.update(vol_a=va, vol_b=vb)
    ra = _latest_annual(pa, cutoff, pf["max_staleness_days"] + 200)
    rb = _latest_annual(pb, cutoff, pf["max_staleness_days"] + 200)
    if ra and rb and ra > 0 and rb > 0:
        subs["size_gap"] = clip01(abs(math.log(ra / rb)) / p["size_scale_log"])
        info.update(revenue_a=ra, revenue_b=rb)
    info["subfeatures"] = subs
    if len(subs) < p["min_subfeatures"]:
        return ComponentResult.unavailable("S", f"only {len(subs)} structural sub-features available", **info)
    bits = [f"{k.replace('_', ' ')} {v:.2f}" for k, v in subs.items()]
    return ComponentResult("S", clip01(float(np.mean(list(subs.values())))), True, "; ".join(bits), inputs=info)


def _latest_annual(profile: CompanyProfile, cutoff: date, max_stale: int) -> float | None:
    cands = [f for f in profile.facts if f.metric == "revenue" and f.filed_at <= cutoff and (f.duration_days or 0) > 340]
    if not cands:
        return None
    f = max(cands, key=lambda f: (f.period_end, f.filed_at))
    return f.value if (cutoff - f.filed_at).days <= max_stale else None


def prices_for(market: MarketData) -> pd.DataFrame:
    return market.prices if {"a", "b"}.issubset(market.prices.columns) else align(market.prices.iloc[:, 0], market.prices.iloc[:, 1])
