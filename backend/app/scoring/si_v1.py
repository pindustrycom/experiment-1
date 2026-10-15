"""SI-v1: market-only benchmark, SI-v1 = (25C + 15D + 25R) * 100 / 65.

C  correlation weakness       = clip(1 - rho_long, 0, 1)
D  correlation deterioration  = clip((rho_long - rho_short) / 0.5, 0, 1)
R  relative-price divergence  = clip(|cumulative log-return gap over long window| / 0.40, 0, 1)

rho_* are Pearson correlations of daily log total returns over the last 504 / 126 sessions.
All three components are required; SI-v1 has no provisional mode.
"""
from __future__ import annotations

from dataclasses import dataclass

from app.research.stats import align, clip01, cumulative_log_gap, log_returns, window_corr
from app.scoring.params import SI_V1_DEFINITION
from app.scoring.types import ComponentResult, MarketData


@dataclass
class SIv1Result:
    value: float | None
    components: dict[str, ComponentResult]
    baselines: dict[str, float | None]
    explanation: str


def compute_si_v1(market: MarketData, definition: dict | None = None) -> SIv1Result:
    d = definition or SI_V1_DEFINITION
    p, w = d["params"], d["weights"]
    L, S = p["long_window"], p["short_window"]
    prices = market.prices
    if not {"a", "b"}.issubset(prices.columns):
        prices = align(prices.iloc[:, 0], prices.iloc[:, 1])
    rets = log_returns(prices)

    rho_long = window_corr(rets, L)
    rho_short = window_corr(rets, S)
    gap = cumulative_log_gap(prices, L)
    n = len(rets)

    comps: dict[str, ComponentResult] = {}
    if rho_long is None:
        comps["C"] = ComponentResult.unavailable("C", f"need {int(0.9 * L)} overlapping returns, have {n}", n_returns=n)
    else:
        comps["C"] = ComponentResult("C", clip01(1 - rho_long), True, f"504-day return correlation {rho_long:.3f}", {"rho_long": rho_long})
    if rho_long is None or rho_short is None:
        comps["D"] = ComponentResult.unavailable("D", "insufficient return history", n_returns=n)
    else:
        v = clip01((rho_long - rho_short) / p["d_scale"])
        comps["D"] = ComponentResult(
            "D", v, True, f"correlation fell from {rho_long:.3f} (504d) to {rho_short:.3f} (126d)" if rho_short < rho_long
            else f"recent correlation {rho_short:.3f} not below long-run {rho_long:.3f}",
            {"rho_long": rho_long, "rho_short": rho_short},
        )
    if gap is None:
        comps["R"] = ComponentResult.unavailable("R", f"need {L + 1} aligned closes", n_prices=len(prices))
    else:
        comps["R"] = ComponentResult(
            "R", clip01(abs(gap) / p["r_scale"]), True, f"2-year cumulative log-return gap {gap:+.3f}", {"log_gap": gap}, signed=gap
        )

    total_w = sum(w.values())
    if all(c.available for c in comps.values()):
        value = sum(w[k] * comps[k].value for k in w) * 100 / total_w  # type: ignore[operator]
        expl = (
            f"SI-v1 {value:.1f}: C={comps['C'].value:.2f}, D={comps['D'].value:.2f}, R={comps['R'].value:.2f}."
        )
    else:
        value = None
        missing = [k for k, c in comps.items() if not c.available]
        expl = f"SI-v1 unavailable: missing {', '.join(missing)} (insufficient price history)."

    baselines = {
        "correlation_only": None if rho_long is None else 100 * clip01(1 - rho_long),
        "previous_return_gap": None if gap is None else 100 * clip01(abs(gap) / p["r_scale"]),
        "rho_long": rho_long,
        "rho_short": rho_short,
        "log_gap_long": gap,
    }
    return SIv1Result(value=None if value is None else round(value, 4), components=comps, baselines=baselines, explanation=expl)
