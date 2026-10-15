"""SI-v1: market-only benchmark, SI-v1 = (25C + 15D + 25R) * 100 / 65.

Ported from the original experiment (run_experiment.py, index_components / calculate_rows):

C  correlation weakness       = clip((0.90 - rho_1y) / 0.90, 0, 1)
D  correlation deterioration  = clip((rho_5y - rho_1y) / 0.30, 0, 1)
R  past return divergence     = clip(|1-year total-return gap in percentage points| / 50, 0, 1)

rho_1y / rho_5y are Pearson correlations of daily log total returns over the last 252 / 1260
sessions. As in the original, a score needs the last 1261 closes with at most 10 missing and at
least 1250 returns. All three components are required; SI-v1 has no provisional mode.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from app.research.stats import align, clip01
from app.scoring.params import SI_V1_DEFINITION
from app.scoring.types import ComponentResult, MarketData


@dataclass
class SIv1Result:
    value: float | None
    components: dict[str, ComponentResult]
    baselines: dict[str, float | None]
    explanation: str


def index_components(rho_short: float, rho_long: float, gap_pp: float, p: dict | None = None) -> tuple[float, float, float]:
    """C, D, R in [0, 1] exactly as the original index_components (which returns them in points)."""
    p = p or SI_V1_DEFINITION["params"]
    c = clip01((p["c_anchor"] - rho_short) / p["c_anchor"])
    d = clip01((rho_long - rho_short) / p["d_scale"])
    r = clip01(gap_pp / p["r_scale_pp"])
    return c, d, r


def _corr(rets: pd.DataFrame) -> float | None:
    if rets["a"].std() == 0 or rets["b"].std() == 0:
        return None
    v = float(rets["a"].corr(rets["b"]))
    return v if np.isfinite(v) else None


def compute_si_v1(market: MarketData, definition: dict | None = None) -> SIv1Result:
    d = definition or SI_V1_DEFINITION
    p, w = d["params"], d["weights"]
    L, S, RW = p["long_window"], p["short_window"], p["r_window"]
    prices = market.prices
    if not {"a", "b"}.issubset(prices.columns):
        prices = align(prices.iloc[:, 0], prices.iloc[:, 1])
    past = prices[["a", "b"]].tail(L + 1)
    n = len(past)

    reason = None
    if n < L + 1:
        reason = f"need {L + 1} closes (5 years), have {n}"
    elif int(past.isna().values.sum()) > p["max_missing_closes"] or bool((past <= 0).values.any()):
        reason = "missing historical prices"
    rets = None
    if reason is None:
        rets = np.log(past).diff().dropna()
        if len(rets) < p["min_returns"]:
            reason = f"need {p['min_returns']} returns, have {len(rets)}"

    rho_long = rho_short = gap_pp = None
    if reason is None:
        rho_long = _corr(rets.tail(L))
        rho_short = _corr(rets.tail(S))
        if rho_long is None or rho_short is None:
            reason = "undefined correlation"
        else:
            a, b = past["a"], past["b"]
            gap_pp = abs(100 * (a.iloc[-1] / a.iloc[-1 - RW] - 1) - 100 * (b.iloc[-1] / b.iloc[-1 - RW] - 1))

    comps: dict[str, ComponentResult] = {}
    if reason is not None:
        for k in "CDR":
            comps[k] = ComponentResult.unavailable(k, reason, n_prices=n)
        value = None
        expl = f"SI-v1 unavailable: {reason}."
    else:
        c, dd, r = index_components(rho_short, rho_long, gap_pp, p)
        comps["C"] = ComponentResult("C", c, True, f"1-year return correlation {rho_short:.3f}", {"rho_short": rho_short})
        comps["D"] = ComponentResult(
            "D", dd, True, f"correlation fell from {rho_long:.3f} (5y) to {rho_short:.3f} (1y)" if rho_short < rho_long
            else f"1-year correlation {rho_short:.3f} not below 5-year {rho_long:.3f}",
            {"rho_long": rho_long, "rho_short": rho_short},
        )
        a, b = past["a"], past["b"]
        signed = 100 * (a.iloc[-1] / a.iloc[-1 - RW] - 1) - 100 * (b.iloc[-1] / b.iloc[-1 - RW] - 1)
        comps["R"] = ComponentResult("R", r, True, f"1-year total-return gap {signed:+.1f} percentage points",
                                     {"gap_pp": gap_pp}, signed=float(signed))
        value = sum(w[k] * comps[k].value for k in w) * 100 / sum(w.values())
        expl = f"SI-v1 {value:.1f}: C={c:.2f}, D={dd:.2f}, R={r:.2f}."

    baselines = {
        # Raw features the original experiment calibrated as baselines, mapped to 0-100 risk.
        "correlation_only": None if rho_short is None else 100 * clip01(1 - rho_short),
        "previous_return_gap": None if gap_pp is None else 100 * clip01(gap_pp / p["r_scale_pp"]),
        "rho_long": rho_long,
        "rho_short": rho_short,
        "gap_pp": gap_pp,
    }
    return SIv1Result(value=None if value is None else round(value, 4), components=comps, baselines=baselines, explanation=expl)
