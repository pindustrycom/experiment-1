"""Historical walk-forward research on stored price history (and point-in-time SEC facts).

Design:
* observation dates t: first NYSE session of each year (pair-years, as in the original experiment)
  or of each month; every feature uses data with close time <= t only (truncate_market);
* label: D(t, H) >= threshold on the same provider's adjusted closes;
* time split train / embargo / test; the embargo removes observations whose label windows would
  overlap the test period (H=504 sessions ~ 2 years);
* SI-v2 weights for the market+fundamental subset are optimised on TRAIN only and then evaluated
  once on TEST; ablations and fixed alternative weights are evaluated on TEST without refitting;
* news components (N, P) need point-in-time archived news. Unless events ingested before t exist,
  they are reported as unavailable instead of being back-filled with today's knowledge.
"""
from __future__ import annotations

import itertools
import logging
from datetime import UTC, date, datetime, timedelta

import numpy as np
import pandas as pd
from sqlalchemy import select
from sqlalchemy.orm import Session

from app.models import BacktestRun, Company, Pair
from app.research.evaluation import evaluate, roc_auc
from app.research.market_calendar import add_trading_days, is_trading_day
from app.scoring.pit import close_timestamp, truncate_market
from app.scoring.si_v1 import compute_si_v1
from app.scoring.si_v2 import compute_si_v2
from app.scoring.types import MarketData, PairInputs
from app.services import fundamentals_service, prices_service

log = logging.getLogger(__name__)

REPORTED_BENCHMARK = {
    "label": "Original SI-v1 experiment (reported results, 100 pairs)",
    "params": {"pairs": 100, "observations": 300, "train": "2015-2019", "embargo": "2020-2021", "evaluation": "2022-2024",
               "target": "|cumulative log total-return gap| >= 0.20 over two years"},
    "metrics": {
        "si_v1": {"roc_auc": 0.6655, "top_quartile_divergence_rate": 0.747, "bottom_quartile_divergence_rate": 0.333},
        "correlation_only": {"roc_auc": 0.6596},
        "previous_return_gap": {"roc_auc": 0.6180},
    },
    "limitations": [
        "Reported by the researcher from a previous run; not reproduced by this application.",
        "SI-v1 did not anticipate the KO/PEP divergence from the January 2024 observation.",
        "Historical results are not evidence that the news-enhanced SI-v2 works.",
    ],
}

PARTIAL = ["F", "M", "S"]
SI_V2_PARAMS_DEFAULT_PARTIAL = {"F": 0.25, "M": 0.20, "S": 0.10}  # provisional SI-v2 weights restricted to F, M, S


def seed_reported_benchmark(db: Session) -> None:
    if db.scalar(select(BacktestRun.id).where(BacktestRun.kind == "reported_historical")) is None:
        db.add(BacktestRun(kind="reported_historical", label=REPORTED_BENCHMARK["label"], params=REPORTED_BENCHMARK["params"],
                           metrics=REPORTED_BENCHMARK["metrics"], observations=[], limitations=REPORTED_BENCHMARK["limitations"]))


def observation_dates(start_year: int, end_year: int, frequency: str) -> list[date]:
    months = {"annual": [1], "quarterly": [1, 4, 7, 10], "monthly": list(range(1, 13))}
    if frequency not in months:
        raise ValueError(f"frequency must be one of {sorted(months)}")
    out = []
    for y in range(start_year, end_year + 1):
        for m in months[frequency]:
            # last trading day of the month, as the original experiment scored at the end of January
            d = (date(y + 1, 1, 1) if m == 12 else date(y, m + 1, 1)) - timedelta(days=1)
            while not is_trading_day(d):
                d = date.fromordinal(d.toordinal() - 1)
            out.append(d)
    return out


def build_observations(db: Session, pairs: list[Pair], dates: list[date], horizon: int, threshold: float) -> list[dict]:
    bench = db.scalar(select(Company).where(Company.is_benchmark.is_(True)))
    obs = []
    for pair in pairs:
        full = prices_service.pair_market_data(db, pair.company_a, pair.company_b, datetime.now(UTC), bench)
        prices = full.prices
        if prices.empty:
            continue
        fa_all = fundamentals_service.company_profile(db, pair.company_a, date.today())
        fb_all = fundamentals_service.company_profile(db, pair.company_b, date.today())
        for t in dates:
            # cutoff = close of the session before t (information available at the open of t)
            prev = prices.index[prices.index < pd.Timestamp(t)]
            if len(prev) == 0:
                continue
            base = pd.Timestamp(prev[-1]).date()
            cutoff = close_timestamp(base)
            end = add_trading_days(base, horizon)
            if pd.Timestamp(end) not in prices.index:
                continue
            hist = truncate_market(prices, cutoff)
            mk = MarketData(prices=hist, benchmark=None if full.benchmark is None else truncate_market(full.benchmark, cutoff),
                            is_total_return=full.is_total_return, provider=full.provider)
            r1 = compute_si_v1(mk)
            if r1.value is None:
                continue
            fa = type(fa_all)(fa_all.ticker, fa_all.sic_code, [f for f in fa_all.facts if f.filed_at <= base])
            fb = type(fb_all)(fb_all.ticker, fb_all.sic_code, [f for f in fb_all.facts if f.filed_at <= base])
            r2 = compute_si_v2(PairInputs(pair.company_a.ticker, pair.company_b.ticker, cutoff, mk, None, None, fa, fb))
            pa0, pb0 = prices.loc[pd.Timestamp(base)]
            pa1, pb1 = prices.loc[pd.Timestamp(end)]
            d = abs(np.log(pa1 / pa0) - np.log(pb1 / pb0))
            obs.append({
                "pair": pair.slug, "industry": pair.industry or "unclassified", "t": str(t), "base": str(base), "end": str(end),
                "si_v1": r1.value, "correlation_only": r1.baselines["correlation_only"], "previous_return_gap": r1.baselines["previous_return_gap"],
                **{f"comp_{k}": (r2.components[k].value if r2.components[k].available else None) for k in ["N", "F", "M", "P", "S"]},
                "divergence": float(d), "diverged": int(d >= threshold), "provider": full.provider, "total_return": full.is_total_return,
            })
    return obs


def partial_score(row: dict, w: dict) -> float | None:
    avail = {k: row[f"comp_{k}"] for k in w if row.get(f"comp_{k}") is not None and w[k] > 0}
    if "M" not in avail:
        return None
    tot = sum(w[k] for k in avail)
    return 100 * sum(w[k] * v for k, v in avail.items()) / tot if tot else None


def weight_grid(keys: list[str], step: float = 0.1) -> list[dict]:
    n = int(round(1 / step))
    out = []
    for combo in itertools.product(range(n + 1), repeat=len(keys)):
        if sum(combo) == n:
            out.append({k: c * step for k, c in zip(keys, combo, strict=True)})
    return out


def _eval_rows(rows: list[dict], key, n_boot: int) -> dict:
    vals = [(key(r), r) for r in rows]
    vals = [(v, r) for v, r in vals if v is not None]
    return evaluate([v for v, _ in vals], [r["diverged"] for _, r in vals], [r["industry"] for _, r in vals],
                    [r["divergence"] for _, r in vals], n_boot=n_boot)


def run_walk_forward(db: Session, train=(2015, 2019), embargo=(2020, 2021), test=(2022, 2024), horizon: int = 504,
                     threshold: float = 0.20, frequency: str = "annual", n_boot: int = 500) -> BacktestRun:
    pairs = db.scalars(select(Pair).where(Pair.status != "archived").order_by(Pair.slug)).all()
    dates = observation_dates(train[0], test[1], frequency)
    obs = build_observations(db, pairs, dates, horizon, threshold)
    tr = [o for o in obs if train[0] <= int(o["t"][:4]) <= train[1] and int(o["end"][:4]) < embargo[1] + 1]
    te = [o for o in obs if test[0] <= int(o["t"][:4]) <= test[1]]
    limitations = [
        "News components N and P are unavailable for historical dates: no point-in-time news archive was ingested before those dates.",
        "SI-v2 here is the market+fundamental subset (F, M, S) re-weighted over available components; it is not the full news index.",
    ]
    if any(not o["total_return"] for o in obs):
        limitations.append("Some series are price-return only (provider lacks dividend adjustment).")
    if frequency != "annual":
        limitations.append(f"{frequency.capitalize()} observations have heavily overlapping label windows; treat CIs as optimistic.")
    default_w = {k: v for k, v in SI_V2_PARAMS_DEFAULT_PARTIAL.items()}
    best_w, best_auc = None, -1.0
    if tr:
        for w in weight_grid(PARTIAL):
            if w["M"] == 0:
                continue
            sc = [(partial_score(o, w), o["diverged"]) for o in tr]
            sc = [x for x in sc if x[0] is not None]
            a = roc_auc([s for s, _ in sc], [y for _, y in sc]) if sc else None
            if a is not None and a > best_auc:
                best_w, best_auc = w, a
    metrics: dict = {"n_observations": len(obs), "n_train": len(tr), "n_test": len(te), "test": {}, "train": {}}
    for name, split in (("train", tr), ("test", te)):
        m = metrics[name]
        m["si_v1"] = _eval_rows(split, lambda r: r["si_v1"], n_boot)
        m["correlation_only"] = _eval_rows(split, lambda r: r["correlation_only"], n_boot)
        m["previous_return_gap"] = _eval_rows(split, lambda r: r["previous_return_gap"], n_boot)
        m["si_v2_partial_default"] = _eval_rows(split, lambda r: partial_score(r, default_w), n_boot)
        if best_w:
            m["si_v2_partial_train_optimised"] = _eval_rows(split, lambda r, w=best_w: partial_score(r, w), n_boot)
    abl = {}
    for drop in PARTIAL:
        if drop == "M":
            continue  # M is required
        w = {k: (0.0 if k == drop else v) for k, v in default_w.items()}
        abl[f"without_{drop}"] = _eval_rows(te, lambda r, w=w: partial_score(r, w), n_boot)
    metrics["ablations_test"] = abl
    metrics["alternative_weights_test"] = {
        "equal": _eval_rows(te, lambda r: partial_score(r, {"F": 1 / 3, "M": 1 / 3, "S": 1 / 3}), n_boot),
        "market_only": _eval_rows(te, lambda r: partial_score(r, {"F": 0, "M": 1, "S": 0}), n_boot),
    }
    metrics["optimised_weights"] = {"weights": best_w, "train_auc": best_auc if best_w else None, "fitted_on": f"{train[0]}-{train[1]} only"}
    run = BacktestRun(kind="walk_forward", label=f"Walk-forward {frequency} H={horizon} thr={threshold}",
                      params={"train": train, "embargo": embargo, "test": test, "horizon": horizon, "threshold": threshold,
                              "frequency": frequency, "pairs": [p.slug for p in pairs]},
                      metrics=metrics, observations=obs, limitations=limitations,
                      is_demo=any(o["provider"] == "demo" for o in obs))
    db.add(run)
    db.commit()
    return run
