import numpy as np
import pytest

from app.research.evaluation import brier, cluster_bootstrap, confusion, evaluate, pr_auc, quartile_rates, roc_auc


def test_roc_auc_perfect_and_random():
    assert roc_auc([1, 2, 3, 4], [0, 0, 1, 1]) == 1.0
    assert roc_auc([4, 3, 2, 1], [0, 0, 1, 1]) == 0.0
    assert roc_auc([1, 1, 1, 1], [0, 1, 0, 1]) == 0.5
    assert roc_auc([1, 2], [1, 1]) is None


def test_roc_auc_matches_bruteforce():
    rng = np.random.default_rng(0)
    s, y = rng.normal(size=200), rng.integers(0, 2, 200)
    pairs = [(a, b) for a in s[y == 1] for b in s[y == 0]]
    brute = np.mean([1.0 if a > b else 0.5 if a == b else 0.0 for a, b in pairs])
    assert roc_auc(s, y) == pytest.approx(brute)


def test_pr_auc():
    assert pr_auc([0.9, 0.8, 0.1], [1, 1, 0]) == pytest.approx(1.0)
    assert pr_auc([0.1, 0.2, 0.9], [1, 0, 0]) == pytest.approx(1 / 3)


def test_brier_and_confusion():
    assert brier([1, 0], [1, 0]) == 0
    c = confusion([80, 20, 75, 10], [1, 0, 0, 1], 70)
    assert (c["tp"], c["fp"], c["fn"], c["tn"]) == (1, 1, 1, 1)


def test_quartiles_and_bootstrap():
    s = np.arange(40)
    y = (s >= 20).astype(int)
    q = quartile_rates(s, y)
    assert q[0]["divergence_rate"] == 0 and q[3]["divergence_rate"] == 1
    ci = cluster_bootstrap(s, y, np.repeat(list("abcd"), 10), n_boot=200)
    assert ci is not None and ci["low"] <= ci["high"]


def test_evaluate_insufficient():
    r = evaluate([10, 20], [0, 1], ["x", "y"])
    assert r["status"] == "insufficient"


# ------------------------------------------------------------------ target definition
def test_realized_divergence_is_absolute_log_return_gap():
    import math

    from app.services.predictions_service import realized_divergence
    d, la, lb = realized_divergence(100.0, 120.0, 50.0, 50.0)
    assert la == pytest.approx(math.log(1.2)) and lb == pytest.approx(0.0)
    assert d == pytest.approx(abs(math.log(1.2)))
    # symmetric in which leg outperforms
    assert realized_divergence(50.0, 50.0, 100.0, 120.0)[0] == pytest.approx(d)
    # dividend re-basing (same factor on both endpoints of one leg) does not change D
    assert realized_divergence(100.0 * 0.97, 120.0 * 0.97, 50.0, 50.0)[0] == pytest.approx(d)


def test_thresholds_per_horizon():
    from app.services.settings_service import DEFAULTS, divergence_threshold
    pred = DEFAULTS["predictions"]
    assert divergence_threshold(504, pred) == pytest.approx(0.20)
    assert divergence_threshold(30, pred) == pytest.approx(0.0488, abs=1e-4)
    assert divergence_threshold(252, pred) == pytest.approx(0.1414, abs=1e-4)
    assert divergence_threshold(30, {**pred, "threshold_overrides": {"30": 0.1}}) == 0.1
    assert divergence_threshold(90, {**pred, "threshold_scaling": "constant"}) == 0.2


def test_maturity_counts_nyse_sessions():
    from datetime import date

    from app.research.market_calendar import add_trading_days
    assert add_trading_days(date(2026, 11, 25), 1) == date(2026, 11, 27)  # skips Thanksgiving
    assert add_trading_days(date(2026, 1, 2), 504) > date(2027, 12, 1)
