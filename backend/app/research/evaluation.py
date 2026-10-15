"""Classification / calibration metrics with no external ML dependency (numpy + scipy)."""
from __future__ import annotations

import numpy as np
from scipy.stats import rankdata, spearmanr

MIN_OBS = 20  # below this, or with a single class, metrics are reported as insufficient


def _arr(x) -> np.ndarray:
    return np.asarray(x, dtype=float)


def roc_auc(scores, labels) -> float | None:
    s, y = _arr(scores), _arr(labels)
    pos, neg = y == 1, y == 0
    n1, n0 = pos.sum(), neg.sum()
    if n1 == 0 or n0 == 0:
        return None
    ranks = rankdata(s)  # average ranks handle ties (Mann-Whitney U)
    return float((ranks[pos].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))


def pr_auc(scores, labels) -> float | None:
    """Average precision (step-wise area under the precision-recall curve)."""
    s, y = _arr(scores), _arr(labels)
    if y.sum() == 0 or y.sum() == len(y):
        return None
    order = np.argsort(-s, kind="mergesort")
    s, y = s[order], y[order]
    # group ties so precision is evaluated at distinct thresholds
    distinct = np.r_[np.where(np.diff(s))[0], len(s) - 1]
    tp = np.cumsum(y)[distinct]
    fp = (distinct + 1) - tp
    precision = tp / (tp + fp)
    recall = tp / y.sum()
    recall_prev = np.r_[0, recall[:-1]]
    return float(np.sum((recall - recall_prev) * precision))


def brier(probs, labels) -> float | None:
    p, y = _arr(probs), _arr(labels)
    if len(p) == 0:
        return None
    return float(np.mean((p - y) ** 2))


def calibration_bins(probs, labels, n_bins: int = 10) -> list[dict]:
    p, y = _arr(probs), _arr(labels)
    edges = np.linspace(0, 1, n_bins + 1)
    out = []
    for i in range(n_bins):
        m = (p >= edges[i]) & ((p < edges[i + 1]) if i < n_bins - 1 else (p <= edges[i + 1]))
        if m.sum():
            out.append({"bin": f"{edges[i]:.1f}-{edges[i+1]:.1f}", "mean_predicted": float(p[m].mean()), "observed_rate": float(y[m].mean()), "n": int(m.sum())})
    return out


def quartile_rates(scores, labels) -> list[dict]:
    s, y = _arr(scores), _arr(labels)
    if len(s) < 4:
        return []
    q = np.quantile(s, [0.25, 0.5, 0.75])
    bucket = np.searchsorted(q, s, side="right")
    return [{"quartile": f"Q{i+1}", "n": int((bucket == i).sum()), "divergence_rate": float(y[bucket == i].mean()) if (bucket == i).any() else None,
             "score_range": [float(s[bucket == i].min()), float(s[bucket == i].max())] if (bucket == i).any() else None} for i in range(4)]


def confusion(scores, labels, threshold: float) -> dict:
    s, y = _arr(scores), _arr(labels).astype(bool)
    pred = s >= threshold
    tp, fp = int((pred & y).sum()), int((pred & ~y).sum())
    fn, tn = int((~pred & y).sum()), int((~pred & ~y).sum())
    return {"threshold": threshold, "tp": tp, "fp": fp, "fn": fn, "tn": tn,
            "precision": tp / (tp + fp) if tp + fp else None, "recall": tp / (tp + fn) if tp + fn else None}


def cluster_bootstrap(scores, labels, clusters, metric=roc_auc, n_boot: int = 1000, seed: int = 7, alpha: float = 0.05) -> dict | None:
    """Percentile CI from resampling whole clusters (industries) with replacement, which respects
    within-industry dependence between pairs."""
    s, y, c = _arr(scores), _arr(labels), np.asarray(clusters)
    uniq = np.unique(c)
    if len(uniq) < 2:
        return None
    idx_by = {u: np.where(c == u)[0] for u in uniq}
    rng = np.random.default_rng(seed)
    vals = []
    for _ in range(n_boot):
        pick = rng.choice(uniq, size=len(uniq), replace=True)
        idx = np.concatenate([idx_by[u] for u in pick])
        v = metric(s[idx], y[idx])
        if v is not None:
            vals.append(v)
    if len(vals) < n_boot * 0.5:
        return None
    return {"low": float(np.quantile(vals, alpha / 2)), "high": float(np.quantile(vals, 1 - alpha / 2)), "n_boot": len(vals), "cluster": "industry"}


def magnitude_correlation(scores, magnitudes) -> dict | None:
    s, m = _arr(scores), _arr(magnitudes)
    if len(s) < MIN_OBS:
        return None
    rho, p = spearmanr(s, m)
    return {"spearman_rho": float(rho), "p_value": float(p)}


def evaluate(scores, labels, clusters, magnitudes=None, thresholds=(50.0, 60.0, 70.0), n_boot: int = 1000) -> dict:
    s, y = _arr(scores), _arr(labels)
    n = len(s)
    out: dict = {"n": n, "positives": int(y.sum()), "base_rate": float(y.mean()) if n else None}
    if n < MIN_OBS or y.sum() == 0 or y.sum() == n:
        out["status"] = "insufficient"
        out["reason"] = f"need >= {MIN_OBS} resolved observations with both outcomes (have {n}, {int(y.sum())} positive)"
        return out
    probs = np.clip(s / 100.0, 0, 1)
    out.update(
        status="ok",
        roc_auc=roc_auc(s, y),
        roc_auc_ci=cluster_bootstrap(s, y, clusters, roc_auc, n_boot),
        pr_auc=pr_auc(s, y),
        brier=brier(probs, y),
        brier_note="score/100 used as a pseudo-probability; the index is not calibrated",
        calibration=calibration_bins(probs, y),
        quartiles=quartile_rates(s, y),
        confusion=[confusion(s, y, t) for t in thresholds],
    )
    if magnitudes is not None:
        out["magnitude"] = magnitude_correlation(s, magnitudes)
    return out
