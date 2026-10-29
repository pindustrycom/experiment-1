"""Leakage checks: changing the future must not change any earlier feature or forecast."""
import numpy as np
import pandas as pd
import torch

import baselines as bl
import data
import models


def _panel(n=24 * 60, seed=0):
    rng = np.random.default_rng(seed)
    idx = pd.date_range("2021-01-01", periods=n, freq="h")
    r = rng.standard_t(4, n) * 0.6
    close = 30000 * np.exp(np.cumsum(r) / 100)
    d = pd.DataFrame({"close": close, "open": close, "high": close * 1.001, "low": close * 0.999}, index=idx)
    d["y"] = 100 * np.log(d.close)
    d["r"] = d.y.diff().fillna(0)
    d["gap"] = False
    d["lvol"] = rng.normal(10, 1, n)
    d["fng"] = 50.0
    d["dvol"] = 60.0
    return d


def test_features_causal():
    d = _panel()
    f0 = data.features(d)
    d2 = d.copy()
    cut = 900
    d2.iloc[cut:, d2.columns.get_loc("r")] *= 3
    d2.iloc[cut:, d2.columns.get_loc("lvol")] += 5
    f1 = data.features(d2)
    pd.testing.assert_frame_equal(f0.iloc[:cut], f1.iloc[:cut])


def test_vol_baselines_causal():
    d = _panel()
    r = d.r.to_numpy().copy()
    hours = d.index.hour.to_numpy()
    s = bl.seasonal(r, hours, np.ones(len(r), bool))
    v0 = bl.ewma_var(r, 0.97, s[hours], s[(hours + 1) % 24])
    r[800] = 50.0
    v1 = bl.ewma_var(r, 0.97, s[hours], s[(hours + 1) % 24])
    assert np.allclose(v0[:801], v1[:801], equal_nan=True) and not np.isclose(v0[801], v1[801])


def test_models_causal():
    torch.manual_seed(0)
    rng = np.random.default_rng(1)
    T, p = 200, 5
    r = torch.tensor(rng.normal(0, 1, T), dtype=torch.float32)
    u = torch.tensor(rng.normal(0, 1, (T, p)), dtype=torch.float32)
    lv = torch.zeros(T)
    ok = torch.ones(T, dtype=torch.bool)
    for kind in models.KINDS:
        m = models.Model(kind, p)
        with torch.no_grad():
            for prm in m.parameters():
                prm.add_(0.1 * torch.randn_like(prm))
            mu0, v0, _ = m(r[:, None], u[:, None], lv[:, None], ok[:, None], m.init_state(1))
            r2, u2 = r.clone(), u.clone()
            j = 120
            r2[j:] += 5          # the return being forecast at step j and everything after
            u2[j + 1:] += 5      # step k sees inputs of row k-1, so step j+1 is the first to see a changed row
            mu1, v1, _ = m(r2[:, None], u2[:, None], lv[:, None], ok[:, None], m.init_state(1))
        assert torch.allclose(mu0[: j + 1], mu1[: j + 1]) and torch.allclose(v0[: j + 1], v1[: j + 1]), kind
        assert not torch.allclose(mu0[j + 1:], mu1[j + 1:]), kind


def test_standardize_uses_train_only():
    d = _panel()
    f = data.features(d)[["r", "absr"]]
    end = d.index[500]
    z0 = data.standardize(f, end)
    f2 = f.copy()
    f2.iloc[600:] *= 10
    z1 = data.standardize(f2, end)
    pd.testing.assert_frame_equal(z0.iloc[:600], z1.iloc[:600])
