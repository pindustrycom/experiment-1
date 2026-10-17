import numpy as np
import pandas as pd

from exp2 import ledger


def rows(v):
    return [{"model": "ssm", "target": "y_sector", "pred": v, "sd": 1.0, "p_up": 0.6, "lo80": v - 1.28, "hi80": v + 1.28}]


def test_append_once_then_score_after_horizon(tmp_path):
    feats = pd.DataFrame({"y_sector": [1.0, -2.0, 0.5, np.nan, np.nan]}, index=pd.bdate_range("2026-01-05", periods=5))
    led = ledger.load(tmp_path / "p.csv")
    led, added = ledger.append(led, "2026-01-05", rows(0.3), "m1", 2, 100.0)
    assert added
    led, added = ledger.append(led, "2026-01-05", rows(9.9), "m2", 2, 100.0)
    assert not added and led["pred"].tolist() == [0.3]
    led, _ = ledger.append(led, "2026-01-08", rows(0.1), "m1", 2, 101.0)
    led, n = ledger.score(led, feats, 2)
    assert n == 1 and led["realized"].iloc[0] == 1.0 and led["target_date"].iloc[0] == "2026-01-07"
    assert pd.isna(led["realized"].iloc[1])
    ledger.save(led, tmp_path / "p.csv")
    again = ledger.load(tmp_path / "p.csv")
    assert again["realized"].iloc[0] == 1.0 and again["as_of"].tolist() == ["2026-01-05", "2026-01-08"]


def test_metrics():
    df = pd.DataFrame({"pred": [1.0, -1.0, 1.0, 1.0], "realized": [2.0, -1.0, -1.0, 0.5], "lo80": -5.0, "hi80": 5.0, "p_up": 0.7})
    m = ledger.metrics(df)
    assert m["n"] == 4 and m["hit"] == 0.75 and m["cover80"] == 1.0
    assert np.isclose(m["skill"], 1 - np.mean([1, 0, 4, 0.25]) / np.mean([4, 1, 1, 0.25]))
