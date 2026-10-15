import json

import numpy as np
import pandas as pd

from exp2 import cli, train
from exp2.config import Config, ModelConfig

from synth import prices
from test_news import FakeScorer

TINY = Config(model=ModelConfig(d_model=8, n_modes=8, steps=30, seeds=(0,), window=256, burn_in=64, val_sessions=100,
                                finetune_steps=10, patience=30))


def write_state(state, n=2200):
    (state / "prices").mkdir(parents=True)
    for t, df in prices(n).items():
        df.reset_index().assign(date=lambda d: d["date"].dt.strftime("%Y-%m-%d")).to_csv(state / "prices" / f"{t}.csv", index=False)
    last = df.index[-1]
    (state / "news" / "raw").mkdir(parents=True)
    items = [{"id": f"n{i}", "ticker": "AAPL", "ts": int((last - pd.Timedelta(days=i % 30) + pd.Timedelta(hours=14)).tz_localize("America/New_York").timestamp()),
              "headline": f"Apple beats estimates {i}" if i % 2 else f"Chip export curbs widen {i}", "summary": "", "source": "x", "url": "", "related": ""}
             for i in range(80)]
    (state / "news" / "raw" / "x.jsonl").write_text("\n".join(json.dumps(i) for i in items))
    (state / "news" / "fetched.json").write_text(json.dumps({"coverage_start": (last - pd.Timedelta(days=60)).strftime("%Y-%m-%d")}))


def test_end_to_end_offline(tmp_path):
    state, site = tmp_path / "state", tmp_path / "site"
    write_state(state)
    s = cli.run(state, site, network=False, scorer=FakeScorer(), cfg=TINY, backtest_kw={"start": "2016-06-01", "every": 126, "steps": 10})
    assert s["steps"]["train"]["status"] == "ok", s["steps"]
    assert s["steps"]["predict"]["status"] == "ok"
    assert s["steps"]["news_score"]["scored"] > 0
    assert s["steps"]["backtest"]["status"] == "ok", s["steps"]["backtest"]
    led = pd.read_csv(state / "ledger" / "predictions.csv")
    assert set(led["model"]) == {"ssm", "zero", "drift", "momentum", "ridge"} and len(led) == 10
    html = (site / "index.html").read_text()
    assert "Forward test" in html and "Backtest" in html and "Chip export curbs" in html
    assert (site / "data" / "summary.json").exists() and (site / "method.html").exists()
    # a second run the same evening keeps the model and doesn't add or change predictions
    s2 = cli.run(state, site, network=False, cfg=TINY)
    assert s2["steps"]["train"]["status"] == "skipped" and s2["steps"]["predict"]["status"] == "skipped"
    assert pd.read_csv(state / "ledger" / "predictions.csv").equals(led)


def test_ssm_learns_a_planted_lead_lag():
    """XLK follows the semiconductor ETF's last five days: the SSM should beat 'no change'."""
    from exp2 import features
    p = prices(2600, seed=3)
    rng = np.random.default_rng(3)
    smh = np.log(p["SMH"]["adj_close"]).diff().fillna(0).to_numpy()
    lead = pd.Series(smh).rolling(5).mean().shift(1).fillna(0).to_numpy()
    r = 0.9 * lead * 5 + rng.normal(0, 0.003, len(smh))
    px = 50 * np.exp(np.cumsum(r))
    p["XLK"] = p["XLK"].assign(adj_close=px, close=px)
    df = features.build(p, None, TINY)
    cfg = Config(model=ModelConfig(d_model=16, n_modes=16, steps=250, seeds=(0,), window=256, burn_in=64, val_sessions=200,
                                   finetune_steps=40, patience=100))
    f = train.fit(df, len(df) - 1, cfg)
    v = f.info["val_mse"]["y_sector"]
    assert v["ssm"] < 0.8 * v["zero"], v
