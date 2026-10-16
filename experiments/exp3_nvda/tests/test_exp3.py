import json
from dataclasses import replace

import numpy as np
import pandas as pd
import torch

from exp3 import cli, collect, config, features, model, score
from exp3.ekf import EKFSSM

CFG = replace(config.DEFAULT, steps=20, window=64, seeds=(0,), val_sessions=60, patience=20)


def _prices(data, start="2016-01-04", end="2025-12-31", seed=0):
    rng = np.random.default_rng(seed)
    days = pd.bdate_range(start, end)
    d = data / "prices"
    d.mkdir(parents=True, exist_ok=True)
    for t in ["NVDA", "QQQ", "SPY"]:
        r = rng.normal(0.0005, 0.02, len(days))
        p = 50 * np.exp(np.cumsum(r))
        split = np.ones(len(days))
        close = p.copy()
        if t == "NVDA":   # a 4:1 split in July 2021: raw closes drop, adjusted closes don't
            i = days.searchsorted(pd.Timestamp("2021-07-20"))
            split[i] = 4.0
            close[:i] = p[:i] * 4
        pd.DataFrame({"date": days.strftime("%Y-%m-%d"), "close": close, "adj_close": p, "adj_volume": 1e6, "split_factor": split}).to_csv(d / f"{t}.csv", index=False)
    pd.DataFrame({"end": ["2015-12-31", "2021-06-30", "2021-10-31"], "filed": ["2016-02-01", "2021-08-20", "2021-11-20"],
                  "shares": [5e8, 6.2e8, 2.5e9], "form": "10-Q"}).to_csv(data / "shares.csv", index=False)
    return days


def test_market_cap_is_continuous_through_a_split(tmp_path):
    _prices(tmp_path)
    px = cli.load_prices(tmp_path)["NVDA"]
    mc = features.market_cap(px, pd.read_csv(tmp_path / "shares.csv"))
    j = mc.loc["2021-07-15":"2021-07-23"]
    assert (j.pct_change().dropna().abs() < 0.2).all()       # no 4x jump on the split day


def test_news_counts_from_the_close_after_its_date():
    sessions = pd.DatetimeIndex(pd.bdate_range("2024-03-04", "2024-03-08"))
    ts = pd.Series(["2024-03-05T08:00:00+00:00", "2024-03-05T19:00:00+00:00"])      # 03:00 and 14:00 New York
    s = features.assign_sessions(features.news_available(ts), sessions)
    assert list(s) == [pd.Timestamp("2024-03-06")] * 2
    # a Reddit post at 14:00 New York counts for that day's close
    assert features.assign_sessions(pd.Series(["2024-03-05T19:00:00+00:00"]), sessions)[0] == pd.Timestamp("2024-03-05")


def test_broker_scores():
    up = {"to_grade": "Buy", "action": "up", "pt": 200, "pt_prior": 150, "pt_action": "Raises"}
    down = {"to_grade": "Sell", "action": "down", "pt": 100, "pt_prior": 150, "pt_action": "Lowers"}
    assert features.broker_score(up) == 1.0 and features.broker_score(down) == -1.0
    assert 0 < features.broker_score({"to_grade": "Overweight", "action": "reit", "pt": None, "pt_prior": None, "pt_action": ""}) < 1
    # empty CSV cells come back as NaN
    assert features.broker_score({"to_grade": float("nan"), "action": "main", "pt": float("nan"), "pt_prior": float("nan"), "pt_action": float("nan")}) == 0.0


def test_unobserved_weeks_are_nan_and_long_memory_persists():
    sessions = pd.DatetimeIndex(pd.bdate_range("2024-01-01", periods=30))
    items = pd.DataFrame({"session": [sessions[2]], "cat": ["news"], "v": [0.8], "w": [1.0]})
    f = features.sentiment_features(items, sessions, week=5, halflife=63)
    assert np.isnan(f["week_news"].iloc[1]) and f["week_news"].iloc[2] == 0.8 and np.isnan(f["week_news"].iloc[10])
    assert abs(f["long_news"].iloc[20] - 0.8) < 1e-9 and np.isnan(f["long_news"].iloc[1])
    assert np.isnan(f["week_speculation"]).all()


def test_news_windows_split_when_capped(tmp_path):
    calls = []

    def fetch(a, b):
        calls.append((a, b))
        n = 120 if a == "2024-01-01" and b == "2024-01-08" else 3
        return [{"id": f"gn{a}{b}{k}", "ts": f"{a}T12:00:00+00:00", "title": f"t{a}{b}{k}", "source": "x"} for k in range(n)]

    st = collect.news(tmp_path, start="2024-01-01", end="2024-01-07", fetch=fetch, pause=0)
    assert calls[0] == ("2024-01-01", "2024-01-08") and len(calls) == 3 and st["items"] == 6
    collect.news(tmp_path, start="2024-01-01", end="2024-01-07", fetch=fetch, pause=0)
    assert len(calls) == 3        # resumable: nothing fetched twice


def test_reddit_pages_and_resumes(tmp_path):
    def fetch(sub, term, after, before):
        base = pd.Timestamp("2024-01-02").timestamp()
        allp = [{"id": f"{sub}{term}{k}", "created_utc": base + k, "title": f"{term} post {k}", "selftext": "", "score": 1, "num_comments": 0} for k in range(150)]
        return [p for p in allp if after <= p["created_utc"] < before][:100]

    st = collect.reddit(tmp_path, start="2024-01-01", end="2024-01-31", subs=["wsb"], terms=["NVDA"], fetch=fetch, pause=0)
    assert st["items"] == 150
    assert len(collect.load_jsonl(tmp_path / "reddit.jsonl")) == 150


class FakeScorer:
    model = "fake"

    def score(self, items):
        out = []
        for k, it in enumerate(items, 1):
            v = 0.6 if "up" in it["title"] else -0.6
            out.append(score.ItemScore(i=k, relevance=0.9, sentiment=v, saturation=v / 2))
        return out, 0.001


def test_end_to_end_with_planted_sentiment(tmp_path):
    days = _prices(tmp_path)
    px = pd.read_csv(tmp_path / "prices" / "NVDA.csv")
    lp = np.log(px["adj_close"].to_numpy())
    fwd = np.r_[lp[5:] - lp[:-5], np.zeros(5)]
    news, reddit = [], []
    for i, d in enumerate(days[300:], 300):
        if i % 2:
            continue
        word = "up" if fwd[i] > 0 else "down"     # planted: sentiment foretells the next week
        news.append({"id": f"gn{i}", "ts": (d - pd.Timedelta(days=1)).strftime("%Y-%m-%dT12:00:00+00:00"), "title": f"nvidia {word} {i}", "source": "x"})
        reddit.append({"id": f"rd{i}", "ts": d.strftime("%Y-%m-%dT15:00:00+00:00"), "sub": "wsb", "title": f"NVDA {word} {i}", "text": "", "score": 1, "comments": 0})
    with open(tmp_path / "news.jsonl", "w") as f:
        f.writelines(json.dumps(r) + "\n" for r in news)
    with open(tmp_path / "reddit.jsonl", "w") as f:
        f.writelines(json.dumps(r) + "\n" for r in reddit)
    pd.DataFrame({"ts": ["2019-05-01T12:00:00+00:00"], "firm": "X", "action": "up", "from_grade": "Hold", "to_grade": "Buy", "pt_action": "Raises",
                  "pt": 10, "pt_prior": 8}).to_csv(tmp_path / "analysts.csv", index=False)
    s = cli.run(tmp_path, tmp_path / "out", do_collect=False, cfg=CFG, scorer=FakeScorer())
    assert s["steps"]["score"]["scored"] > 0 and s["steps"]["model"]["status"] == "ok", s["steps"]
    r = s["results"]
    assert set(r["test"]) == {*model.VARIANTS, "ridge", "ridge_price", "momentum", "drift", "zero"}
    # the planted signal is found by the models that see sentiment, and not by those that don't
    assert r["test"]["ridge"]["skill"] > 0.2 > r["test"]["ridge_price"]["skill"]
    assert r["test"]["ekf"]["ic"] > 0.3 and r["test"]["ekf_res"]["ic"] > 0.1
    f = s["model"]["ekf"]["ekf_res"]["fading"][0]
    assert len(f["energy"]) == CFG.d_state // 2 and all(0 < x <= 1.05 for x in f["fade_per_session"])
    assert r["sentiment_adds"]["ridge_vs_ridge_price"]["t"] < -2
    assert (tmp_path / "out" / "index.html").exists()
    p = pd.read_csv(tmp_path / "out" / "predictions.csv")
    assert p[p["period"] == "test"]["date"].min() >= "2022-01-01" and p["date"].max() <= "2025-12-31"


def test_resonance_gain_reduces_to_h2_for_white_input():
    m = EKFSSM(3, 1, d_state=6, norm="resonance", radius="exp", r_max=0.98)   # sharper poles alias on a finite grid
    m.spectrum = torch.ones(512)
    r = m.r().detach()
    assert torch.allclose(m.resonance().detach(), 1 / (1 - r ** 2), rtol=2e-2)
    h = EKFSSM(3, 1, d_state=6, norm="h2", radius="exp", r_max=0.98)
    h.spectrum = torch.ones(512)
    h.load_state_dict(m.state_dict())
    assert torch.allclose(m.gain().detach(), h.gain().detach(), rtol=1e-2)


def test_resonance_gain_tames_a_resonant_mode():
    """A mode tuned to the input's dominant frequency gets a smaller input gain than under the flat (H2) rule."""
    m = EKFSSM(3, 1, d_state=4, norm="resonance", radius="exp", r_min=0.95, r_max=0.95)
    with torch.no_grad():
        m.theta.copy_(torch.tensor([2 * torch.pi * 8 / 64, 2 * torch.pi * 25 / 64]))
    S = torch.full((64,), 0.1)
    S[8] = S[64 - 8] = 10.0
    m.spectrum = S / S.mean()
    g = m.gain().detach()[::2]
    assert g[0] < g[1]
