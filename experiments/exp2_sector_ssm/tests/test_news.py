from datetime import datetime
from zoneinfo import ZoneInfo

import pandas as pd

from exp2 import news
from exp2.news import StoryScore

NY = ZoneInfo("America/New_York")
SESS = pd.DatetimeIndex(pd.to_datetime(["2026-10-08", "2026-10-09", "2026-10-12", "2026-10-13"]))


def ts(s):
    return int(datetime.fromisoformat(s).replace(tzinfo=NY).timestamp())


def test_stories_count_for_the_next_close():
    t = pd.Series([ts("2026-10-08T15:59"), ts("2026-10-08T16:01"), ts("2026-10-10T12:00"), ts("2026-10-13T17:00")])
    got = news.assign_sessions(t, SESS)
    assert list(got[:3].dt.strftime("%Y-%m-%d")) == ["2026-10-08", "2026-10-09", "2026-10-12"]
    assert pd.isna(got[3])  # after the last known close: belongs to a session that hasn't closed


def test_screen_drops_roundups_and_duplicates():
    df = pd.DataFrame({"id": ["a", "b", "c"], "ticker": ["AAPL"] * 3, "ts": [1, 2, 3],
                       "headline": ["Apple beats on iPhone sales", "Apple beats on iPhone sales!", "Top 10 stocks to buy now"]})
    assert list(news.screen(df)["id"]) == ["a"]


def test_cap_spreads_across_companies():
    df = pd.DataFrame({"id": list("abcdef"), "ticker": ["AAPL"] * 4 + ["MSFT"] * 2, "ts": range(6),
                       "session": [SESS[0]] * 6})
    kept = news.cap_per_session(df, 4)
    assert sorted(kept["ticker"]) == ["AAPL", "AAPL", "MSFT", "MSFT"]


class FakeScorer:
    model = "fake"

    def __init__(self):
        self.calls = 0

    def score(self, stories):
        self.calls += 1
        return [StoryScore(i=k, relevance=0.5, sentiment=0.2 if "beats" in s["headline"] else -0.4)
                for k, s in enumerate(stories, 1)], 0.01


def test_scoring_is_cached_and_budgeted(tmp_path):
    df = pd.DataFrame({"id": [f"s{i}" for i in range(60)], "ticker": "AAPL", "ts": range(60), "headline": "Apple beats",
                       "summary": "", "session": SESS[0]})
    sc = FakeScorer()
    r = news.score_pending(tmp_path, df, sc, budget_usd=0.02, batch=25)
    assert r["scored"] == 50 and r["budget_stop"]  # stops once the budget is reached
    r2 = news.score_pending(tmp_path, df.head(50), sc, budget_usd=1.0, batch=25)
    assert r2["scored"] == 0 and sc.calls == 2  # nothing re-scored


def test_session_features_mark_missing_coverage():
    sess = pd.bdate_range("2026-09-01", "2026-10-13")
    stories = pd.DataFrame({"id": ["a", "b"], "ticker": ["AAPL", "MSFT"], "ts": [0, 0],
                            "session": [sess[-1], sess[-1]]})
    scores = {"a": {"relevance": 1.0, "sentiment": 0.5}, "b": {"relevance": 0.2, "sentiment": -0.5}}
    f = news.session_features(stories, scores, sess, "2026-09-20")
    assert f.loc[: "2026-09-26", "news_avail"].eq(0).all() and f.loc["2026-09-28":, "news_avail"].eq(1).all()
    last = f.iloc[-1]
    assert last["news_tone"] > 0 and abs(last["news_count"] - 1.0986) < 1e-3
    assert news.session_features(stories, scores, sess, None)["news_avail"].eq(0).all()
