"""The GitHub Pages dashboard: builds from a real (empty) production database and refuses demo data."""
from datetime import UTC, datetime

import pytest

from tests.conftest import requires_db

pytestmark = requires_db


def test_builds_every_page_with_honest_empty_states(db, tmp_path):
    from app.static_site import build
    r = build(tmp_path / "site", allow_demo=False)
    assert r["pairs"] == 10 and r["demo"] is False
    index = (tmp_path / "site" / "index.html").read_text()
    assert "KO / PEP" in index and "No price data has been loaded yet" in index
    assert 'class="score na"' in index  # unavailable shown as n/a, never as a number
    for f in ["predictions.html", "performance.html", "about.html", "pairs/KO-PEP.html", "data/predictions.csv", ".nojekyll"]:
        assert (tmp_path / "site" / f).exists(), f
    assert "<script" not in index
    assert "0.6655" in (tmp_path / "site" / "performance.html").read_text()  # reported benchmark shown separately


def test_refuses_to_publish_simulated_data(db, tmp_path):
    from app.models import NewsArticle
    from app.static_site import build
    db.add(NewsArticle(cluster_key="demo-1", headline="[SIMULATED] x", first_published_at=datetime.now(UTC), tickers=["KO"], is_demo=True))
    db.commit()
    with pytest.raises(SystemExit, match="simulated"):
        build(tmp_path / "site", allow_demo=False)
    assert not (tmp_path / "site").exists()


def test_builder_account_cannot_be_reused(db, tmp_path):
    from sqlalchemy import select

    from app.core.security import verify_password
    from app.models import User
    from app.static_site import BUILDER_EMAIL, build
    build(tmp_path / "site", allow_demo=False)
    db.expire_all()
    u = db.scalar(select(User).where(User.email == BUILDER_EMAIL))
    assert u.role == "viewer" and not verify_password(u.password_hash, "")


def test_backfill_rebuilds_weekly_history_and_the_site_draws_it(db, tmp_path):
    from datetime import date

    import numpy as np
    from sqlalchemy import func, select

    from app.data_sources.base import PriceBar
    from app.models import Alert, Company, IndexScore, Pair
    from app.research.market_calendar import trading_days_between
    from app.services import prices_service, scoring_service
    from app.static_site import build

    days = trading_days_between(date(2018, 1, 2), date(2024, 2, 2))
    rng = np.random.default_rng(1)
    common = np.cumsum(rng.normal(0, 0.01, len(days)))
    for i, t in enumerate(["KO", "PEP"]):
        c = db.scalar(select(Company).where(Company.ticker == t))
        px = 50 * np.exp(common + np.cumsum(rng.normal(0, 0.006 * (i + 1), len(days))))
        prices_service.upsert_bars(db, c.id, "tiingo", [PriceBar(d, float(p), float(p), 1e6) for d, p in zip(days, px, strict=True)],
                                   is_total_return=True)
    db.commit()

    r = scoring_service.backfill_history(db, date(2024, 1, 8), end=date(2024, 2, 2))
    assert r["weeks"] == 4 and r["complete"] and r["errors"] == 0
    pair = db.scalar(select(Pair).where(Pair.slug == "KO-PEP"))
    rows = db.scalars(select(IndexScore).where(IndexScore.pair_id == pair.id, IndexScore.index_name == "si_v1")
                      .order_by(IndexScore.as_of)).all()
    assert [s.as_of.date().weekday() for s in rows] == [4, 4, 4, 4]  # last session of each week
    assert all(s.trigger == "backfill" and s.value is not None for s in rows)
    assert db.scalar(select(func.count()).select_from(Alert)) == 0
    again = scoring_service.backfill_history(db, date(2024, 1, 8), end=date(2024, 2, 2))
    assert again["scored"] == 0  # incremental: nothing is scored twice

    build(tmp_path / "site", allow_demo=False)
    assert 'class="spark"' in (tmp_path / "site" / "index.html").read_text()
    page = (tmp_path / "site" / "pairs" / "KO-PEP.html").read_text()
    assert "rebuilt from past prices" in page and "<polyline" in page
