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
