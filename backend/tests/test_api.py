"""API integration tests against a real PostgreSQL database (production mode, no providers configured)."""
from __future__ import annotations

from datetime import UTC, date, datetime, timedelta

import pytest

from tests.conftest import ADMIN, CSRF, requires_db

pytestmark = requires_db
V1 = "/api/v1"


def test_health_is_public(client):
    r = client.get(f"{V1}/health")
    assert r.status_code == 200
    assert r.json()["mode"] == "production"


def test_openapi_documents_v1_only(client):
    spec = client.get("/api/openapi.json").json()
    paths = spec["paths"]
    assert f"{V1}/pairs" in paths and f"{V1}/jobs/run" in paths
    assert "/api/pairs" not in paths  # alias is excluded from the schema


def test_requires_auth(client):
    r = client.get(f"{V1}/pairs")
    assert r.status_code == 401
    assert r.json()["error"]["code"] == "unauthorized"


def test_login_wrong_password_and_rate_limit(client, admin_client):
    for _ in range(3):
        r = client.post(f"{V1}/auth/login", json={"email": ADMIN["email"], "password": "wrong-password-123"}, headers=CSRF)
        assert r.status_code == 401
    codes = [client.post(f"{V1}/auth/login", json={"email": ADMIN["email"], "password": "nope-nope-nope"}, headers=CSRF).status_code for _ in range(15)]
    assert 429 in codes


def test_session_cookie_flags(db, app):
    from fastapi.testclient import TestClient

    from tests.conftest import _make_user
    _make_user(db, "c@test.example", "Cookie-Pass-12345", "admin")
    r = TestClient(app).post(f"{V1}/auth/login", json={"email": "c@test.example", "password": "Cookie-Pass-12345"}, headers=CSRF)
    sc = r.headers["set-cookie"].lower()
    assert "httponly" in sc and "samesite=strict" in sc


def test_me_and_alias(admin_client):
    assert admin_client.get(f"{V1}/auth/me").json()["role"] == "admin"
    assert admin_client.get("/api/auth/me").status_code == 200


def test_csrf_header_required_for_mutations(admin_client):
    r = admin_client.post(f"{V1}/pairs", json={"ticker_a": "AAPL", "ticker_b": "MSFT"})
    assert r.status_code == 403
    assert "CSRF" in r.json()["error"]["message"]


def test_initial_pairs_are_seeded_and_deduplicated(admin_client):
    r = admin_client.get(f"{V1}/pairs?status=all&page_size=100")
    assert r.status_code == 200
    slugs = {p["slug"] for p in r.json()["items"]}
    assert len(slugs) == 11  # 12 requested pairs; LOW/HD duplicates HD/LOW
    assert "KO-PEP" in slugs
    assert sum(1 for s in slugs if set(s.split("-")) == {"HD", "LOW"}) == 1


def test_scores_unavailable_without_data_not_zero(admin_client):
    p = admin_client.get(f"{V1}/pairs/KO-PEP").json()
    assert p["si_v2"] is None or p["si_v2"]["value"] is None
    assert p["is_demo"] is False


def test_pair_crud_and_archive_keeps_history(admin_client, db):
    r = admin_client.post(f"{V1}/pairs", json={"ticker_a": "aapl", "ticker_b": "MSFT", "industry": "Technology"}, headers=CSRF)
    assert r.status_code == 201, r.text
    slug = r.json()["slug"]
    assert slug == "AAPL-MSFT"
    dup = admin_client.post(f"{V1}/pairs", json={"ticker_a": "MSFT", "ticker_b": "AAPL"}, headers=CSRF)
    assert dup.status_code == 409
    same = admin_client.post(f"{V1}/pairs", json={"ticker_a": "MSFT", "ticker_b": "MSFT"}, headers=CSRF)
    assert same.status_code == 422
    bad = admin_client.post(f"{V1}/pairs", json={"ticker_a": "MS FT;", "ticker_b": "AAPL"}, headers=CSRF)
    assert bad.status_code == 422

    r = admin_client.patch(f"{V1}/pairs/{slug}", json={"alert_threshold": 65, "monitoring_interval_minutes": 30}, headers=CSRF)
    assert r.status_code == 200 and r.json()["alert_threshold"] == 65
    assert admin_client.patch(f"{V1}/pairs/{slug}", json={"monitoring_interval_minutes": 5}, headers=CSRF).status_code == 422

    # a pair with a stored score cannot be hard-deleted, only archived
    from sqlalchemy import select

    from app.models import IndexScore, ModelVersion, Pair
    pair = db.scalar(select(Pair).where(Pair.slug == slug))
    mv = db.scalar(select(ModelVersion).where(ModelVersion.index_name == "si_v1"))
    db.add(IndexScore(pair_id=pair.id, model_version_id=mv.id, index_name="si_v1", as_of=datetime.now(UTC), value=None,
                      status="unavailable", coverage=0, direction="not_applicable", explanation="t", trigger="test", inputs_digest="x"))
    db.commit()
    assert admin_client.delete(f"{V1}/pairs/{slug}?hard=true", headers=CSRF).status_code == 409
    r = admin_client.delete(f"{V1}/pairs/{slug}", headers=CSRF)
    assert r.status_code == 200
    got = admin_client.get(f"{V1}/pairs/{slug}").json()
    assert got["status"] == "archived" and got["archived_at"]
    assert slug not in {p["slug"] for p in admin_client.get(f"{V1}/pairs").json()["items"]}
    assert slug in {p["slug"] for p in admin_client.get(f"{V1}/pairs?status=archived").json()["items"]}
    assert len(admin_client.get(f"{V1}/pairs/{slug}/scores").json()) == 1
    # restore
    assert admin_client.patch(f"{V1}/pairs/{slug}", json={"status": "active"}, headers=CSRF).json()["archived_at"] is None


def test_pair_lookup_by_uid_and_reverse_slug(admin_client):
    p = admin_client.get(f"{V1}/pairs/KO-PEP").json()
    assert admin_client.get(f"{V1}/pairs/{p['uid']}").json()["slug"] == "KO-PEP"
    assert admin_client.get(f"{V1}/pairs/PEP-KO").json()["slug"] == "KO-PEP"
    assert admin_client.get(f"{V1}/pairs/NOPE-ZZZ").status_code == 404


def test_viewer_is_read_only(viewer_client):
    assert viewer_client.get(f"{V1}/pairs").status_code == 200
    assert viewer_client.post(f"{V1}/pairs", json={"ticker_a": "AAPL", "ticker_b": "MSFT"}, headers=CSRF).status_code == 403
    assert viewer_client.post(f"{V1}/jobs/pause", headers=CSRF).status_code == 403
    assert viewer_client.put(f"{V1}/settings", json={"section": "budget", "value": {"monthly_ai_limit_usd": 1}}, headers=CSRF).status_code == 403


def test_settings_never_expose_secrets(admin_client, monkeypatch):
    from pydantic import SecretStr

    from app.core.config import get_settings
    monkeypatch.setattr(get_settings(), "anthropic_api_key", SecretStr("sk-ant-should-never-leak"))
    body = admin_client.get(f"{V1}/settings").text
    assert "sk-ant-should-never-leak" not in body
    assert admin_client.get(f"{V1}/settings").json()["environment"]["providers_configured"]["anthropic"] is True


def test_settings_validation_and_weight_versioning(admin_client):
    before = admin_client.get(f"{V1}/settings/model-versions").json()
    r = admin_client.put(f"{V1}/settings", json={"section": "scoring", "value": {"si_v2_weights": {"N": 30, "F": 30, "M": 20, "P": 10, "S": 10}}}, headers=CSRF)
    assert r.status_code == 200, r.text
    after = admin_client.get(f"{V1}/settings/model-versions").json()
    v2 = [m for m in after if m["index_name"] == "si_v2"]
    assert len(after) == len(before) + 1
    assert sum(1 for m in v2 if m["is_active"]) == 1
    assert next(m for m in v2 if m["is_active"])["weights"]["F"] == pytest.approx(0.30)
    bad = admin_client.put(f"{V1}/settings", json={"section": "scoring", "value": {"si_v2_weights": {"N": -1, "F": 0, "M": 0, "P": 0, "S": 0}}}, headers=CSRF)
    assert bad.status_code == 422
    assert admin_client.put(f"{V1}/settings", json={"section": "llm", "value": {"effort": "turbo"}}, headers=CSRF).status_code == 422
    assert admin_client.put(f"{V1}/settings", json={"section": "news", "value": {"rss_templates": ["javascript:alert(1)"]}}, headers=CSRF).status_code == 422
    assert admin_client.put(f"{V1}/settings", json={"section": "nope", "value": {}}, headers=CSRF).status_code == 422


def test_jobs_queue_pause_resume_and_duplicates(admin_client):
    jobs = admin_client.get(f"{V1}/jobs").json()
    names = {j["name"] for j in jobs}
    assert {"ingest_news", "ingest_prices", "compute_scores", "snapshot_predictions", "evaluate_outcomes", "backup_database"} <= names
    assert "demo_refresh" not in names  # production mode
    r = admin_client.post(f"{V1}/jobs/run", json={"job": "compute_scores"}, headers=CSRF)
    assert r.status_code == 202
    assert admin_client.post(f"{V1}/jobs/run", json={"job": "compute_scores"}, headers=CSRF).status_code == 409
    assert admin_client.post(f"{V1}/jobs/run", json={"job": "rm_rf"}, headers=CSRF).status_code in (404, 422)
    assert admin_client.post(f"{V1}/jobs/pause", headers=CSRF).status_code == 200
    assert admin_client.get(f"{V1}/system/status").json()["bot"]["paused"] is True
    assert admin_client.post(f"{V1}/jobs/resume", headers=CSRF).status_code == 200
    runs = admin_client.get(f"{V1}/jobs/runs?job=compute_scores").json()
    assert runs["total"] == 1 and runs["items"][0]["status"] == "queued"


def test_schedule_update_validates_cron(admin_client):
    ok = admin_client.patch(f"{V1}/jobs/schedules/ingest_news", json={"cron": "*/30 * * * *"}, headers=CSRF)
    assert ok.status_code == 200 and ok.json()["cron"] == "*/30 * * * *"
    assert ok.json()["next_run"]
    assert admin_client.patch(f"{V1}/jobs/schedules/ingest_news", json={"cron": "every hour"}, headers=CSRF).status_code == 422


def test_backtest_param_validation(admin_client):
    assert admin_client.post(f"{V1}/research/backtests", json={"frequency": "daily"}, headers=CSRF).status_code == 422
    assert admin_client.post(f"{V1}/research/backtests", json={"horizon": "abc"}, headers=CSRF).status_code == 422
    assert admin_client.post(f"{V1}/research/backtests", json={"evil": 1}, headers=CSRF).status_code == 422
    assert admin_client.post(f"{V1}/research/backtests", json={"horizon": 252, "threshold": 0.15, "frequency": "quarterly"}, headers=CSRF).status_code == 202


def test_production_has_no_demo_data_and_reports_freshness(admin_client):
    s = admin_client.get(f"{V1}/system/status").json()
    assert s["mode"] == "production"
    assert s["last_price_update"] is None and s["last_news_update"] is None
    assert s["providers"]["anthropic"] is False
    assert admin_client.get(f"{V1}/news").json()["total"] == 0


def test_performance_and_exports_empty_state(admin_client):
    perf = admin_client.get(f"{V1}/performance?n_boot=100").json()
    assert perf["live"]["contains_simulated"] is False
    assert any(b["kind"] == "reported_historical" for b in perf["backtests"])
    r = admin_client.get(f"{V1}/research/export/predictions.csv")
    assert r.status_code == 200 and r.headers["content-type"].startswith("text/csv")
    assert admin_client.get(f"{V1}/research/export/secrets.csv").status_code == 404
    pdf = admin_client.get(f"{V1}/research/report.pdf")
    assert pdf.status_code == 200 and pdf.content[:4] == b"%PDF"


def test_watchlists_crud(admin_client):
    pairs = admin_client.get(f"{V1}/pairs").json()["items"]
    r = admin_client.post(f"{V1}/watchlists", json={"name": "Staples", "pair_ids": [pairs[0]["id"]]}, headers=CSRF)
    assert r.status_code == 201
    wid = r.json()["id"]
    assert admin_client.post(f"{V1}/watchlists", json={"name": "Staples"}, headers=CSRF).status_code == 409
    r = admin_client.put(f"{V1}/watchlists/{wid}", json={"name": "Staples 2", "pair_ids": [p["id"] for p in pairs[:3]]}, headers=CSRF)
    assert len(r.json()["pair_ids"]) == 3
    filtered = admin_client.get(f"{V1}/pairs?watchlist_id={wid}").json()["items"]
    assert len(filtered) == 3
    assert admin_client.delete(f"{V1}/watchlists/{wid}", headers=CSRF).status_code == 204


def test_change_password_and_users(admin_client):
    r = admin_client.post(f"{V1}/auth/change-password", json={"current_password": "wrong", "new_password": "Another-Pass-9876"}, headers=CSRF)
    assert r.status_code in (400, 401, 403)
    assert admin_client.post(f"{V1}/auth/change-password", json={"current_password": ADMIN["password"], "new_password": "short"}, headers=CSRF).status_code == 422
    assert admin_client.post(f"{V1}/auth/change-password", json={"current_password": ADMIN["password"], "new_password": "Another-Pass-9876"}, headers=CSRF).status_code == 204
    r = admin_client.post(f"{V1}/auth/users", json={"email": "new@test.example", "password": "New-User-Pass-123", "role": "viewer"}, headers=CSRF)
    assert r.status_code == 201
    assert "password" not in r.text.lower() or "password_hash" not in r.text


def test_structured_validation_errors(admin_client):
    r = admin_client.get(f"{V1}/pairs?page=0")
    assert r.status_code == 422
    body = r.json()["error"]
    assert body["code"] == "validation_error" and isinstance(body["details"], list)


def test_predictions_are_immutable(db):
    """Database triggers reject UPDATE/DELETE on prediction snapshots, even from the application role."""
    from sqlalchemy import select, text
    from sqlalchemy.exc import DBAPIError

    from app.models import Pair, PredictionSnapshot
    pair = db.scalar(select(Pair).where(Pair.slug == "KO-PEP"))
    snap = PredictionSnapshot(pair_id=pair.id, prediction_date=date(2026, 1, 5), data_cutoff=datetime(2026, 1, 5, 22, tzinfo=UTC),
                              horizon_days=30, target_version="div1-H30-T0.0488", target_definition={"threshold": 0.0488},
                              si_v1=40.0, si_v2=None, si_v2_status="unavailable", components={}, baselines={}, direction="not_applicable",
                              market_info={}, base_date=date(2026, 1, 5), base_price_a=60.0, base_price_b=170.0,
                              price_provider="test", is_total_return=True)
    db.add(snap)
    db.commit()
    with pytest.raises(DBAPIError):
        db.execute(text("UPDATE prediction_snapshots SET si_v1 = 99 WHERE id = :i"), {"i": snap.id})
        db.commit()
    db.rollback()
    with pytest.raises(DBAPIError):
        db.execute(text("DELETE FROM prediction_snapshots WHERE id = :i"), {"i": snap.id})
        db.commit()
    db.rollback()


def test_api_rate_limit(admin_client, monkeypatch):
    from app.core.config import get_settings
    monkeypatch.setattr(get_settings(), "api_rate_limit_per_minute", 5)
    codes = [admin_client.get(f"{V1}/pairs").status_code for _ in range(8)]
    assert codes.count(429) >= 2
    assert admin_client.get(f"{V1}/health").status_code == 200  # health is never rate limited


def test_news_filters_validate(admin_client):
    assert admin_client.get(f"{V1}/news?ticker=WAYTOOLONGTICKER").status_code == 422
    assert admin_client.get(f"{V1}/news?material_only=true&changed_only=true&since={(datetime.now(UTC) - timedelta(days=3)).isoformat().replace('+00:00', 'Z')}").status_code == 200


def test_production_flags_leftover_demo_data(admin_client, db):
    from app.models import NewsArticle
    assert admin_client.get(f"{V1}/system/status").json()["demo_data_present"] is False
    db.add(NewsArticle(cluster_key="demo-1", headline="[SIMULATED] x", first_published_at=datetime.now(UTC), tickers=["KO"], is_demo=True))
    db.commit()
    assert admin_client.get(f"{V1}/system/status").json()["demo_data_present"] is True
