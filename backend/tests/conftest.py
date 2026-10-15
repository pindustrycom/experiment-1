"""Shared fixtures. Database tests run against a dedicated PostgreSQL database (TEST_DATABASE_URL,
default postgresql+psycopg://sm:sm@localhost/sm_test) and are skipped when it is unreachable.
Never point TEST_DATABASE_URL at a database whose data you want to keep: tables are truncated."""
from __future__ import annotations

import os

import pytest

TEST_DB = os.environ.get("TEST_DATABASE_URL", "postgresql+psycopg://sm:sm@localhost/sm_test")
os.environ["DATABASE_URL"] = TEST_DB
os.environ.setdefault("APP_MODE", "production")
os.environ.setdefault("ENVIRONMENT", "test")
os.environ.setdefault("SECRET_KEY", "test-secret-key-that-is-long-enough-0123456789")
os.environ.setdefault("COOKIE_SECURE", "false")
os.environ.setdefault("LOG_LEVEL", "WARNING")
os.environ["ANTHROPIC_API_KEY"] = ""
os.environ["TIINGO_API_KEY"] = ""
os.environ["FINNHUB_API_KEY"] = ""
os.environ["ALPHAVANTAGE_API_KEY"] = ""
os.environ["ENABLE_STOOQ"] = "false"
os.environ["SEC_USER_AGENT"] = ""

ADMIN = {"email": "admin@test.example", "password": "Correct-Horse-1234"}
VIEWER = {"email": "viewer@test.example", "password": "Viewer-Pass-12345"}
CSRF = {"X-Requested-With": "splittability"}


def _db_available() -> bool:
    from sqlalchemy import create_engine, text
    try:
        eng = create_engine(TEST_DB, connect_args={"connect_timeout": 3})
        with eng.connect() as c:
            c.execute(text("select 1"))
        eng.dispose()
        return True
    except Exception:
        return False


DB_AVAILABLE = _db_available()
requires_db = pytest.mark.skipif(not DB_AVAILABLE, reason=f"test database not reachable at {TEST_DB}")


def _alembic_config():
    from alembic.config import Config
    here = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    cfg = Config(os.path.join(here, "alembic.ini"))
    cfg.set_main_option("script_location", os.path.join(here, "migrations"))
    cfg.set_main_option("sqlalchemy.url", TEST_DB.replace("%", "%%"))
    return cfg


@pytest.fixture(scope="session")
def migrated():
    if not DB_AVAILABLE:
        pytest.skip("test database not reachable")
    from alembic import command
    from sqlalchemy import create_engine, text
    eng = create_engine(TEST_DB)
    with eng.begin() as c:
        c.execute(text("DROP SCHEMA public CASCADE; CREATE SCHEMA public;"))
    eng.dispose()
    command.upgrade(_alembic_config(), "head")
    from app.core.db import reset_engine
    reset_engine(TEST_DB)
    yield
    from app.core.db import get_engine
    get_engine().dispose()


@pytest.fixture()
def db(migrated):
    """A clean database with reference data (companies, initial pairs, model versions)."""
    from sqlalchemy import text

    from app.api.ratelimit import limiter
    from app.core.db import Base, get_engine, session_factory
    from app.services import backtest_service, jobs
    from app.services.reference import seed_reference_data
    tables = ", ".join(f'"{t.name}"' for t in Base.metadata.sorted_tables)
    with get_engine().begin() as c:
        c.execute(text(f"TRUNCATE {tables} RESTART IDENTITY CASCADE"))
    limiter.reset()
    s = session_factory()()
    seed_reference_data(s)  # same bootstrap as `python -m app.cli init --no-demo`
    backtest_service.seed_reported_benchmark(s)
    jobs.ensure_scheduled_jobs(s, False)
    s.commit()
    yield s
    s.close()


@pytest.fixture()
def app(db):
    from app.main import create_app
    return create_app()


def _make_user(db, email, password, role):
    from app.core.security import hash_password
    from app.models import User
    u = User(email=email, password_hash=hash_password(password), role=role)
    db.add(u)
    db.commit()
    return u


@pytest.fixture()
def client(app):
    from fastapi.testclient import TestClient
    return TestClient(app, base_url="http://testserver")


@pytest.fixture()
def admin_client(db, app):
    from fastapi.testclient import TestClient
    _make_user(db, ADMIN["email"], ADMIN["password"], "admin")
    c = TestClient(app, base_url="http://testserver")
    r = c.post("/api/v1/auth/login", json=ADMIN, headers=CSRF)
    assert r.status_code == 200, r.text
    return c


@pytest.fixture()
def viewer_client(db, app):
    from fastapi.testclient import TestClient
    _make_user(db, VIEWER["email"], VIEWER["password"], "viewer")
    c = TestClient(app, base_url="http://testserver")
    r = c.post("/api/v1/auth/login", json=VIEWER, headers=CSRF)
    assert r.status_code == 200, r.text
    return c
