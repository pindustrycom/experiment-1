"""Environment-based configuration.

Secrets (API keys, passwords, signing keys) come only from environment variables and are
never returned by any API endpoint. Runtime-tunable research settings (weights, cadence,
budgets) live in the database (see app.services.settings_service), not here.
"""
from __future__ import annotations

from functools import lru_cache
from typing import Literal

from pydantic import Field, SecretStr, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    # "demo": synthetic, clearly-labelled data is generated and no external calls are required.
    # "production": only real providers are used; failures are reported, never papered over.
    app_mode: Literal["demo", "production"] = "demo"
    environment: Literal["development", "test", "production"] = "development"
    log_level: str = "INFO"

    database_url: str = "postgresql+psycopg://sm:sm@localhost:5432/sm"

    # Auth
    secret_key: SecretStr = SecretStr("dev-insecure-change-me")
    session_ttl_hours: int = 12
    cookie_secure: bool = False
    login_rate_limit_per_minute: int = 10
    api_rate_limit_per_minute: int = 600

    # Optional bootstrap admin (created by `python -m app.cli create-admin` or on first start)
    bootstrap_admin_email: str | None = None
    bootstrap_admin_password: SecretStr | None = None

    # Claude
    anthropic_api_key: SecretStr | None = None
    claude_model: str = "claude-opus-5-5"
    claude_effort: Literal["low", "medium", "high", "xhigh", "max"] = "medium"
    claude_timeout_seconds: float = 120.0

    # Market data / news providers (all optional)
    tiingo_api_key: SecretStr | None = None
    finnhub_api_key: SecretStr | None = None
    alphavantage_api_key: SecretStr | None = None
    sec_user_agent: str | None = Field(
        default=None,
        description="SEC EDGAR requires a descriptive User-Agent with contact e-mail, "
        "e.g. 'SplittabilityMonitor research you@example.com'.",
    )
    enable_stooq: bool = True

    http_timeout_seconds: float = 20.0

    # Worker
    worker_poll_seconds: int = 10
    backup_dir: str = "/backups"
    backup_retention_days: int = 14
    scheduler_timezone: str = "America/New_York"

    cors_origins: list[str] = []

    @field_validator("database_url")
    @classmethod
    def _normalize_db_url(cls, v: str) -> str:
        if v.startswith("postgres://"):
            v = "postgresql+psycopg://" + v[len("postgres://"):]
        elif v.startswith("postgresql://"):
            v = "postgresql+psycopg://" + v[len("postgresql://"):]
        return v

    @property
    def is_demo(self) -> bool:
        return self.app_mode == "demo"

    def secret(self, name: str) -> str | None:
        val = getattr(self, name, None)
        if val is None:
            return None
        s = val.get_secret_value() if isinstance(val, SecretStr) else str(val)
        return s or None

    def validate_for_production(self) -> list[str]:
        problems: list[str] = []
        if self.environment == "production":
            if self.secret("secret_key") in (None, "dev-insecure-change-me") or len(self.secret("secret_key") or "") < 32:
                problems.append("SECRET_KEY must be set to a random string of at least 32 characters")
            if not self.cookie_secure:
                problems.append("COOKIE_SECURE should be true behind HTTPS")
        return problems


@lru_cache
def get_settings() -> Settings:
    return Settings()
