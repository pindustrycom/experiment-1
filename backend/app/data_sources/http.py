"""Shared HTTP layer: timeouts, retries with exponential backoff + jitter, Retry-After support,
per-provider rate limiting, secret redaction and usage accounting."""
from __future__ import annotations

import logging
import random
import re
import threading
import time
from collections.abc import Callable
from contextvars import ContextVar

import httpx

from app.core.config import get_settings
from app.data_sources.base import ProviderError

log = logging.getLogger(__name__)

# Set by the job runner so provider usage rows are attributed to the job that caused them.
current_job_run_id: ContextVar[int | None] = ContextVar("current_job_run_id", default=None)
UsageSink = Callable[..., None]
_usage_sink: UsageSink | None = None

_SECRET_PARAMS = re.compile(r"((?:token|apikey|api_key|key)=)[^&\s]+", re.I)


def redact(s: str) -> str:
    return _SECRET_PARAMS.sub(r"\1***", s)


def set_usage_sink(sink: UsageSink | None) -> None:
    global _usage_sink
    _usage_sink = sink


def record_usage(**kw) -> None:
    if _usage_sink is not None:
        try:
            _usage_sink(job_run_id=current_job_run_id.get(), **kw)
        except Exception:  # accounting must never break ingestion
            log.exception("usage sink failed")


class RateLimiter:
    """Minimum spacing between calls per provider (simple, process-local)."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._next: dict[str, float] = {}

    def wait(self, provider: str, min_interval: float) -> None:
        if min_interval <= 0:
            return
        with self._lock:
            now = time.monotonic()
            t = max(now, self._next.get(provider, 0.0))
            self._next[provider] = t + min_interval
        delay = t - now
        if delay > 0:
            time.sleep(delay)


rate_limiter = RateLimiter()

PROVIDER_MIN_INTERVAL = {"sec": 0.15, "finnhub": 1.1, "alphavantage": 12.5, "tiingo": 0.5, "stooq": 1.0, "rss": 0.5}


def request(
    provider: str,
    method: str,
    url: str,
    *,
    params: dict | None = None,
    headers: dict | None = None,
    operation: str = "request",
    max_attempts: int = 4,
    base_delay: float = 1.0,
    client: httpx.Client | None = None,
    sleep: Callable[[float], None] = time.sleep,
) -> httpx.Response:
    s = get_settings()
    attempt, last_exc = 0, None
    owns = client is None
    client = client or httpx.Client(timeout=s.http_timeout_seconds, follow_redirects=True)
    try:
        while attempt < max_attempts:
            attempt += 1
            rate_limiter.wait(provider, PROVIDER_MIN_INTERVAL.get(provider, 0.0))
            t0 = time.monotonic()
            status = None
            try:
                resp = client.request(method, url, params=params, headers=headers)
                status = resp.status_code
                latency = int((time.monotonic() - t0) * 1000)
                if status == 429 or status >= 500:
                    record_usage(provider=provider, operation=operation, success=False, status_code=status, latency_ms=latency, error=f"HTTP {status}")
                    retry_after = resp.headers.get("Retry-After")
                    delay = float(retry_after) if retry_after and retry_after.isdigit() else base_delay * 2 ** (attempt - 1)
                    last_exc = ProviderError(provider, f"HTTP {status} from {redact(str(resp.url))}", status)
                    if attempt < max_attempts:
                        sleep(min(delay, 60) + random.uniform(0, 0.25))
                    continue
                if status >= 400:
                    record_usage(provider=provider, operation=operation, success=False, status_code=status, latency_ms=latency, error=f"HTTP {status}")
                    raise ProviderError(provider, f"HTTP {status} from {redact(str(resp.url))}", status, retryable=False)
                record_usage(provider=provider, operation=operation, success=True, status_code=status, latency_ms=latency)
                return resp
            except httpx.HTTPError as e:
                latency = int((time.monotonic() - t0) * 1000)
                record_usage(provider=provider, operation=operation, success=False, status_code=status, latency_ms=latency, error=redact(type(e).__name__ + ": " + str(e)))
                last_exc = ProviderError(provider, redact(f"{type(e).__name__}: {e}"))
                if attempt < max_attempts:
                    sleep(base_delay * 2 ** (attempt - 1) + random.uniform(0, 0.25))
        assert last_exc is not None
        raise last_exc
    finally:
        if owns:
            client.close()
