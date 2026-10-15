"""FastAPI application: versioned REST API under /api/v1 (aliased at /api), OpenAPI at /api/docs."""
from __future__ import annotations

import logging
import time
import uuid

from fastapi import APIRouter, FastAPI, HTTPException, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from starlette.middleware.base import BaseHTTPMiddleware

from app.api.ratelimit import client_ip, limiter
from app.api.routes import auth, jobs, news, pairs, research, settings, system, watchlists
from app.core.config import get_settings
from app.core.logging import configure_logging
from app.services import usage

log = logging.getLogger("api")


def build_router() -> APIRouter:
    r = APIRouter()
    for mod in (system, auth, pairs, watchlists, news, research, jobs, settings):
        r.include_router(mod.router)
    return r


class RequestContext(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next):
        s = get_settings()
        rid = request.headers.get("x-request-id") or uuid.uuid4().hex[:12]
        if request.url.path.startswith("/api") and not request.url.path.endswith("/health"):
            ok, retry = limiter.allow(f"api:{client_ip(request)}", s.api_rate_limit_per_minute)
            if not ok:
                return JSONResponse({"error": {"code": "rate_limited", "message": f"retry in {retry}s"}}, status_code=429, headers={"Retry-After": str(retry)})
        t0 = time.monotonic()
        resp = await call_next(request)
        resp.headers["x-request-id"] = rid
        resp.headers["X-Content-Type-Options"] = "nosniff"
        resp.headers["Cache-Control"] = "no-store"
        log.info("request", extra={"rid": rid, "method": request.method, "path": request.url.path, "status": resp.status_code,
                                   "ms": int((time.monotonic() - t0) * 1000)})
        return resp


def create_app() -> FastAPI:
    s = get_settings()
    configure_logging(s.log_level)
    problems = s.validate_for_production()
    if problems:
        raise RuntimeError("unsafe production configuration: " + "; ".join(problems))
    usage.install()
    app = FastAPI(title="Splittability Monitor API", version=system.VERSION, docs_url="/api/docs", redoc_url=None,
                  openapi_url="/api/openapi.json",
                  description="Research and paper-tracking API. Scores are relative risk indices, not probabilities or trading advice.")
    app.add_middleware(RequestContext)
    app.include_router(build_router(), prefix="/api/v1")
    app.include_router(build_router(), prefix="/api", include_in_schema=False)

    @app.exception_handler(HTTPException)
    async def http_error(_: Request, exc: HTTPException):
        detail = exc.detail
        msg = detail if isinstance(detail, str) else (detail.get("message") if isinstance(detail, dict) else str(detail))
        codes = {400: "bad_request", 401: "unauthorized", 403: "forbidden", 404: "not_found", 409: "conflict", 422: "validation_error", 429: "rate_limited"}
        return JSONResponse({"error": {"code": codes.get(exc.status_code, "error"), "message": msg, "details": detail if isinstance(detail, dict) else None}},
                            status_code=exc.status_code, headers=getattr(exc, "headers", None))

    @app.exception_handler(RequestValidationError)
    async def validation_error(_: Request, exc: RequestValidationError):
        errs = [{"loc": list(e.get("loc", [])), "msg": e.get("msg"), "type": e.get("type")} for e in exc.errors()]
        return JSONResponse({"error": {"code": "validation_error", "message": "request validation failed", "details": errs}}, status_code=422)

    @app.exception_handler(Exception)
    async def unhandled(_: Request, exc: Exception):
        log.exception("unhandled error")
        return JSONResponse({"error": {"code": "internal_error", "message": "internal server error"}}, status_code=500)

    return app


app = create_app()
