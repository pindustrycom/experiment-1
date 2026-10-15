# Backend API + worker image (same image, different command).

# pg_dump/pg_restore for the backup job are taken from the official PostgreSQL image so the client
# always matches the server's major version (no OS package repository needed).
FROM postgres:17-trixie AS pgclient

FROM python:3.12-slim-trixie AS base

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

COPY --from=pgclient /usr/lib/postgresql/17/bin/pg_dump /usr/lib/postgresql/17/bin/pg_restore /usr/local/bin/
COPY --from=pgclient /usr/lib/x86_64-linux-gnu/libpq.so.5* /usr/lib/x86_64-linux-gnu/libgssapi_krb5.so.2* \
     /usr/lib/x86_64-linux-gnu/libldap.so.2* /usr/lib/x86_64-linux-gnu/liblber.so.2* /usr/lib/x86_64-linux-gnu/libxxhash.so.0* \
     /usr/lib/x86_64-linux-gnu/libkrb5.so.3* /usr/lib/x86_64-linux-gnu/libk5crypto.so.3* /usr/lib/x86_64-linux-gnu/libcom_err.so.2* \
     /usr/lib/x86_64-linux-gnu/libkrb5support.so.0* /usr/lib/x86_64-linux-gnu/libsasl2.so.2* /usr/lib/x86_64-linux-gnu/libkeyutils.so.1* \
     /usr/lib/x86_64-linux-gnu/
COPY --from=pgclient /usr/share/zoneinfo /usr/share/zoneinfo
RUN pg_dump --version

WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt

COPY pyproject.toml alembic.ini ./
COPY app ./app
COPY worker ./worker
COPY migrations ./migrations

RUN useradd --create-home --uid 10001 app \
 && mkdir -p /backups && chown app:app /backups
USER app

EXPOSE 8000
# Bound to the internal Docker network only; Caddy is the sole public entry point and sets
# X-Forwarded-For, which uvicorn trusts so per-client rate limits see the real client address.
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "2", "--proxy-headers", "--forwarded-allow-ips", "*", "--no-server-header"]
