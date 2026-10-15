"""Daily logical backups with pg_dump (custom format) and age-based retention."""
from __future__ import annotations

import os
import subprocess
from datetime import UTC, datetime, timedelta
from pathlib import Path
from urllib.parse import unquote, urlsplit

from app.core.config import get_settings


def pg_env_and_args() -> tuple[dict, list[str]]:
    u = urlsplit(get_settings().database_url.replace("+psycopg", ""))
    env = {**os.environ, "PGPASSWORD": unquote(u.password or "")}
    args = ["-h", u.hostname or "localhost", "-p", str(u.port or 5432), "-U", unquote(u.username or "postgres"), "-d", u.path.lstrip("/")]
    return env, args


def run_backup(now: datetime | None = None) -> dict:
    s = get_settings()
    now = now or datetime.now(UTC)
    d = Path(s.backup_dir)
    d.mkdir(parents=True, exist_ok=True)
    target = d / f"splittability-{now.strftime('%Y%m%dT%H%M%SZ')}.dump"
    env, args = pg_env_and_args()
    tmp = target.with_suffix(".partial")
    proc = subprocess.run(["pg_dump", "--format=custom", "--no-owner", "--file", str(tmp), *args], env=env, capture_output=True, text=True, timeout=3600)
    if proc.returncode != 0:
        tmp.unlink(missing_ok=True)
        raise RuntimeError(f"pg_dump failed: {proc.stderr.strip()[:1000]}")
    tmp.rename(target)
    removed = []
    cutoff = now - timedelta(days=s.backup_retention_days)
    for f in sorted(d.glob("splittability-*.dump")):
        if datetime.fromtimestamp(f.stat().st_mtime, UTC) < cutoff and f != target:
            f.unlink()
            removed.append(f.name)
    return {"file": str(target), "bytes": target.stat().st_size, "removed": removed}


def list_backups() -> list[dict]:
    d = Path(get_settings().backup_dir)
    if not d.exists():
        return []
    return [{"name": f.name, "bytes": f.stat().st_size, "modified": datetime.fromtimestamp(f.stat().st_mtime, UTC).isoformat()}
            for f in sorted(d.glob("splittability-*.dump"), reverse=True)]
