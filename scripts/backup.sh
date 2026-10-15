#!/usr/bin/env bash
# On-demand backup: pg_dump (custom format) from the running stack into ./backups on the host.
# Nightly backups also run automatically inside the worker (02:00 New York time, 14-day retention)
# into the "backups" Docker volume; this script is for ad-hoc copies and off-server storage.
set -euo pipefail
cd "$(dirname "$0")/.."
mkdir -p backups
ts=$(date -u +%Y%m%dT%H%M%SZ)
out="backups/splittability-${ts}.dump"
docker compose exec -T db sh -c 'pg_dump --format=custom --no-owner -U "$POSTGRES_USER" -d "$POSTGRES_DB"' > "${out}.partial"
mv "${out}.partial" "${out}"
echo "wrote ${out} ($(du -h "${out}" | cut -f1))"
# Copy the worker's automatic backups out of the Docker volume as well:
docker compose cp worker:/backups/. backups/ >/dev/null 2>&1 && echo "copied automatic backups to ./backups" || true
