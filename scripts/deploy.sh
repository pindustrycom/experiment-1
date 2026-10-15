#!/usr/bin/env bash
# Update to the latest code and restart. Data (database volume, backups, certificates) is kept.
set -euo pipefail
cd "$(dirname "$0")/.."
[ -f .env ] || { echo "missing .env (copy .env.example and fill it in)" >&2; exit 1; }
if [ -d .git ]; then git pull --ff-only; fi
scripts/backup.sh || echo "warning: pre-deploy backup failed (first deploy?)"
docker compose up -d --build
docker compose ps
