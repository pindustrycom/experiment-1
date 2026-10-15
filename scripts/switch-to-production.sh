#!/usr/bin/env bash
# Convert a demo installation to production: removes ALL simulated research data (refuses if any
# real data exists), then restarts in APP_MODE=production. Set APP_MODE=production in .env first.
set -euo pipefail
cd "$(dirname "$0")/.."
grep -q '^APP_MODE=production' .env || { echo "set APP_MODE=production in .env first" >&2; exit 1; }
docker compose stop worker backend
docker compose run --rm migrate python -m app.cli wipe-demo --yes
docker compose up -d
echo "production mode: the worker will fetch real data on its next runs (or use Bot activity > Run now)"
