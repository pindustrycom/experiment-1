#!/usr/bin/env bash
# Restore a pg_dump custom-format backup into the stack's database. DESTROYS the current data.
#   scripts/restore.sh backups/splittability-20261015T020000Z.dump
set -euo pipefail
cd "$(dirname "$0")/.."
file="${1:?usage: scripts/restore.sh <backup.dump>}"
[ -f "$file" ] || { echo "no such file: $file" >&2; exit 1; }
read -r -p "This replaces ALL data in the database with ${file}. Type 'restore' to continue: " ok
[ "$ok" = "restore" ] || { echo "aborted"; exit 1; }
echo "stopping app containers..."
docker compose stop backend worker frontend
echo "restoring..."
docker compose exec -T db sh -c 'pg_restore --clean --if-exists --no-owner --single-transaction -U "$POSTGRES_USER" -d "$POSTGRES_DB"' < "$file"
echo "applying any newer migrations and restarting..."
docker compose run --rm migrate
docker compose up -d
echo "restore complete"
