#!/usr/bin/env bash
# Create (or reset the password of) an admin user. Prompts for the password.
#   scripts/create-admin.sh you@example.com
set -euo pipefail
cd "$(dirname "$0")/.."
docker compose run --rm migrate python -m app.cli create-admin --email "${1:?usage: scripts/create-admin.sh <email>}"
