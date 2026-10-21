#!/usr/bin/env bash
# Encrypt or decrypt files kept in the repository's GitHub releases (database dumps, site builds, experiment state).
# The repository is public, so its releases are too: the password lock on the site (lock_site.py) means nothing if
# the unlocked pages and their data can be downloaded from a release. Every workflow seals a file before it
# uploads it and opens it after it downloads it.
#
#   STATE_KEY=... experiments/hub/state_crypt.sh seal FILE...   encrypt each file in place
#   STATE_KEY=... experiments/hub/state_crypt.sh open FILE...   decrypt each file in place
#
# The workflows set STATE_KEY from the STATE_KEY secret, or from the site PASSWORD when there is none. Set a separate
# STATE_KEY if the site password may change: files sealed under the old key can only be opened with it.
# "open" leaves a file that was never sealed (saved before this script existed) as it is, so the first run after the
# change reads the old files and the next upload replaces them with sealed ones.
set -euo pipefail
mode=${1:-}
shift || true
[ "$mode" = seal ] || [ "$mode" = open ] || { echo "usage: $0 seal|open FILE..." >&2; exit 2; }
[ -n "${STATE_KEY:-}" ] || { echo "STATE_KEY is not set: refusing to handle release files without a key." >&2; exit 1; }
enc=(openssl enc -aes-256-cbc -md sha256 -pbkdf2 -iter 200000 -pass env:STATE_KEY)
for f in "$@"; do
  sealed=false; cmp -s -n 8 "$f" <(printf 'Salted__') && sealed=true
  if [ "$mode" = seal ]; then
    $sealed && { echo "$f is already sealed" >&2; continue; }
    "${enc[@]}" -salt -in "$f" -out "$f.sealed" && mv "$f.sealed" "$f"
  elif $sealed; then
    "${enc[@]}" -d -in "$f" -out "$f.open" || { rm -f "$f.open"; echo "could not open $f: wrong STATE_KEY?" >&2; exit 1; }
    mv "$f.open" "$f"
  fi
done
