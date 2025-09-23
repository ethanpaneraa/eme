#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "$0")/../.." && pwd)"
export GROUPME_GROUP_ID="89417887"
export GROUPME_ACCESS_TOKEN="${GROUPME_ACCESS_TOKEN:-}"

if [[ -z "${GROUPME_ACCESS_TOKEN:-}" ]]; then
  echo "Set GROUPME_ACCESS_TOKEN in your shell to test." >&2
  exit 1
fi

"$ROOT_DIR/server/scripts/fetch_incremental.sh"


