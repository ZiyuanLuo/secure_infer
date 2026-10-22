#!/usr/bin/env bash
set -euo pipefail
cd -- "$(dirname -- "${BASH_SOURCE[0]}")"
if [[ -f .env ]]; then
  set -a
  source .env
  set +a
fi
export SERVER_URL="${SERVER_URL:-http://127.0.0.1:${PORT:-8080}}"
exec "${PYTHON:-python3}" request.py "$@"
