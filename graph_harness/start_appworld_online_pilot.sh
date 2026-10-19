#!/usr/bin/env bash
set -euo pipefail

HARNESS_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
if [[ -n "${APPWORLD_PYTHON:-}" && -x "$APPWORLD_PYTHON" ]]; then
  BOOTSTRAP_PYTHON="$APPWORLD_PYTHON"
elif command -v python3 >/dev/null 2>&1; then
  BOOTSTRAP_PYTHON="$(command -v python3)"
elif command -v python >/dev/null 2>&1; then
  BOOTSTRAP_PYTHON="$(command -v python)"
else
  echo "Stopped: no bootstrap Python found. Activate a Python environment first." >&2
  exit 2
fi
exec "$BOOTSTRAP_PYTHON" "$HARNESS_ROOT/start_appworld_online_pilot.py" "$@"
