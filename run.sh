#!/usr/bin/env bash
# Unified root entrypoint (wrapper) for the project.
# Delegates to Docker GUI runner by default, but also provides:
#   ./run.sh --healthcheck   # run environment + GUI + ML quick diagnostics
#   ./run.sh --headless      # (pass-through) container headless GUI import test
#   ./run.sh --local         # (pass-through) run GUI natively
#   ./run.sh --rebuild       # (pass-through) force Docker image rebuild
#   ./run.sh --help          # show this help

set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DOCKER_RUN_SCRIPT="$ROOT/scripts/docker/run.sh"
HEALTH_CHECK_PY="$ROOT/scripts/diagnostics/health_check.py"

print_help() {
  cat <<'EOF'
Machine Learning Model - Root Launcher

Usage:
  ./run.sh [options]

Primary (delegated) options (passed through to Docker runner):
  --rebuild        Rebuild the GUI Docker image
  --gpu            Enable GPU passthrough
  --headless       Run container headless import self-test
  --local          Run GUI natively without Docker

Wrapper-specific options:
  --healthcheck    Run comprehensive host diagnostics (Python, deps, GUI, ML)
  -h, --help       Show this help

Examples:
  ./run.sh                 # Launch GUI (Docker)
  ./run.sh --healthcheck   # Verify environment health
  ./run.sh --local         # Launch GUI natively
  ./run.sh --rebuild --gpu # Rebuild image and run with GPU

EOF
}

if [[ $# -gt 0 ]]; then
  case "$1" in
    --help|-h)
      print_help
      exit 0
      ;;
    --healthcheck)
      shift
      if [[ ! -f "$HEALTH_CHECK_PY" ]]; then
        echo "[error] Health check script missing at $HEALTH_CHECK_PY" >&2
        exit 2
      fi
      if command -v python3 >/dev/null 2>&1; then
        echo "[diag] Running health diagnostics..."
        python3 "$HEALTH_CHECK_PY" "$@"
      else
        echo "[error] python3 not found in PATH" >&2
        exit 3
      fi
      exit $?
      ;;
  esac
fi

if [[ ! -f "$DOCKER_RUN_SCRIPT" ]]; then
  echo "[error] Docker run script not found at $DOCKER_RUN_SCRIPT" >&2
  echo "You may have an incomplete checkout. Expected scripts/docker/run.sh" >&2
  exit 4
fi

# Pass everything else through to the docker runner.
exec "$DOCKER_RUN_SCRIPT" "$@"
