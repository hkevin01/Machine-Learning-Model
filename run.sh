#!/usr/bin/env bash
# =============================================================================
# Machine Learning Model — unified entry point
#
# Usage:
#   ./run.sh              # build & run the CLI app
#   ./run.sh test         # build & run the full test suite
#   ./run.sh gui          # build & launch the PyQt6 GUI (needs X11)
#   ./run.sh build        # build all images without running
#   ./run.sh --help       # show this help
# =============================================================================
set -euo pipefail

CMD="${1:-app}"

case "$CMD" in
  app)
    docker compose up --build app
    ;;
  test)
    docker compose run --rm --build test
    ;;
  gui)
    xhost +local:docker 2>/dev/null || true
    docker compose up --build gui
    ;;
  build)
    docker compose build
    ;;
  --help|-h|help)
    sed -n '3,10p' "$0"
    ;;
  *)
    echo "Unknown command: $CMD  (use app | test | gui | build)" >&2
    exit 1
    ;;
esac
