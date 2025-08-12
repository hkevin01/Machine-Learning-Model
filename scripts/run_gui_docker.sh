#!/usr/bin/env bash
set -euo pipefail

# Ensure DISPLAY is set (default to :0)
export DISPLAY="${DISPLAY:-:0}"

echo "[INFO] Using DISPLAY=$DISPLAY"

# Allow local docker user to access X server (Linux)
if command -v xhost >/dev/null 2>&1; then
  (xhost +SI:localuser:root >/dev/null 2>&1 || true)
  (xhost +local:docker >/dev/null 2>&1 || true)
fi

echo "[INFO] Building and starting ml-gui container..."
docker compose up --build ml-gui
