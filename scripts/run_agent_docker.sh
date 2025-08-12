#!/usr/bin/env bash
set -euo pipefail

echo "[INFO] Building and starting ml-agent container..."

docker compose up --build ml-agent
