#!/bin/bash
# Deprecated stub -> scripts/docker/run.sh

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TARGET="$ROOT/scripts/docker/run.sh"

echo "⚠️  Deprecated: run.sh moved to scripts/docker/" >&2

if [[ -f "$TARGET" ]]; then
    exec bash "$TARGET" "$@"
else
    echo "❌ Relocated Docker script missing." >&2
    exit 1
fi
