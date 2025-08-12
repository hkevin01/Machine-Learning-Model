#!/bin/bash
# Deprecated stub -> scripts/testing/test_agent_workflow.sh

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TARGET="$ROOT/scripts/testing/test_agent_workflow.sh"

echo "⚠️  Deprecated: test_agent_workflow.sh moved to scripts/testing/" >&2

if [[ -f "$TARGET" ]]; then
    exec bash "$TARGET" "$@"
else
    echo "❌ Relocated test script missing." >&2
    exit 1
fi
