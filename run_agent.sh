#!/usr/bin/env bash
# Deprecated root stub; delegates to scripts/agent/run_agent.sh
echo "⚠️  Deprecated: run_agent.sh moved to scripts/agent/run_agent.sh" >&2
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
"$SCRIPT_DIR/scripts/agent/run_agent.sh" "$@"
