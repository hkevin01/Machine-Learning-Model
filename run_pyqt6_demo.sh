#!/usr/bin/env bash
echo "⚠️  Deprecated: run_pyqt6_demo.sh moved to scripts/gui/run_pyqt6_demo.sh" >&2
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
"$SCRIPT_DIR/scripts/gui/run_pyqt6_demo.sh" "$@"
