#!/usr/bin/env bash
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
chmod +x "${SCRIPT_DIR}/demo_pyqt6_gui.py" || true
python "${SCRIPT_DIR}/demo_pyqt6_gui.py"
