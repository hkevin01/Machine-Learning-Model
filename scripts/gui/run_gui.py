#!/usr/bin/env python3
"""GUI launcher (relocated from project root).

Maintained backward compatibility via a thin root stub.
"""
from __future__ import annotations

import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
SRC = os.path.join(ROOT, 'src')
if SRC not in sys.path:
    sys.path.insert(0, SRC)

def main() -> int:
    print("🚀 Starting Machine Learning Framework Explorer (relocated launcher)...")
    try:
        from machine_learning_model.gui.main_window_pyqt6 import (
            main as gui_main,  # type: ignore
        )
    except ImportError as exc:  # pragma: no cover
        print(f"❌ PyQt6 GUI import error: {exc}")
        return 1
    return gui_main()


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
