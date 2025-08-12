#!/usr/bin/env python3
"""Demo launcher (relocated)."""
from __future__ import annotations

import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
SRC = os.path.join(ROOT, 'src')
if SRC not in sys.path:
    sys.path.insert(0, SRC)

def main() -> int:
    print("Demo PyQt6 GUI - relocated script")
    from machine_learning_model.gui.main_window_pyqt6 import (
        main as gui_main,  # type: ignore
    )
    return gui_main()

if __name__ == '__main__':
    raise SystemExit(main())
