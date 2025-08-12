#!/usr/bin/env python3
"""Deprecated stub for run_gui_pyqt6.py -> scripts/gui/run_gui_pyqt6.py"""
from __future__ import annotations

import os
import runpy
import sys

ROOT = os.path.dirname(__file__)
TARGET = os.path.join(ROOT, 'scripts', 'gui', 'run_gui_pyqt6.py')
print("⚠️  Deprecated: run_gui_pyqt6.py moved to scripts/gui/run_gui_pyqt6.py", file=sys.stderr)
if os.path.isfile(TARGET):
    runpy.run_path(TARGET, run_name='__main__')
else:
    print("❌ Relocated script missing.", file=sys.stderr)
    sys.exit(1)
