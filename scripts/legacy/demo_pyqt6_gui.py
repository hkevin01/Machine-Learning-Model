#!/usr/bin/env python3
"""Deprecated stub for demo_pyqt6_gui.py -> scripts/gui/demo_pyqt6_gui.py"""
from __future__ import annotations

import os
import runpy
import sys

ROOT = os.path.dirname(os.path.dirname(__file__))  # Go up one level from scripts/legacy/
TARGET = os.path.join(ROOT, 'scripts', 'gui', 'demo_pyqt6_gui.py')
print("⚠️  Deprecated: demo_pyqt6_gui.py moved to scripts/gui/demo_pyqt6_gui.py", file=sys.stderr)
print("⚠️  This stub is now located in scripts/legacy/demo_pyqt6_gui.py", file=sys.stderr)
if os.path.isfile(TARGET):
    runpy.run_path(TARGET, run_name='__main__')
else:
    print("❌ Relocated script missing.", file=sys.stderr)
    sys.exit(1)
