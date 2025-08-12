#!/usr/bin/env python3
"""Deprecated stub -> scripts/testing/test_imports.py"""
from __future__ import annotations

import os
import runpy
import sys

ROOT = os.path.dirname(__file__)
TARGET = os.path.join(ROOT, 'scripts', 'testing', 'test_imports.py')
print("⚠️  Deprecated: test_imports.py moved to scripts/testing/", file=sys.stderr)
if os.path.isfile(TARGET):
    runpy.run_path(TARGET, run_name='__main__')
else:
    print("❌ Relocated test script missing.", file=sys.stderr)
    sys.exit(1)
