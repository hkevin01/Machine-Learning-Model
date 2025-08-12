#!/usr/bin/env python3
"""Deprecated stub for quick_test_agent.py (moved to scripts/agent/)."""
from __future__ import annotations

import os
import runpy
import sys

ROOT = os.path.dirname(__file__)
TARGET = os.path.join(ROOT, 'scripts', 'agent', 'quick_test_agent.py')
print("⚠️  Deprecated: quick_test_agent.py moved to scripts/agent/quick_test_agent.py", file=sys.stderr)
if os.path.isfile(TARGET):
    runpy.run_path(TARGET, run_name='__main__')
else:
    print("❌ Relocated quick test script missing.", file=sys.stderr)
    sys.exit(1)
