#!/usr/bin/env python3
"""Deprecated stub -> scripts/testing/test_agent_workflow.sh"""
from __future__ import annotations

import os
import subprocess
import sys

ROOT = os.path.dirname(__file__)
TARGET = os.path.join(ROOT, 'scripts', 'testing', 'test_agent_workflow.sh')
print("⚠️  Deprecated: test_agent_workflow.sh moved to scripts/testing/", file=sys.stderr)
if os.path.isfile(TARGET):
    # Run the bash script
    result = subprocess.run(['bash', TARGET], cwd=ROOT)
    sys.exit(result.returncode)
else:
    print("❌ Relocated test script missing.", file=sys.stderr)
    sys.exit(1)
