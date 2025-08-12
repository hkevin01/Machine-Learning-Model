#!/usr/bin/env python3
"""Deprecated stub -> docs/PROJECT_CLEANUP_SUMMARY.md"""
from __future__ import annotations

import os
import sys

ROOT = os.path.dirname(__file__)
TARGET = os.path.join(ROOT, 'docs', 'PROJECT_CLEANUP_SUMMARY.md')
print("⚠️  Deprecated: PROJECT_CLEANUP_SUMMARY.md moved to docs/", file=sys.stderr)
print(f"📄 View at: {TARGET}", file=sys.stderr)
sys.exit(1)
