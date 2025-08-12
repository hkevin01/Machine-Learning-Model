#!/usr/bin/env python3
"""Backward-compatible stub for relocated GUI launcher.

Delegates to scripts/gui/run_gui.py and emits a deprecation notice.
"""
from __future__ import annotations

import os
import runpy

ROOT = os.path.dirname(os.path.dirname(__file__))  # Go up from scripts/legacy/
RELOCATED = os.path.join(ROOT, 'scripts', 'gui', 'run_gui.py')


def main() -> int:
    print("⚠️  Deprecated: run_gui.py moved to scripts/gui/run_gui.py")
    print("⚠️  This stub is now in scripts/legacy/run_gui.py")
    if os.path.isfile(RELOCATED):
        runpy.run_path(RELOCATED, run_name='__main__')
        return 0
    print("❌ Relocated launcher missing; please reinstall or check repository integrity.")
    return 1


if __name__ == '__main__':  # pragma: no cover
    raise SystemExit(main())
