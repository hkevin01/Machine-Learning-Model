#!/usr/bin/env python3
"""One-shot script to further minimize root directory clutter.

Actions:
1. Move meta summary/cleanup markdown files to docs/maintenance/
2. Remove deprecated root stubs already represented elsewhere.
3. Leave essential project metadata intact.

Safe: Only touches known redundant files.
"""
from __future__ import annotations

import shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent
DOC_MAINT = ROOT / "docs" / "maintenance"
LEGACY = ROOT / "scripts" / "legacy"

META_MOVES = [
    "ROOT_CLEANUP_FINAL.md",
    "TEST_CLEANUP_COMPLETE.md",
]

REDUNDANT_STUBS = [
    # Python/gui/agent stubs (real versions live in scripts/*)
    "run_gui.py",
    "run_gui_pyqt6.py",
    "run_gui_pyqt6_only.py",
    "demo_pyqt6_gui.py",
    "quick_test_agent.py",
    # Shell wrappers
    "activate_venv.sh",
    "run.sh",
    "run_agent.sh",
    "run_pyqt6_demo.sh",
    "test_agent_workflow.sh",
    # Docs redirect stubs
    "PROJECT_CLEANUP_SUMMARY.md",
    "PYQT6_GUI_GUIDE.md",
    # Temporary tools
    "cleanup_root.py",
    "move_to_legacy.py",
    "move_test_stubs.py",
]

# Already moved test_*.py earlier; keep any residual safety list
REDUNDANT_STUBS += [
    "test_algorithm_database.py",
    "test_complete_gui.py",
    "test_enhanced_results.py",
    "test_gui_headless.py",
    "test_imports.py",
    "test_pyqt6_gui.py",
]


def move_meta_files():
    DOC_MAINT.mkdir(parents=True, exist_ok=True)
    for name in META_MOVES:
        src = ROOT / name
        if src.exists():
            dest = DOC_MAINT / name
            try:
                shutil.move(str(src), dest)
                print(f"📦 Moved meta doc {name} -> docs/maintenance/")
            except Exception as e:
                print(f"❌ Failed to move {name}: {e}")
        else:
            print(f"ℹ️  Meta doc not present: {name}")


def remove_redundant():
    removed = []
    for name in REDUNDANT_STUBS:
        path = ROOT / name
        if path.exists():
            try:
                path.unlink()
                removed.append(name)
                print(f"🗑️  Removed {name}")
            except Exception as e:
                print(f"❌ Failed to remove {name}: {e}")
    return removed


def summarize(removed: list[str]):
    print("\nSummary:")
    print(f"  Removed {len(removed)} redundant root files")
    if removed:
        for r in sorted(removed):
            print(f"   - {r}")
    print("\nRemaining recommended root contents: core config, metadata, src/, scripts/, tests/, docs/, data/, models/.")


def main():
    print("🚀 Starting root minimization")
    move_meta_files()
    removed = remove_redundant()
    summarize(removed)
    print("✅ Root minimization complete")

if __name__ == "__main__":
    main()
