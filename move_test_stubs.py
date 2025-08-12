#!/usr/bin/env python3
"""
Move test_*.py stub files from root to scripts/legacy/ and clean up root directory.
The functional test files are already in scripts/testing/ so these root files are just stubs.
"""

import shutil
from pathlib import Path


def main():
    """Move test stub files from root to scripts/legacy/"""

    # Get the root directory (where this script is located)
    root_dir = Path(__file__).parent
    legacy_dir = root_dir / "scripts" / "legacy"

    # Ensure legacy directory exists
    legacy_dir.mkdir(parents=True, exist_ok=True)

    # Test stub files to move from root to scripts/legacy/
    test_stub_files = [
        "test_algorithm_database.py",
        "test_complete_gui.py",
        "test_enhanced_results.py",
        "test_gui_headless.py",
        "test_imports.py",
        "test_pyqt6_gui.py"
    ]

    print("🧹 Moving test_*.py stub files from root to scripts/legacy/")
    print("   (Functional versions remain in scripts/testing/)")
    print()

    moved_count = 0
    for filename in test_stub_files:
        source_path = root_dir / filename
        dest_path = legacy_dir / filename

        if source_path.exists():
            print(f"   Moving: {filename} → scripts/legacy/{filename}")
            shutil.move(str(source_path), str(dest_path))
            moved_count += 1
        else:
            print(f"   Skip: {filename} (not found)")

    print()
    print(f"✅ Moved {moved_count} test stub files to scripts/legacy/")
    print()
    print("📋 Test organization summary:")
    print("   • Functional tests: scripts/testing/")
    print("   • Unit tests: tests/")
    print("   • Legacy stubs: scripts/legacy/")
    print()
    print("🎯 Usage:")
    print("   python scripts/testing/test_enhanced_results.py")
    print("   python scripts/testing/test_pyqt6_gui.py")
    print("   pytest tests/")

if __name__ == "__main__":
    main()
