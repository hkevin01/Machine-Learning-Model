#!/usr/bin/env python3
"""Clean up root directory by removing all stub files."""

import os


def main():
    """Remove all stub files from root directory."""
    root_dir = os.path.dirname(__file__)

    # Files to remove from root (they're already organized in proper locations)
    files_to_remove = [
        # Python stubs (functionality exists in organized locations)
        'demo_pyqt6_gui.py',
        'quick_test_agent.py',
        'run_gui.py',
        'run_gui_pyqt6.py',
        'run_gui_pyqt6_only.py',
        'test_algorithm_database.py',
        'test_complete_gui.py',
        'test_enhanced_results.py',
        'test_gui_headless.py',
        'test_imports.py',
        'test_pyqt6_gui.py',
        # Shell scripts (functionality exists in organized locations)
        'activate_venv.sh',
        'run.sh',
        'run_agent.sh',
        'run_pyqt6_demo.sh',
        'test_agent_workflow.sh',
        # Batch files
        'run_agent.bat',
        # Documentation stubs (content exists in docs/)
        'PROJECT_CLEANUP_SUMMARY.md',
        'PYQT6_GUI_GUIDE.md',
        # This migration script itself
        'move_to_legacy.py'
    ]

    removed_files = []

    for filename in files_to_remove:
        file_path = os.path.join(root_dir, filename)

        if os.path.exists(file_path):
            try:
                os.remove(file_path)
                removed_files.append(filename)
                print(f"🗑️  Removed {filename}")
            except Exception as e:
                print(f"❌ Failed to remove {filename}: {e}")
        else:
            print(f"⚠️  File not found: {filename}")

    print(f"\n🎉 Successfully removed {len(removed_files)} stub files from root")

    if removed_files:
        print("\nRemoved files (functionality preserved in organized locations):")
        for filename in sorted(removed_files):
            print(f"  - {filename}")

    print("\n✅ Root directory cleanup complete!")
    print("All functionality is preserved in organized subdirectories:")
    print("  - scripts/gui/ - GUI scripts")
    print("  - scripts/agent/ - Agent scripts")
    print("  - scripts/testing/ - Test scripts")
    print("  - scripts/env/ - Environment scripts")
    print("  - scripts/docker/ - Docker scripts")
    print("  - config/ - Configuration files")
    print("  - docs/ - Documentation")

if __name__ == '__main__':
    main()
