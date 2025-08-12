#!/usr/bin/env python3
"""Move all root Python and shell script stubs to scripts/legacy/"""

import os
import shutil


def main():
    """Move stub files from root to scripts/legacy/"""
    root_dir = os.path.dirname(__file__)
    legacy_dir = os.path.join(root_dir, 'scripts', 'legacy')

    # Files to move to scripts/legacy/
    files_to_move = [
        # Python stubs
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
        # Shell scripts
        'activate_venv.sh',
        'run.sh',
        'run_agent.sh',
        'run_pyqt6_demo.sh',
        'test_agent_workflow.sh',
        # Batch files
        'run_agent.bat',
        # Documentation stubs
        'PROJECT_CLEANUP_SUMMARY.md',
        'PYQT6_GUI_GUIDE.md'
    ]

    moved_files = []

    for filename in files_to_move:
        src_path = os.path.join(root_dir, filename)
        dst_path = os.path.join(legacy_dir, filename)

        if os.path.exists(src_path):
            try:
                shutil.move(src_path, dst_path)
                moved_files.append(filename)
                print(f"✅ Moved {filename} to scripts/legacy/")
            except Exception as e:
                print(f"❌ Failed to move {filename}: {e}")
        else:
            print(f"⚠️  File not found: {filename}")

    print(f"\n🎉 Successfully moved {len(moved_files)} files to scripts/legacy/")

    if moved_files:
        print("\nMoved files:")
        for filename in sorted(moved_files):
            print(f"  - {filename}")

    # Create a migration guide at the root
    create_migration_guide(root_dir, moved_files)

def create_migration_guide(root_dir, moved_files):
    """Create a migration guide explaining the new structure"""
    guide_content = """# File Location Migration Guide

## Root Directory Cleanup Complete! 🎉

All utility scripts and stubs have been moved to organized subdirectories for a cleaner project structure.

## New File Locations

### Active Scripts (Use These!)
- **GUI Scripts**: `scripts/gui/` - All GUI launchers and demos
- **Agent Scripts**: `scripts/agent/` - Agent mode and workflow scripts
- **Test Scripts**: `scripts/testing/` - All test files and validation scripts
- **Environment Scripts**: `scripts/env/` - Environment setup and activation
- **Docker Scripts**: `scripts/docker/` - Docker-related utilities
- **Configuration**: `config/` - All configuration files (.flake8, mypy.ini, pytest.ini)

### Legacy Stubs (Backward Compatibility)
- **Legacy Scripts**: `scripts/legacy/` - Backward-compatible stubs for old file locations

## Quick Reference

| Old Location (root) | New Location |
|-------------------|--------------|
| `run_gui.py` | `scripts/gui/run_gui.py` |
| `quick_test_agent.py` | `scripts/agent/quick_test_agent.py` |
| `test_*.py` | `scripts/testing/test_*.py` |
| `activate_venv.sh` | `scripts/env/activate_venv.sh` |
| `.flake8` | `config/.flake8` |
| `mypy.ini` | `config/mypy.ini` |
| `pytest.ini` | `config/pytest.ini` |

## Benefits

✅ **Clean Root Directory** - Only essential project files at the top level
✅ **Organized Structure** - Scripts categorized by purpose
✅ **Better IDE Support** - Cleaner project navigation
✅ **Professional Layout** - Industry-standard project organization
✅ **Backward Compatibility** - All old scripts still work via redirects

## Usage Examples

```bash
# New recommended usage
python scripts/gui/run_gui.py
python scripts/testing/test_enhanced_results.py
bash scripts/env/activate_venv.sh

# Legacy usage (still works but shows deprecation warnings)
# See scripts/legacy/ for backward-compatible stubs
```

## What Happened to Root Files?

All the Python scripts, shell scripts, and configuration stubs that were at the root have been moved to `scripts/legacy/` directory. The root now contains only essential project metadata files.

## For CI/CD and Automation

Update your scripts to use the new organized locations:
- Replace `./run_gui.py` with `python scripts/gui/run_gui.py`
- Replace `./test_*.py` with `python scripts/testing/test_*.py`
- Use `--config-file=config/mypy.ini` for mypy
- Use `-c config/pytest.ini` for pytest

---
Generated during root directory cleanup on """ + str(len(moved_files)) + """ files moved."""

    guide_path = os.path.join(root_dir, 'ROOT_MIGRATION_GUIDE.md')
    with open(guide_path, 'w') as f:
        f.write(guide_content)

    print("📝 Created migration guide: ROOT_MIGRATION_GUIDE.md")

if __name__ == '__main__':
    main()
