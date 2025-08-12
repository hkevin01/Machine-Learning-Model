#!/bin/bash
# Remove all stub files from root directory
# Functionality is preserved in organized subdirectories

echo "🧹 Cleaning up root directory..."

# Remove Python stub files
echo "Removing Python stubs..."
rm -f demo_pyqt6_gui.py
rm -f quick_test_agent.py
rm -f run_gui.py
rm -f run_gui_pyqt6.py
rm -f run_gui_pyqt6_only.py
rm -f test_algorithm_database.py
rm -f test_complete_gui.py
rm -f test_enhanced_results.py
rm -f test_gui_headless.py
rm -f test_imports.py
rm -f test_pyqt6_gui.py

# Remove shell script stubs
echo "Removing shell script stubs..."
rm -f activate_venv.sh
rm -f run.sh
rm -f run_agent.sh
rm -f run_pyqt6_demo.sh
rm -f test_agent_workflow.sh

# Remove batch files
echo "Removing batch file stubs..."
rm -f run_agent.bat

# Remove documentation stubs
echo "Removing documentation stubs..."
rm -f PROJECT_CLEANUP_SUMMARY.md
rm -f PYQT6_GUI_GUIDE.md

# Remove cleanup scripts
echo "Removing temporary cleanup scripts..."
rm -f cleanup_root.py
rm -f move_to_legacy.py

echo "✅ Root directory cleanup complete!"
echo ""
echo "All functionality preserved in organized locations:"
echo "  📁 scripts/gui/ - GUI scripts"
echo "  📁 scripts/agent/ - Agent scripts"
echo "  📁 scripts/testing/ - Test scripts"
echo "  📁 scripts/env/ - Environment scripts"
echo "  📁 scripts/docker/ - Docker scripts"
echo "  📁 config/ - Configuration files"
echo "  📁 docs/ - Documentation"
echo ""
echo "Root now contains only essential project files! 🎉"
