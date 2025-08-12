#!/bin/bash
# (Relocated) ML Workflow Agent - Launch Script
# Original file moved from project root to scripts/agent/run_agent.sh

echo "🤖 Machine Learning Workflow Agent - Launch Script"
echo "=================================================="

# Check if we're in project root (one level up from this script directory)
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "${SCRIPT_DIR}/../.." && pwd)"
cd "$ROOT_DIR" || {
  echo "❌ Failed to change to project root" >&2
  exit 1
}

# Check for Python
if ! command -v python3 &> /dev/null; then
	echo "❌ Error: Python 3 is required but not installed."; exit 1
fi

# Activate virtual environment if it exists
if [ -d "Learning" ]; then
	echo "🔧 Activating virtual environment..."; source Learning/bin/activate
elif [ -d "venv" ]; then
	echo "🔧 Activating virtual environment..."; source venv/bin/activate
elif [ -d ".venv" ]; then
	echo "🔧 Activating virtual environment..."; source .venv/bin/activate
else
	echo "⚠️  Warning: No virtual environment found. Using system Python."
fi

echo "📦 Checking dependencies..."
python3 - <<'PY'
import sys, importlib
required = ['pandas','numpy','scikit-learn','matplotlib','seaborn']
missing = []
for pkg in required:
	try: importlib.import_module(pkg.replace('-', '_'))
	except Exception: missing.append(pkg)
if missing:
	print(f"❌ Missing packages: {missing}")
else:
	print("✅ All dependencies satisfied")
PY

echo "\n🚀 Launching ML Workflow Agent..."
echo "=================================================="

if [ -f "src/machine_learning_model/main_app.py" ]; then
	python3 -m src.machine_learning_model.main_app
elif [ -f "src/machine_learning_model/gui/workflow_gui.py" ]; then
	python3 -m src.machine_learning_model.gui.workflow_gui
else
	echo "⚠️  Agent module not found, launching traditional GUI..."
	python3 run_gui.py
fi

echo "\n👋 Thanks for using ML Workflow Agent!"
