#!/bin/bash

# ML Workflow Agent - Launch Script
# This script launches the agent-based ML workflow navigator

echo "🤖 Machine Learning Workflow Agent - Launch Script"
echo "=================================================="

# Check if we're in the right directory
if [ ! -f "run_gui.py" ]; then
    echo "❌ Error: run_gui.py not found. Please run this script from the project root directory."
    exit 1
fi

# Check for Python
if ! command -v python3 &> /dev/null; then
    echo "❌ Error: Python 3 is required but not installed."
    exit 1
fi

# Activate virtual environment if it exists
if [ -d "Learning" ]; then
    echo "🔧 Activating virtual environment..."
    source Learning/bin/activate
elif [ -d "venv" ]; then
    echo "🔧 Activating virtual environment..."
    source venv/bin/activate
elif [ -d ".venv" ]; then
    echo "🔧 Activating virtual environment..."
    source .venv/bin/activate
else
    echo "⚠️  Warning: No virtual environment found. Using system Python."
fi

# Check for required packages
echo "📦 Checking dependencies..."
python3 -c "
import sys
required_packages = ['tkinter', 'pandas', 'numpy', 'scikit-learn', 'matplotlib', 'seaborn']
missing_packages = []

for package in required_packages:
    try:
        if package == 'tkinter':
            import tkinter
        else:
            __import__(package)
    except ImportError:
        missing_packages.append(package)

if missing_packages:
    print(f'❌ Missing packages: {missing_packages}')
    print('📥 Installing missing packages...')
    import subprocess
    for package in missing_packages:
        if package == 'tkinter':
            print('⚠️  tkinter needs to be installed system-wide. On Ubuntu: sudo apt-get install python3-tk')
        else:
            subprocess.call([sys.executable, '-m', 'pip', 'install', package])
else:
    print('✅ All dependencies satisfied')
"

# Launch the agent mode application
echo ""
echo "🚀 Launching ML Workflow Agent..."
echo "=================================================="

# First try the new agent mode launcher
if [ -f "src/machine_learning_model/main_app.py" ]; then
    python3 -m src.machine_learning_model.main_app
elif [ -f "src/machine_learning_model/gui/workflow_gui.py" ]; then
    python3 -m src.machine_learning_model.gui.workflow_gui
else
    # Fallback to traditional GUI
    echo "⚠️  Agent mode not found, launching traditional GUI..."
    python3 run_gui.py
fi

echo ""
echo "👋 Thanks for using ML Workflow Agent!"
