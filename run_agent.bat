@echo off
REM ML Workflow Agent - Windows Launch Script
REM This script launches the agent-based ML workflow navigator

echo 🤖 Machine Learning Workflow Agent - Launch Script
echo ==================================================

REM Check if we're in the right directory
if not exist "run_gui.py" (
    echo ❌ Error: run_gui.py not found. Please run this script from the project root directory.
    pause
    exit /b 1
)

REM Check for Python
python --version >nul 2>&1
if errorlevel 1 (
    echo ❌ Error: Python is required but not installed.
    pause
    exit /b 1
)

REM Activate virtual environment if it exists
if exist "Learning\Scripts\activate.bat" (
    echo 🔧 Activating virtual environment...
    call Learning\Scripts\activate.bat
) else if exist "venv\Scripts\activate.bat" (
    echo 🔧 Activating virtual environment...
    call venv\Scripts\activate.bat
) else if exist ".venv\Scripts\activate.bat" (
    echo 🔧 Activating virtual environment...
    call .venv\Scripts\activate.bat
) else (
    echo ⚠️  Warning: No virtual environment found. Using system Python.
)

REM Check for required packages
echo 📦 Checking dependencies...
python -c "import sys; required_packages = ['tkinter', 'pandas', 'numpy', 'scikit-learn', 'matplotlib', 'seaborn']; missing_packages = []; [missing_packages.append(package) for package in required_packages if (lambda package: False if package == 'tkinter' else True)(__import__(package)) is None]; print('✅ All dependencies satisfied') if not missing_packages else print(f'❌ Missing packages: {missing_packages}')"

REM Launch the agent mode application
echo.
echo 🚀 Launching ML Workflow Agent...
echo ==================================================

REM First try the new agent mode launcher
if exist "src\machine_learning_model\main_app.py" (
    python -m src.machine_learning_model.main_app
) else if exist "src\machine_learning_model\gui\workflow_gui.py" (
    python -m src.machine_learning_model.gui.workflow_gui
) else (
    REM Fallback to traditional GUI
    echo ⚠️  Agent mode not found, launching traditional GUI...
    python run_gui.py
)

echo.
echo 👋 Thanks for using ML Workflow Agent!
pause
