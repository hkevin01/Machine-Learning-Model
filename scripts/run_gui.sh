#!/bin/bash
# GUI Launcher Script for Machine Learning Framework Explorer
# This script sets up the environment and launches the GUI application

# Colors for output
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[0;33m'
RED='\033[0;31m'
PURPLE='\033[0;35m'
NC='\033[0m'

echo -e "${PURPLE}🚀 Machine Learning GUI Launcher${NC}"
echo -e "${BLUE}==============================${NC}"

# Check if virtual environment exists
VENV_DIR="./venv"
if [ ! -d "$VENV_DIR" ]; then
    echo -e "${YELLOW}⚠️ Virtual environment not found. Creating one...${NC}"
    ./scripts/setup_virtualenv.sh
    if [ $? -ne 0 ]; then
        echo -e "${RED}❌ Failed to create virtual environment${NC}"
        exit 1
    fi
fi

# Activate virtual environment
echo -e "${BLUE}🔄 Activating virtual environment...${NC}"
source "$VENV_DIR/bin/activate"
if [ $? -ne 0 ]; then
    echo -e "${RED}❌ Failed to activate virtual environment${NC}"
    exit 1
fi
echo -e "${GREEN}✅ Virtual environment activated${NC}"

# Install GUI dependencies if needed
echo -e "${BLUE}🔄 Checking GUI dependencies...${NC}"
pip install -q tkinter matplotlib seaborn plotly scikit-learn pandas numpy 2>/dev/null || {
    echo -e "${YELLOW}⚠️ Installing GUI dependencies...${NC}"
    pip install matplotlib seaborn plotly scikit-learn pandas numpy
}

# Check if GUI module exists
if [ ! -f "src/machine_learning_model/gui/main_window.py" ]; then
    echo -e "${YELLOW}⚠️ GUI module not found. Creating basic GUI structure...${NC}"
    mkdir -p src/machine_learning_model/gui
    # The GUI files will be created separately
fi

# Launch the GUI
echo -e "${BLUE}🚀 Launching Machine Learning GUI...${NC}"
python3 run_gui.py

# Deactivate virtual environment
deactivate
echo -e "${GREEN}✅ GUI session ended${NC}"
