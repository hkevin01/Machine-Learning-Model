#!/bin/bash
# Ubuntu Setup Script for Machine Learning Framework
# This script sets up the environment and installs dependencies on Ubuntu

# Colors
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[0;33m'
RED='\033[0;31m'
PURPLE='\033[0;35m'
NC='\033[0m'

echo -e "${PURPLE}==========================================${NC}"
echo -e "${PURPLE}Machine Learning Framework - Ubuntu Setup${NC}"
echo -e "${PURPLE}==========================================${NC}"

# Check if Python is installed
if ! command -v python3 &> /dev/null; then
    echo -e "${RED}ERROR: Python3 is not installed${NC}"
    echo -e "${YELLOW}Installing Python3...${NC}"
    sudo apt update
    sudo apt install -y python3 python3-pip python3-venv python3-dev
    if [ $? -ne 0 ]; then
        echo -e "${RED}Failed to install Python3${NC}"
        exit 1
    fi
fi

echo -e "${GREEN}✓ Python3 is installed${NC}"
python3 --version

# Check if we're in the correct directory
if [ ! -d "src/machine_learning_model" ]; then
    echo -e "${RED}ERROR: Please run this script from the project root directory${NC}"
    exit 1
fi

# Install system dependencies for GUI (tkinter)
echo -e "${BLUE}Installing system dependencies...${NC}"
sudo apt update
sudo apt install -y python3-tk python3-dev build-essential

# Create virtual environment
echo -e "${BLUE}Creating virtual environment...${NC}"
if [ -d "venv" ]; then
    echo -e "${YELLOW}Virtual environment already exists${NC}"
else
    python3 -m venv venv
    if [ $? -ne 0 ]; then
        echo -e "${RED}ERROR: Failed to create virtual environment${NC}"
        exit 1
    fi
    echo -e "${GREEN}✓ Virtual environment created${NC}"
fi

# Activate virtual environment
echo -e "${BLUE}Activating virtual environment...${NC}"
source venv/bin/activate
if [ $? -ne 0 ]; then
    echo -e "${RED}ERROR: Failed to activate virtual environment${NC}"
    exit 1
fi

echo -e "${GREEN}✓ Virtual environment activated${NC}"

# Upgrade pip
echo -e "${BLUE}Upgrading pip...${NC}"
pip install --upgrade pip
if [ $? -ne 0 ]; then
    echo -e "${YELLOW}WARNING: Failed to upgrade pip, continuing...${NC}"
fi

# Install dependencies
echo -e "${BLUE}Installing dependencies...${NC}"
pip install numpy pandas scikit-learn matplotlib seaborn plotly
if [ $? -ne 0 ]; then
    echo -e "${RED}ERROR: Failed to install dependencies${NC}"
    exit 1
fi

# Install development dependencies
echo -e "${BLUE}Installing development dependencies...${NC}"
pip install pytest pytest-cov black isort mypy flake8
if [ $? -ne 0 ]; then
    echo -e "${YELLOW}WARNING: Failed to install development dependencies, continuing...${NC}"
fi

# Create necessary directories
echo -e "${BLUE}Creating project directories...${NC}"
mkdir -p test-outputs/{logs,reports,coverage,artifacts}
echo -e "${GREEN}✓ Project directories created${NC}"

# Set permissions for scripts
echo -e "${BLUE}Setting script permissions...${NC}"
chmod +x scripts/*.sh
chmod +x scripts/git-tools/*.sh

# Run tests to verify installation
echo -e "${BLUE}Running tests to verify installation...${NC}"
python -m pytest tests/ -v --tb=short
if [ $? -ne 0 ]; then
    echo -e "${YELLOW}WARNING: Some tests failed, but setup may still be functional${NC}"
fi

echo -e "${PURPLE}==========================================${NC}"
echo -e "${GREEN}Setup completed successfully!${NC}"
echo -e "${PURPLE}==========================================${NC}"
echo
echo -e "${BLUE}To activate the environment in the future, run:${NC}"
echo -e "  source venv/bin/activate"
echo
echo -e "${BLUE}To run the GUI:${NC}"
echo -e "  ./scripts/run_gui.sh"
echo -e "  OR: python run_gui.py"
echo
echo -e "${BLUE}To run tests:${NC}"
echo -e "  ./scripts/run_tests.sh"
echo -e "  OR: python -m pytest tests/"
echo
