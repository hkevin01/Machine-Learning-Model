#!/bin/bash
# Script to set up a Python virtual environment and install dependencies

# Colors for output
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[0;33m'
RED='\033[0;31m'
NC='\033[0m' # No Color

echo -e "${BLUE}🔧 Setting up Python virtual environment...${NC}"

# Check Python version
PYTHON_VERSION=$(python3 --version 2>&1)
if [[ $? -ne 0 ]]; then
    echo -e "${RED}❌ Python3 is not installed or not found in PATH.${NC}"
    exit 1
fi
echo -e "${GREEN}✅ Found Python version: $PYTHON_VERSION${NC}"

# Define virtual environment directory
VENV_DIR="./venv"

# Check if virtual environment already exists
if [ -d "$VENV_DIR" ]; then
    echo -e "${YELLOW}⚠️ Virtual environment already exists at $VENV_DIR.${NC}"
    echo -e "${BLUE}🔄 Activating existing virtual environment...${NC}"
else
    # Create a new virtual environment
    echo -e "${BLUE}🔄 Creating a new virtual environment at $VENV_DIR...${NC}"
    python3 -m venv "$VENV_DIR"
    if [[ $? -ne 0 ]]; then
        echo -e "${RED}❌ Failed to create virtual environment.${NC}"
        exit 1
    fi
    echo -e "${GREEN}✅ Virtual environment created successfully.${NC}"
fi

# Activate the virtual environment
source "$VENV_DIR/bin/activate"
if [[ $? -ne 0 ]]; then
    echo -e "${RED}❌ Failed to activate virtual environment.${NC}"
    exit 1
fi
echo -e "${GREEN}✅ Virtual environment activated.${NC}"

# Upgrade pip to the latest version
echo -e "${BLUE}🔄 Upgrading pip...${NC}"
pip install --upgrade pip
if [[ $? -ne 0 ]]; then
    echo -e "${RED}❌ Failed to upgrade pip.${NC}"
    deactivate
    exit 1
fi
echo -e "${GREEN}✅ Pip upgraded successfully.${NC}"

# Install numpy as an example
echo -e "${BLUE}🔄 Installing numpy...${NC}"
pip install numpy
if [[ $? -ne 0 ]]; then
    echo -e "${RED}❌ Failed to install numpy.${NC}"
    deactivate
    exit 1
fi
echo -e "${GREEN}✅ Numpy installed successfully.${NC}"

# Deactivate the virtual environment
deactivate
echo -e "${BLUE}🔄 Virtual environment deactivated.${NC}"

echo -e "${PURPLE}🎉 Setup completed!${NC}"
echo -e "${BLUE}🔗 To activate the virtual environment:${NC}"
echo -e "   For Linux/MacOS: source $VENV_DIR/bin/activate"
echo -e "   For Windows PowerShell: .\\venv\\Scripts\\Activate"
echo -e "   For Windows Command Prompt: venv\\Scripts\\activate.bat"
echo -e "${BLUE}🔗 To deactivate the virtual environment, run:${NC}"
echo -e "   deactivate"
