#!/bin/bash
# Script to fix isort client connection issues

# Colors for output
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[0;33m'
RED='\033[0;31m'
NC='\033[0m' # No Color

echo -e "${BLUE}🔧 Fixing isort client connection issues...${NC}"

# Check if virtual environment is activated
if [ -z "$VIRTUAL_ENV" ]; then
    echo -e "${YELLOW}⚠️ Virtual environment is not activated.${NC}"
    echo -e "${BLUE}🔄 Activating virtual environment...${NC}"
    if [ -d "./venv" ]; then
        source ./venv/bin/activate
        if [[ $? -ne 0 ]]; then
            echo -e "${RED}❌ Failed to activate virtual environment.${NC}"
            exit 1
        fi
        echo -e "${GREEN}✅ Virtual environment activated.${NC}"
    else
        echo -e "${RED}❌ Virtual environment not found. Please run setup_virtualenv.sh first.${NC}"
        exit 1
    fi
fi

# Check if isort is installed
echo -e "${BLUE}🔍 Checking if isort is installed...${NC}"
if ! command -v isort &>/dev/null; then
    echo -e "${YELLOW}⚠️ isort is not installed. Installing isort...${NC}"
    pip install isort
    if [[ $? -ne 0 ]]; then
        echo -e "${RED}❌ Failed to install isort.${NC}"
        deactivate
        exit 1
    fi
    echo -e "${GREEN}✅ isort installed successfully.${NC}"
else
    echo -e "${GREEN}✅ isort is already installed.${NC}"
fi

# Check if isort server is running
echo -e "${BLUE}🔄 Restarting isort server...${NC}"
isort --version
if [[ $? -ne 0 ]]; then
    echo -e "${RED}❌ isort client connection failed. Attempting to restart server...${NC}"
    pkill -f isort &>/dev/null
    isort --version
    if [[ $? -ne 0 ]]; then
        echo -e "${RED}❌ Failed to restart isort server.${NC}"
        deactivate
        exit 1
    fi
fi
echo -e "${GREEN}✅ isort client connection fixed.${NC}"

# Deactivate virtual environment
deactivate
echo -e "${BLUE}🔄 Virtual environment deactivated.${NC}"

echo -e "${PURPLE}🎉 isort client issue resolved!${NC}"
