#!/bin/bash
# Comprehensive Test Suite
# This script runs all tests (pytest) 
# and executes linters (flake8).

set -e  # Exit on any error

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Create test output directories
mkdir -p test-outputs/reports
mkdir -p test-outputs/coverage

echo -e "${BLUE}🚀 Starting Comprehensive Test Suite...${NC}"

# Function to check if command exists
command_exists() {
    command -v "$1" >/dev/null 2>&1
}

# Check if we're in a virtual environment
if [[ "$VIRTUAL_ENV" == "" ]]; then
    echo -e "${YELLOW}⚠️  Warning: Not in a virtual environment${NC}"
    echo -e "${YELLOW}   Consider activating your virtual environment first${NC}"
fi

# Run pytest with coverage and save report
echo -e "${BLUE}🔄 Running pytest with coverage...${NC}"
if command_exists pytest; then
    pytest --cov=src --cov-report=html:test-outputs/coverage --cov-report=term-missing --cov-report=xml:test-outputs/reports/coverage.xml -v > test-outputs/reports/pytest.log 2>&1
    if [ $? -eq 0 ]; then
        echo -e "${GREEN}✅ pytest passed. Report stored in test-outputs/reports/pytest.log${NC}"
        echo -e "${GREEN}📊 Coverage report available in test-outputs/coverage/index.html${NC}"
    else
        echo -e "${YELLOW}⚠️ pytest found issues. Check test-outputs/reports/pytest.log for details.${NC}"
    fi
else
    echo -e "${RED}❌ pytest not found. Please install pytest.${NC}"
    exit 1
fi

# Run flake8 for linting and save report
echo -e "${BLUE}🔄 Running flake8 linting...${NC}"
if command_exists flake8; then
    flake8 src/ tests/ --max-line-length=88 --extend-ignore=E203,W503 --output-file=test-outputs/reports/flake8.log
    if [ $? -eq 0 ]; then
        echo -e "${GREEN}✅ flake8 passed. Report stored in test-outputs/reports/flake8.log${NC}"
    else
        echo -e "${YELLOW}⚠️ flake8 found issues. Check test-outputs/reports/flake8.log for details.${NC}"
    fi
else
    echo -e "${RED}❌ flake8 not found. Please install flake8.${NC}"
fi

# Run black for code formatting check
echo -e "${BLUE}🔄 Running black code formatting check...${NC}"
if command_exists black; then
    black --check --diff src/ tests/ > test-outputs/reports/black.log 2>&1
    if [ $? -eq 0 ]; then
        echo -e "${GREEN}✅ black passed. Code is properly formatted.${NC}"
    else
        echo -e "${YELLOW}⚠️ black found formatting issues. Check test-outputs/reports/black.log for details.${NC}"
        echo -e "${YELLOW}   Run 'black src/ tests/' to fix formatting issues.${NC}"
    fi
else
    echo -e "${RED}❌ black not found. Please install black.${NC}"
fi

# Run isort for import sorting check
echo -e "${BLUE}🔄 Running isort import sorting check...${NC}"
if command_exists isort; then
    isort --check-only --diff src/ tests/ > test-outputs/reports/isort.log 2>&1
    if [ $? -eq 0 ]; then
        echo -e "${GREEN}✅ isort passed. Imports are properly sorted.${NC}"
    else
        echo -e "${YELLOW}⚠️ isort found import sorting issues. Check test-outputs/reports/isort.log for details.${NC}"
        echo -e "${YELLOW}   Run 'isort src/ tests/' to fix import sorting issues.${NC}"
    fi
else
    echo -e "${RED}❌ isort not found. Please install isort.${NC}"
fi

echo -e "${GREEN}🎉 Comprehensive test suite completed!${NC}"
echo -e "${BLUE}📁 All reports saved in test-outputs/reports/${NC}"
echo -e "${BLUE}📊 Coverage report: test-outputs/coverage/index.html${NC}"
