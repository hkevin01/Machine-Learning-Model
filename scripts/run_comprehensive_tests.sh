#!/bin/bash
# Comprehensive Test Runner Script
# This script activates the virtual environment, runs pytest with coverage,
# and executes linters (flake8) and static analysis (mypy).
# Test logs and reports are saved in the test-outputs folder.

# Colors for output
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[0;33m'
RED='\033[0;31m'
NC='\033[0m'

echo -e "${BLUE}🔧 Starting comprehensive tests...${NC}"

# Check if virtual environment exists
VENV_DIR="./venv"
if [ ! -d "$VENV_DIR" ]; then
    echo -e "${RED}❌ Virtual environment not found. Please run setup_virtualenv.sh first.${NC}"
    exit 1
fi

# Activate virtual environment
source "$VENV_DIR/bin/activate"
if [ $? -ne 0 ]; then
    echo -e "${RED}❌ Failed to activate virtual environment.${NC}"
    exit 1
fi
echo -e "${GREEN}✅ Virtual environment activated.${NC}"

# Ensure test-outputs directories exist
mkdir -p test-outputs/logs test-outputs/reports test-outputs/coverage

# Run pytest and store output log
echo -e "${BLUE}🔄 Running pytest...${NC}"
pytest --maxfail=1 --disable-warnings > test-outputs/logs/pytest.log 2>&1
if [ $? -eq 0 ]; then
    echo -e "${GREEN}✅ Pytest completed successfully. Log stored in test-outputs/logs/pytest.log${NC}"
else
    echo -e "${YELLOW}⚠️ Pytest encountered issues. Check test-outputs/logs/pytest.log for details.${NC}"
fi

# Run coverage: execute tests and generate reports
echo -e "${BLUE}🔄 Running coverage...${NC}"
coverage run -m pytest > /dev/null 2>&1sable-warnings > /dev/null 2>&1
coverage report > test-outputs/coverage/coverage.txt
coverage html -d test-outputs/coverage/htmlerage.txt
if [ $? -eq 0 ]; then-outputs/coverage/html
    echo -e "${GREEN}✅ Coverage reports generated in test-outputs/coverage (txt and html)${NC}"
elseecho -e "${GREEN}✅ Coverage reports generated in test-outputs/coverage (txt and html)${NC}"
    echo -e "${YELLOW}⚠️ Coverage encountered issues. Check test-outputs/coverage/coverage.txt for details.${NC}"
fi  echo -e "${YELLOW}⚠️ Coverage encountered issues. Check test-outputs/coverage/coverage.txt for details.${NC}"
fi
# Run flake8 for linting and save report
echo -e "${BLUE}🔄 Running flake8 linting...${NC}"
flake8 . > test-outputs/reports/flake8.log 2>&1C}"
if [ $? -eq 0 ]; thents/reports/flake8.log 2>&1
    echo -e "${GREEN}✅ flake8 passed. Report stored in test-outputs/reports/flake8.log${NC}"
elseecho -e "${GREEN}✅ flake8 passed. Report stored in test-outputs/reports/flake8.log${NC}"
    echo -e "${YELLOW}⚠️ flake8 found issues. Check test-outputs/reports/flake8.log for details.${NC}"
fi  echo -e "${YELLOW}⚠️ flake8 found issues. Check test-outputs/reports/flake8.log for details.${NC}"
fi
echo -e "${BLUE}🔄 Running mypy static type check...${NC}"
mypy . > test-outputs/reports/mypy.log 2>&1
if [ $? -eq 0 ]; then check...${NC}"
    echo -e "${GREEN}✅ mypy passed. Report stored in test-outputs/reports/mypy.log${NC}"/reports/mypy.log 2>&1
else
    echo -e "${YELLOW}⚠️ mypy found issues. Check test-outputs/reports/mypy.log for details.${NC}"echo -e "${GREEN}✅ mypy passed. Report stored in test-outputs/reports/mypy.log${NC}"
fi
  echo -e "${YELLOW}⚠️ mypy found issues. Check test-outputs/reports/mypy.log for details.${NC}"
# Deactivate virtual environmentfi
deactivate
echo -e "${GREEN}✅ Virtual environment deactivated.${NC}"te virtual environment

echo -e "${PURPLE}🎉 Comprehensive tests complete!${NC}"echo -e "${GREEN}✅ Virtual environment deactivated.${NC}"
