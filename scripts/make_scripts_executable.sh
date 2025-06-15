#!/bin/bash
# Make Scripts Executable
# Sets executable permissions for all scripts in the scripts directory and subdirectories

# Force all output to terminal immediately
exec > >(tee /dev/tty)
exec 2>&1

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[0;33m'
BLUE='\033[0;34m'
PURPLE='\033[0;35m'
NC='\033[0m' # No Color

echo -e "${PURPLE}🔧 Script Permission Fixer${NC}"
echo -e "${BLUE}==================================${NC}"
echo -e "${BLUE}⏰ Started at: $(date)${NC}"
echo

# Check if we're in the right directory (project root)
if [ ! -d "scripts" ]; then
    echo -e "${RED}❌ Error: 'scripts' directory not found${NC}"
    echo -e "${YELLOW}💡 Please run this script from the project root directory${NC}"
    exit 1
fi

echo -e "${BLUE}📂 Finding all shell scripts...${NC}"

# Find all .sh files in scripts directory and subdirectories
SCRIPT_COUNT=0
FAILED_COUNT=0

find scripts -type f -name "*.sh" | while read script; do
    echo -n "  Setting permissions for ${script}... "

    if chmod +x "$script"; then
        echo -e "${GREEN}✓${NC}"
        ((SCRIPT_COUNT++))
    else
        echo -e "${RED}✗${NC}"
        ((FAILED_COUNT++))
    fi
done

# Make this script executable first to ensure it can be run
chmod +x "$0"

echo
echo -e "${BLUE}📋 Results:${NC}"
echo -e "  - ${GREEN}${SCRIPT_COUNT} scripts${NC} made executable"

if [ $FAILED_COUNT -gt 0 ]; then
    echo -e "  - ${RED}${FAILED_COUNT} scripts${NC} failed"
fi

echo
echo -e "${BLUE}🔄 To run a script:${NC}"
echo -e "  ${YELLOW}./scripts/category/script_name.sh${NC}"
echo
echo -e "${BLUE}💡 Examples:${NC}"
echo -e "  ${YELLOW}./scripts/development/setup_venv.sh${NC}"
echo -e "  ${YELLOW}./scripts/testing/quick_test.sh${NC}"
echo -e "  ${YELLOW}./scripts/fixes/fix_mypy_daemon.sh${NC}"
echo -e "  ${YELLOW}./scripts/git-tools/fix_git_unstaged_commits.sh${NC}"

echo
echo -e "${GREEN}✅ All scripts are now executable!${NC}"
