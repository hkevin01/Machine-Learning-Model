#!/bin/bash
# Simplified Script to Stage All Changes

# Colors
GREEN='\033[0;32m'
BLUE='\033[0;34m'
PURPLE='\033[0;35m'
NC='\033[0m'

echo -e "${PURPLE}🔧 Simplified Git Unstaged Commits Script${NC}"
echo -e "${BLUE}==============================${NC}"

# Check if we're in a git repository
if ! git rev-parse --is-inside-work-tree &>/dev/null; then
    echo -e "${RED}❌ Error: Not inside a git repository${NC}"
    exit 1
fi

# Stage all changes
echo -e "${BLUE}🔄 Staging all changes...${NC}"
git add -A

# Check if there are staged changes
if git diff --cached --quiet; then
    echo -e "${YELLOW}⚠️ No changes to commit${NC}"
    exit 0
fi

echo -e "${GREEN}✅ Changes staged successfully${NC}"
echo -e "${BLUE}🔄 Ready for commit via Source Control extension${NC}"
