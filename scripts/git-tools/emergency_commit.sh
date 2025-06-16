#!/bin/bash
# Emergency Commit Script - Bypasses pre-commit hooks
# Use when normal commits are failing and you need to make an emergency commit

# Force output to terminal
exec > >(tee /dev/tty)
exec 2>&1

# Colors
GREEN='\033[0;32m'
YELLOW='\033[0;33m'
BLUE='\033[0;34m'
RED='\033[0;31m'
NC='\033[0m'

echo -e "${YELLOW}⚠️ EMERGENCY COMMIT - Bypassing pre-commit hooks${NC}"

# Get commit message
if [ $# -eq 0 ]; then
    echo -e "${BLUE}Enter commit message:${NC}"
    read -p "> " COMMIT_MSG
    if [ -z "$COMMIT_MSG" ]; then
        COMMIT_MSG="Emergency commit - bypassing pre-commit hooks"
    fi
else
    COMMIT_MSG="$1"
fi

# Perform commit with --no-verify flag
echo -e "${BLUE}Committing with: ${COMMIT_MSG}${NC}"
git commit --no-verify -m "$COMMIT_MSG"

if [ $? -eq 0 ]; then
    echo -e "${GREEN}✅ Emergency commit successful!${NC}"
else
    echo -e "${RED}❌ Commit failed${NC}"
fi
