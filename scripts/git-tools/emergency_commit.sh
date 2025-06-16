#!/bin/bash
# Emergency commit script - bypasses ALL restrictions

# Colors
GREEN='\033[0;32m'
YELLOW='\033[0;33m'
BLUE='\033[0;34m'
RED='\033[0;31m'
NC='\033[0m'

echo -e "${YELLOW}🚨 EMERGENCY COMMIT - No restrictions${NC}"

# Get commit message
if [ $# -eq 0 ]; then
    echo -e "${BLUE}Enter commit message:${NC}"
    read -p "> " COMMIT_MSG
    if [ -z "$COMMIT_MSG" ]; then
        COMMIT_MSG="Emergency commit - no restrictions"
    fi
else
    COMMIT_MSG="$1"
fi

# Stage everything
git add -A

# Commit with no verification
echo -e "${BLUE}Committing: ${COMMIT_MSG}${NC}"
git commit --no-verify -m "$COMMIT_MSG"

if [ $? -eq 0 ]; then
    echo -e "${GREEN}✅ Commit successful!${NC}"
else
    echo -e "${RED}❌ Commit failed${NC}"
fi
