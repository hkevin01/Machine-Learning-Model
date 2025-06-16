#!/bin/bash
# Clean up backup files that cause issues with end-of-file fixer

# Colors for output
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[0;33m'
NC='\033[0m' # No Color

echo -e "${BLUE}🧹 Cleaning up backup files...${NC}"

# Remove pre-commit config backups
if ls .pre-commit-config.yaml.bak* 1> /dev/null 2>&1; then
    echo -e "${YELLOW}🗑️ Removing pre-commit config backups...${NC}"
    rm -f .pre-commit-config.yaml.bak*
fi

# Remove other backup files
echo -e "${YELLOW}🗑️ Removing other backup files...${NC}"
find . -type f \( -name "*.bak" -o -name "*.bak.*" -o -name "*.tmp" -o -name "*.orig" -o -name "*~" \) -delete

echo -e "${GREEN}✅ Backup files cleaned up${NC}"

# Stage pre-commit config
if [ -f ".pre-commit-config.yaml" ]; then
    echo -e "${BLUE}📦 Staging pre-commit config...${NC}"
    git add -f .pre-commit-config.yaml
fi

echo -e "${GREEN}✅ Done!${NC}"
chmod +x "$0"
