#!/bin/bash
# Fix Pre-commit Config Issues
# Ensures the pre-commit configuration file is properly fixed and staged

# Force output to terminal
exec > >(tee /dev/tty)
exec 2>&1

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[0;33m'
BLUE='\033[0;34m'
PURPLE='\033[0;35m'
NC='\033[0m' # No Color

echo -e "${PURPLE}🔧 Pre-commit Config Fix Script${NC}"
echo -e "${BLUE}==============================${NC}"
echo -e "${BLUE}⏰ Started at: $(date)${NC}"
echo -e "${BLUE}📍 Current directory: $(pwd)${NC}"

# Check if we're in a git repository
if ! git rev-parse --is-inside-work-tree &>/dev/null; then
    echo -e "${RED}❌ Error: Not inside a git repository${NC}"
    exit 1
fi

echo -e "${BLUE}🔎 Checking for pre-commit config file...${NC}"
if [ ! -f ".pre-commit-config.yaml" ]; then
    echo -e "${RED}❌ Pre-commit config file not found${NC}"
    echo -e "${YELLOW}💡 Create one or run from the project root directory${NC}"
    exit 1
fi

echo -e "${BLUE}🔧 Fixing pre-commit config file...${NC}"

# Backup original file
cp .pre-commit-config.yaml .pre-commit-config.yaml.bak
echo -e "${GREEN}✅ Backup created: .pre-commit-config.yaml.bak${NC}"

# Remove trailing whitespace (twice to be sure)
echo -e "${BLUE}✂️ Removing trailing whitespace...${NC}"
sed -i 's/[[:space:]]*$//' .pre-commit-config.yaml
sed -i 's/[[:space:]]*$//' .pre-commit-config.yaml

# Ensure file ends with exactly one newline
echo -e "${BLUE}📄 Ensuring file ends with newline...${NC}"
if [ -s ".pre-commit-config.yaml" ]; then
    # Add temporary marker
    echo "# END OF FILE MARKER" >> .pre-commit-config.yaml
    # Remove trailing empty lines
    sed -i -e :a -e '/^\n*$/{$d;N;ba' -e '}' .pre-commit-config.yaml
    # Add single newline
    echo "" >> .pre-commit-config.yaml
    # Remove marker if it exists
    sed -i '/# END OF FILE MARKER/d' .pre-commit-config.yaml
fi

# Fix specific pre-commit config format issues
echo -e "${BLUE}🔄 Fixing format issues...${NC}"

# Fix indentation
sed -i 's/^[ ]*\([^[:space:]]\)/    \1/g' .pre-commit-config.yaml

# Make sure mypy has explicit-package-bases
if grep -q "repo: https://github.com/pre-commit/mirrors-mypy" ".pre-commit-config.yaml"; then
    if ! grep -q "explicit-package-bases" ".pre-commit-config.yaml"; then
        echo -e "${BLUE}➕ Adding explicit-package-bases to mypy config...${NC}"
        sed -i '/repo: https:\/\/github.com\/pre-commit\/mirrors-mypy/,/hooks:/{/args:/s/\[\(.*\)\]/[\1, "--explicit-package-bases"]/}' ".pre-commit-config.yaml"
        if [ $? -eq 0 ]; then
            echo -e "${GREEN}✅ Added --explicit-package-bases to mypy config${NC}"
        else
            echo -e "${YELLOW}⚠️ Could not update mypy config automatically${NC}"
        fi
    else
        echo -e "${GREEN}✅ explicit-package-bases already in mypy config${NC}"
    fi
fi

# Force stage the pre-commit config with git
echo -e "${BLUE}📦 Staging pre-commit config...${NC}"
git add .pre-commit-config.yaml

# Verify it was staged
if git diff --cached --quiet -- .pre-commit-config.yaml; then
    echo -e "${RED}❌ Failed to stage pre-commit config${NC}"
    echo -e "${YELLOW}💡 Try manually: git add .pre-commit-config.yaml${NC}"

    # Try absolute path as fallback
    echo -e "${BLUE}🔄 Trying with absolute path...${NC}"
    git add "$(pwd)/.pre-commit-config.yaml"

    if git diff --cached --quiet -- .pre-commit-config.yaml; then
        echo -e "${RED}❌ Still failed to stage with absolute path${NC}"
        echo -e "${YELLOW}Manual intervention required:${NC}"
        echo -e "1. Run: git add .pre-commit-config.yaml"
        echo -e "2. Verify with: git status"
        exit 1
    else
        echo -e "${GREEN}✅ Successfully staged with absolute path${NC}"
    fi
else
    echo -e "${GREEN}✅ Pre-commit config staged successfully${NC}"
fi

echo -e "\n${PURPLE}🎉 Pre-commit config fix completed!${NC}"
echo -e "${BLUE}📋 Summary:${NC}"
echo -e "✅ Fixed format issues in pre-commit config"
echo -e "✅ Ensured file ends with newline"
echo -e "✅ Staged pre-commit config"

echo -e "\n${BLUE}🔄 Next steps:${NC}"
echo -e "1. Run: git status (verify .pre-commit-config.yaml is staged)"
echo -e "2. Commit your changes: git commit -m \"Fix pre-commit config\""
echo -e "3. Push your changes: git push"

exit 0
