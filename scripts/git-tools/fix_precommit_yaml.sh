#!/bin/bash
# Fix Pre-commit YAML Configuration
# Repairs syntax issues in .pre-commit-config.yaml

# Force output to terminal
exec > >(tee /dev/tty)
exec 2>&1

# Colors
GREEN='\033[0;32m'
BLUE='\033[0;34m'
PURPLE='\033[0;35m'
YELLOW='\033[0;33m'
RED='\033[0;31m'
NC='\033[0m'

echo -e "${PURPLE}🔧 Pre-commit YAML Fix Script${NC}"
echo -e "${BLUE}==============================${NC}"
echo -e "${BLUE}⏰ Started at: $(date)${NC}"
echo -e "${BLUE}📍 Current directory: $(pwd)${NC}"

# Check if we're in a git repository
if ! git rev-parse --is-inside-work-tree &>/dev/null; then
    echo -e "${RED}❌ Error: Not inside a git repository${NC}"
    exit 1
fi

# Create a valid pre-commit config file
echo -e "${BLUE}🔄 Creating valid pre-commit config...${NC}"

# Backup existing config if it exists
if [ -f ".pre-commit-config.yaml" ]; then
    cp .pre-commit-config.yaml .pre-commit-config.yaml.bak.$(date +%s)
    echo -e "${BLUE}📑 Backed up existing config${NC}"
fi

# Create a new, valid config
cat > .pre-commit-config.yaml << 'EOF'
# Pre-commit hooks configuration
# See https://pre-commit.com for more information
repos:
-   repo: https://github.com/pre-commit/pre-commit-hooks
    rev: v4.4.0
    hooks:
    -   id: trailing-whitespace
    -   id: end-of-file-fixer
    -   id: check-yaml
    -   id: check-added-large-files
    -   id: check-merge-conflict
    -   id: debug-statements

-   repo: https://github.com/psf/black
    rev: 23.3.0
    hooks:
    -   id: black

-   repo: https://github.com/pycqa/isort
    rev: 5.12.0
    hooks:
    -   id: isort

-   repo: https://github.com/pycqa/flake8
    rev: 6.0.0
    hooks:
    -   id: flake8
        additional_dependencies: [flake8-docstrings]

-   repo: https://github.com/pre-commit/mirrors-mypy
    rev: v1.3.0
    hooks:
    -   id: mypy
        args: ["--explicit-package-bases"]
        additional_dependencies: [types-requests]
EOF

# Ensure file ends with newline
echo >> .pre-commit-config.yaml

# Verify YAML syntax
if command -v python3 >/dev/null 2>&1; then
    echo -e "${BLUE}🔍 Verifying YAML syntax...${NC}"
    if python3 -c 'import yaml; yaml.safe_load(open(".pre-commit-config.yaml"))' 2>/dev/null; then
        echo -e "${GREEN}✅ YAML syntax is valid${NC}"
    else
        echo -e "${RED}❌ YAML syntax validation failed${NC}"
        exit 1
    fi
fi

# Stage the file
git add -f .pre-commit-config.yaml
echo -e "${GREEN}✅ Pre-commit config fixed and staged${NC}"

exit 0
