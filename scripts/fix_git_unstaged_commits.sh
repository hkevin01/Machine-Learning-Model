#!/bin/bash
# Fix Git Warning for Unstaged Commits
# Resolves issues with unstaged changes and ensures a clean working directory

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[0;33m'
BLUE='\033[0;34m'
PURPLE='\033[0;35m'
NC='\033[0m' # No Color

echo -e "${PURPLE}🔧 Git Unstaged Commits Fix Script${NC}"
echo -e "${BLUE}==================================${NC}"

# Check if we're in a git repository
if ! git rev-parse --is-inside-work-tree &>/dev/null; then
    echo -e "${RED}❌ Error: Not inside a git repository${NC}"
    exit 1
fi

# Show current branch
CURRENT_BRANCH=$(git branch --show-current)
echo -e "${BLUE}🔄 Current branch: $CURRENT_BRANCH${NC}"

# Show unstaged changes
echo -e "${BLUE}📋 Checking for unstaged changes...${NC}"
if git diff --quiet; then
    echo -e "${GREEN}✅ No unstaged changes found${NC}"
else
    echo -e "${YELLOW}⚠️  Unstaged changes detected:${NC}"
    git status -s
fi

# Fix end-of-file issues
echo -e "${BLUE}📄 Fixing end-of-file issues...${NC}"
find . -type f \( -name "*.sh" -o -name "*.py" -o -name "*.md" -o -name "*.yaml" -o -name "*.yml" -o -name "*.txt" \) \
    -not -path "./.git/*" \
    -not -path "./venv/*" \
    -not -path "./.venv/*" \
    -exec sh -c 'if [ -s "$1" ] && [ "$(tail -c1 "$1" | wc -l)" -eq 0 ]; then echo >> "$1"; fi' _ {} \;
echo -e "${GREEN}✅ End-of-file issues fixed${NC}"

# Prompt user to stage changes
echo -e "${BLUE}📝 Staging all changes...${NC}"
git add -A

# Check if there are staged changes
if git diff --cached --quiet; then
    echo -e "${YELLOW}⚠️  No changes staged for commit${NC}"
    exit 0
fi

# Commit changes
echo -e "${BLUE}💾 Creating commit...${NC}"
COMMIT_MESSAGE="Fix unstaged changes and resolve end-of-file issues"
git commit -m "$COMMIT_MESSAGE"

if [ $? -eq 0 ]; then
    echo -e "${GREEN}✅ Commit created successfully${NC}"
else
    echo -e "${RED}❌ Commit failed${NC}"
    exit 1
fi

# Push changes to remote
echo -e "${BLUE}📡 Pushing changes to remote...${NC}"
REMOTE_NAME=$(git remote | head -n1)
if [ -z "$REMOTE_NAME" ]; then
    echo -e "${RED}❌ No remote repository configured${NC}"
    echo -e "${YELLOW}💡 Add a remote first: git remote add origin <github-url>${NC}"
    exit 1
fi

git push "$REMOTE_NAME" "$CURRENT_BRANCH"
if [ $? -eq 0 ]; then
    echo -e "${GREEN}✅ Successfully pushed to remote${NC}"
else
    echo -e "${RED}❌ Failed to push to remote${NC}"
    exit 1
fi

echo -e "\n${PURPLE}🎉 Git unstaged commits fix completed!${NC}"
echo -e "${BLUE}📋 Summary:${NC}"
echo -e "✅ Fixed end-of-file issues"
echo -e "✅ Staged all changes"
echo -e "✅ Created commit: $COMMIT_MESSAGE"
echo -e "✅ Pushed changes to remote"
