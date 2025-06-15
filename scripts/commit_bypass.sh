#!/bin/bash
# Emergency commit script that bypasses pre-commit hooks
# Customized for Machine Learning Model project
# Use only when pre-commit is broken and you need to commit urgently

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[0;33m'
BLUE='\033[0;34m'
PURPLE='\033[0;35m'
NC='\033[0m' # No Color

PROJECT_NAME="Machine Learning Model"
PROJECT_PREFIX="[ML]"

# Default commit message if none provided
DEFAULT_COMMIT_MESSAGE="Emergency commit - bypass pre-commit hooks"

if [ $# -eq 0 ]; then
    echo -e "${YELLOW}No commit message provided. Using default: '$DEFAULT_COMMIT_MESSAGE'${NC}"
    echo -e "${BLUE}Usage: $0 'commit message' [--no-sync]${NC}"
    echo -e "${YELLOW}Options:${NC}"
    echo -e "  --no-sync         Skip syncing with remote repository"
    echo -e "\n${BLUE}Examples:${NC}"
    echo -e "${YELLOW}  $0 'Fix data loader bug'${NC}"
    echo -e "${YELLOW}  $0 'Update model training script' --no-sync${NC}"
    echo -e "${YELLOW}  $0  # Uses default commit message and syncs${NC}"
    COMMIT_MESSAGE="$DEFAULT_COMMIT_MESSAGE"
else
    COMMIT_MESSAGE="$1"
fi

SYNC_REMOTE=true

# Parse all arguments
shift
while [[ $# -gt 0 ]]; do
    case $1 in
        --no-sync)
            SYNC_REMOTE=false
            shift
            ;;
        *)
            echo -e "${YELLOW}Unknown option: $1${NC}"
            shift
            ;;
    esac
done

echo -e "${PURPLE}🚀 $PROJECT_NAME Emergency Commit Tool${NC}"
echo -e "${YELLOW}⚠️  Bypassing pre-commit hooks for emergency commit${NC}"

# Check if we're in a git repo
if ! git rev-parse --is-inside-work-tree &>/dev/null; then
    echo -e "${RED}❌ Error: Not inside a git repository${NC}"
    exit 1
fi

# Show current branch
CURRENT_BRANCH=$(git branch --show-current)
echo -e "${BLUE}🔄 Current branch: $CURRENT_BRANCH${NC}"

# Check for large files and exclude them
echo -e "${BLUE}🔍 Checking for large files...${NC}"
find . -type f -size +10M -not -path "./.git/*" -not -path "./venv/*" -not -path "./.venv/*" > large_files.tmp

if [ -s large_files.tmp ]; then
    echo -e "${YELLOW}⚠️  Found large files that will be excluded:${NC}"
    while IFS= read -r file; do
        echo -e "${YELLOW}  - $file${NC}"
        # Add to .gitignore if not already there
        basename_file=$(basename "$file")
        if ! grep -Fxq "$basename_file" .gitignore 2>/dev/null; then
            echo "$basename_file" >> .gitignore
        fi
        # Also add the full path
        if ! grep -Fxq "$file" .gitignore 2>/dev/null; then
            echo "$file" >> .gitignore
        fi
    done < large_files.tmp
    #
    # Stage the updated .gitignore
    git add .gitignore
    echo -e "${GREEN}✅ Updated .gitignore with large files${NC}"
fi
rm -f large_files.tmp

# Exclude specific problematic paths
echo -e "${BLUE}🧹 Excluding problematic files from staging...${NC}"
# Reset any previously staged large files
git reset HEAD models/legacy/MODELS/Machine/bin/python* 2>/dev/null || true
git reset HEAD models/legacy/MODELS/ 2>/dev/null || true
git reset HEAD MODELS/ 2>/dev/null || true

# Stage only safe files
echo -e "${BLUE}📝 Staging safe files...${NC}"
git add -A
git reset HEAD models/legacy/MODELS/Machine/bin/python* 2>/dev/null || true
git reset HEAD "models/legacy/MODELS/Machine/bin/"* 2>/dev/null || true

# Check for staged changes
if git diff --cached --quiet; then
    echo -e "${YELLOW}⚠️  No staged changes to commit${NC}"
    exit 0
fi

# Show what will be committed
echo -e "${BLUE}📋 Files to be committed:${NC}"
git diff --cached --name-only | head -10
TOTAL_FILES=$(git diff --cached --name-only | wc -l)
if [ "$TOTAL_FILES" -gt 10 ]; then
    echo -e "${YELLOW}... and $(($TOTAL_FILES - 10)) more files${NC}"
fi

# Commit bypassing pre-commit hooks
echo -e "${BLUE}💾 Creating emergency commit...${NC}"
git commit --no-verify -m "$COMMIT_MESSAGE"
if [ $? -eq 0 ]; then
    echo -e "${GREEN}✅ Emergency commit completed for $PROJECT_NAME${NC}"
    echo -e "${BLUE}📝 Commit message: '$COMMIT_MESSAGE'${NC}"
else
    echo -e "${RED}❌ Commit failed${NC}"
    exit 1
fi

# Sync with remote if requested
if [ "$SYNC_REMOTE" = true ]; then
    REMOTE_NAME=$(git remote | head -n1)
    if [ -z "$REMOTE_NAME" ]; then
        echo -e "${RED}❌ No remote repository configured${NC}"
        echo -e "${YELLOW}💡 Add a remote first: git remote add origin <github-url>${NC}"
        exit 1
    fi
    echo -e "${BLUE}📡 Pushing changes to remote: $REMOTE_NAME${NC}"
    git push $REMOTE_NAME $CURRENT_BRANCH
    if [ $? -eq 0 ]; then
        echo -e "${GREEN}✅ Successfully pushed to remote${NC}"
    else
        echo -e "${RED}❌ Failed to push to remote${NC}"
        exit 1
    fi
else
    echo -e "${YELLOW}• Remember to push changes: git push (or use --sync next time)${NC}"
fi

echo -e "\n${PURPLE}📋 Next Steps:${NC}"
echo -e "${YELLOW}1. Verify commit: git log --oneline -1${NC}"
echo -e "${YELLOW}2. Clean up large files in models/legacy/MODELS/${NC}"
echo -e "${YELLOW}3. Fix pre-commit issues gradually${NC}"
echo -e "${YELLOW}4. Run tests: make test${NC}"
echo -e "${YELLOW}5. Check code: make check${NC}"
