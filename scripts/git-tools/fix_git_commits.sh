#!/bin/bash
# Fix Git Commit Issues Script
# Resolves common git commit problems and staging issues

# Force all output to terminal
exec > >(tee /dev/tty)
exec 2>&1

# Enable debugging output
set -x

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[0;33m'
BLUE='\033[0;34m'
PURPLE='\033[0;35m'
NC='\033[0m' # No Color sequence

echo -e "${PURPLE}🔧 Git Commit Fix Script${NC}"
echo -e "${BLUE}========================${NC}"
echo -e "${BLUE}⏰ Started at: $(date)${NC}"
echo -e "${BLUE}📍 Current directory: $(pwd)${NC}"

# Check if we're in a git repository
if ! git rev-parse --is-inside-work-tree &>/dev/null; then
    echo -e "${RED}❌ Error: Not inside a git repository${NC}"
    exit 1
fi

echo -e "${BLUE}📍 Current status:${NC}"
echo -e "Repository: $(pwd)"
echo -e "Branch: $(git branch --show-current)"

# Function to clean up problematic files
cleanup_problematic_files() {
    echo -e "\n${BLUE}🧹 Cleaning up problematic files...${NC}"

    # Remove large binary files from staging
    echo -e "${YELLOW}Removing large binary files from staging...${NC}"
    git reset HEAD models/legacy/MODELS/Machine/bin/python* 2>/dev/null || true
    git reset HEAD "models/legacy/MODELS/Machine/bin/"* 2>/dev/null || true
    git reset HEAD MODELS/ 2>/dev/null || true

    # Find and exclude large files
    echo -e "${YELLOW}Finding large files (>10MB)...${NC}"
    find . -type f -size +10M -not -path "./.git/*" -not -path "./venv/*" -not -path "./.venv/*" > large_files_found.tmp 2>/dev/null || true

    if [ -s large_files_found.tmp ]; then
        echo -e "${YELLOW}⚠️  Found large files that need to be excluded:${NC}"
        while IFS= read -r file; do
            echo -e "${YELLOW}  - $file${NC}"

            # Add to .gitignore if not already there
            if [ -f "$file" ]; then
                # Add the exact file path
                if ! grep -Fq "$file" .gitignore 2>/dev/null; then
                    echo "$file" >> .gitignore
                fi

                # Add pattern for similar files
                file_pattern="$(dirname "$file")/$(basename "$file" | sed 's/[0-9.]*$/*/g')"
                if [ "$file_pattern" != "$file" ] && ! grep -Fq "$file_pattern" .gitignore 2>/dev/null; then
                    echo "$file_pattern" >> .gitignore
                fi
            fi
        done < large_files_found.tmp

        echo -e "${GREEN}✅ Updated .gitignore with large file exclusions${NC}"
    fi

    rm -f large_files_found.tmp

    # Clean up common cache and temporary files
    echo "Cleaning up cache and temporary files..."
    rm -rf "__pycache__" 2>/dev/null || true
    rm -rf "*.pyc" 2>/dev/null || true
    rm -rf ".pytest_cache" 2>/dev/null || true
    rm -rf ".coverage" 2>/dev/null || true
    rm -rf "htmlcov" 2>/dev/null || true
}

# Function to fix .gitignore
fix_gitignore() {
    echo -e "\n${BLUE}📝 Updating .gitignore...${NC}"

    # Add common exclusions if not present
    GITIGNORE_ADDITIONS=(
        "# Large files and binaries"
        "*.exe"
        "*.dll"
        "*.so"
        "*.dylib"
        "models/legacy/MODELS/"
        "MODELS/"
        "venv/"
        ".venv/"
        "__pycache__/"
        "*.pyc"
        "*.pyo"
        "*.log"
        ".pytest_cache/"
        ".mypy_cache/"
        ".coverage"
        "htmlcov/"
        "# Large model files"
        "*.pkl"
        "*.joblib"
        "*.h5"
        "*.onnx"
        "*.pt"
        "*.pth"
    )

    for item in "${GITIGNORE_ADDITIONS[@]}"; do
        if [[ "$item" == \#* ]]; then
            # It's a comment, add it if this section doesn't exist
            if ! grep -q "$item" .gitignore 2>/dev/null; then
                echo "$item" >> .gitignore
            fi
        else
            # It's a pattern, add if not present
            if ! grep -Fq "$item" .gitignore 2>/dev/null; then
                echo "$item" >> .gitignore
            fi
        fi
    done

    echo -e "${GREEN}✅ .gitignore updated${NC}"
}

# Function to clean working directory
clean_working_directory() {
    echo -e "\n${BLUE}🗑️  Cleaning working directory...${NC}"

    # Remove any temporary files
    find . -name "*.tmp" -type f -delete 2>/dev/null || true
    find . -name "*.temp" -type f -delete 2>/dev/null || true
    find . -name ".DS_Store" -type f -delete 2>/dev/null || true
    find . -name "Thumbs.db" -type f -delete 2>/dev/null || true

    echo -e "${GREEN}✅ Temporary files cleaned${NC}"
}

# Function to stage files safely
stage_files_safely() {
    echo -e "\n${BLUE}📦 Staging files safely...${NC}"

    # First, add .gitignore
    git add .gitignore

    # Stage specific safe directories
    SAFE_DIRS=(
        "src/"
        "tests/"
        "docs/"
        "scripts/"
        "data/raw/"
        "notebooks/"
        "config/"
        "examples/"
        ".github/"
        ".copilot/"
    )

    for dir in "${SAFE_DIRS[@]}"; do
        if [ -d "$dir" ]; then
            echo -e "${YELLOW}Staging $dir${NC}"
            git add "$dir" 2>/dev/null || true
        fi
    done

    # Stage specific safe files in root
    SAFE_FILES=(
        "README.md"
        "requirements.txt"
        "requirements-dev.txt"
        "pyproject.toml"
        ".pre-commit-config.yaml"
        "Makefile"
        "CHANGELOG.md"
        "CONTRIBUTING.md"
        "LICENSE"
        ".env.example"
    )

    for file in "${SAFE_FILES[@]}"; do
        if [ -f "$file" ]; then
            echo -e "${YELLOW}Staging $file${NC}"
            git add "$file" 2>/dev/null || true
        fi
    done

    echo -e "${GREEN}✅ Safe files staged${NC}"
}

# Function to fix whitespace issues
fix_whitespace_issues() {
    echo -e "\n${BLUE}✂️  Fixing whitespace issues...${NC}"

    # Fix trailing whitespace in most files
    echo -e "${YELLOW}Removing trailing whitespace...${NC}"
    find . -type f \( -name "*.py" -o -name "*.sh" -o -name "*.yaml" -o -name "*.yml" \
        -o -name "*.json" -o -name "*.txt" -o -name "*.html" -o -name "*.css" -o -name "*.js" \) \
        -not -path "./.git/*" -not -path "./venv/*" -not -path "./.venv/*" \
        > /tmp/whitespace_fix_files.tmp

    while IFS= read -r file; do
        if [ -f "$file" ]; then
            # Create backup
            cp "$file" "$file.bak"
            # Remove trailing whitespace
            sed -i 's/[[:space:]]*$//' "$file"
            # Check if changes were made
            if cmp -s "$file" "$file.bak"; then
                rm "$file.bak"  # No changes, remove backup
            else
                echo -e "  Fixed: $file"
                rm "$file.bak"
            fi
        fi
    done < /tmp/whitespace_fix_files.tmp

    # Handle Markdown files separately (preserve intentional trailing spaces)
    echo -e "${YELLOW}Carefully fixing Markdown files...${NC}"
    find . -type f -name "*.md" -not -path "./.git/*" -not -path "./venv/*" -not -path "./.venv/*" \
        -exec sh -c '
            file="$1"
            cp "$file" "$file.tmp"
            # Preserve markdown line breaks (double spaces at end of line)
            awk "{
                if (match(substr(\$0, length(\$0)-1, 2), \"  $\")) {
                    print \$0;  # Keep markdown line breaks
                } else {
                    sub(/[[:space:]]*$/, \"\");  # Remove other trailing spaces
                    print \$0;
                }
            }" "$file.tmp" > "$file"
            rm "$file.tmp"
            echo "  Fixed: $file"
        ' sh {} \;

    # Ensure files end with newline
    echo -e "${YELLOW}Ensuring files end with newline...${NC}"
    find . -type f \( -name "*.py" -o -name "*.sh" -o -name "*.md" -o -name "*.yaml" -o -name "*.yml" \
        -o -name "*.json" -o -name "*.txt" \) -not -path "./.git/*" -not -path "./venv/*" -not -path "./.venv/*" \
        -exec sh -c 'if [ -s "$1" ] && [ "$(tail -c1 "$1" | wc -l)" -eq 0 ]; then echo >> "$1"; fi' _ {} \;

    # Clean up temporary files
    rm -f /tmp/whitespace_fix_files.tmp

    echo -e "${GREEN}✅ Whitespace issues fixed${NC}"
}

# Function to create commit (DISABLED - NO COMMIT MODE)
create_commit() {
    echo -e "\n${BLUE}💾 Preparing commit (NO COMMIT MODE)...${NC}"

    # Check if there are staged changes
    if git diff --cached --quiet; then
        echo -e "${YELLOW}⚠️  No staged changes ready for commit${NC}"
        return 1
    fi

    # Show what will be committed
    echo -e "${BLUE}📋 Files ready to be committed:${NC}"
    git diff --cached --name-only | head -20
    TOTAL_FILES=$(git diff --cached --name-only | wc -l)
    if [ "$TOTAL_FILES" -gt 20 ]; then
        echo -e "${YELLOW}... and $(($TOTAL_FILES - 20)) more files${NC}"
    fi

    # Get commit message
    COMMIT_MESSAGE="${1:-Add datasets and update project structure}"

    echo -e "${GREEN}✅ Files are staged and ready for commit${NC}"
    echo -e "${BLUE}Suggested commit message: '$COMMIT_MESSAGE'${NC}"
    echo -e "${YELLOW}💡 To commit manually, run: git commit -m \"$COMMIT_MESSAGE\"${NC}"

    return 0
}

# Function to push to remote (DISABLED - NO COMMIT MODE)
push_to_remote() {
    echo -e "\n${BLUE}📡 Push preparation (NO COMMIT MODE)...${NC}"

    REMOTE_NAME=$(git remote | head -n1)
    if [ -z "$REMOTE_NAME" ]; then
        echo -e "${RED}❌ No remote repository configured${NC}"
        echo -e "${YELLOW}💡 Add a remote first: git remote add origin <github-url>${NC}"
        return 1
    fi

    CURRENT_BRANCH=$(git branch --show-current)
    echo -e "${BLUE}Ready to push to $REMOTE_NAME/$CURRENT_BRANCH${NC}"
    echo -e "${YELLOW}💡 To push manually, run: git push $REMOTE_NAME $CURRENT_BRANCH${NC}"

    return 0
}

# Main execution
main() {
    local commit_message="${1:-Add datasets and update project progress}"
    local should_push="${2:-false}"

    echo -e "${PURPLE}🚀 Starting Git Commit Fix Process (NO COMMIT MODE)${NC}"

    # Step 1: Clean up problematic files
    cleanup_problematic_files

    # Step 2: Fix .gitignore
    fix_gitignore

    # Step 3: Clean working directory
    clean_working_directory

    # Step 4: Stage files safely
    stage_files_safely

    # Step 5: Prepare commit (but don't commit)
    if create_commit "$commit_message"; then
        echo -e "${GREEN}✅ Files staged and ready for commit${NC}"

        # Step 6: Show push instructions
        if [ "$should_push" = "true" ]; then
            push_to_remote
        fi
    else
        echo -e "${YELLOW}⚠️  No changes to stage${NC}"
    fi

    echo -e "\n${PURPLE}🎉 Git commit fix completed (NO COMMIT MODE)!${NC}"
    echo -e "${BLUE}📋 Summary:${NC}"
    echo -e "✅ Cleaned up large files"
    echo -e "✅ Updated .gitignore"
    echo -e "✅ Staged safe files"
    echo -e "✅ Files ready for commit"

    echo -e "\n${BLUE}🔗 Next steps (MANUAL):${NC}"
    echo -e "1. Review staged files: git status"
    echo -e "2. Commit when ready: git commit -m \"$commit_message\""
    echo -e "3. Push if needed: git push"
    echo -e "4. Check commit: git log --oneline -1"
}

# Parse command line arguments
COMMIT_MSG="Add datasets and update project structure - Phase 1 completion"
SHOULD_PUSH="true"
FIX_WHITESPACE_ONLY="false"

while [[ $# -gt 0 ]]; do
    case $1 in
        -m|--message)
            COMMIT_MSG="$2"
            shift 2
            ;;
        --no-push)
            SHOULD_PUSH="false"
            shift
            ;;
        --fix-whitespace)
            FIX_WHITESPACE_ONLY="true"
            shift
            ;;
        -h|--help)
            echo "Usage: $0 [-m|--message 'commit message'] [--no-push] [--fix-whitespace]"
            echo "Options:"
            echo "  -m, --message      Custom commit message"
            echo "  --no-push          Don't push to remote after commit"
            echo "  --fix-whitespace   Only fix whitespace issues and exit"
            echo "  -h, --help         Show this help"
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            echo "Use -h or --help for usage information"
            exit 1
            ;;
    esac
done

# Run just whitespace fix if requested
if [ "$FIX_WHITESPACE_ONLY" = "true" ]; then
    echo -e "${PURPLE}🔧 Running whitespace fix only${NC}"
    fix_whitespace_issues
    echo -e "${GREEN}✅ Whitespace fix completed${NC}"
    exit 0
fi

# Run main function (force no-push mode)
main "$COMMIT_MSG" "false"
