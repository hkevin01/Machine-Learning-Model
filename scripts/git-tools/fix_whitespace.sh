#!/bin/bash
# Fix Whitespace Issues and Stage Changes for Clean Git Commits
# Resolves whitespace issues and handles staging in one unified script

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

# Parse command line arguments
AUTO_STAGE=false
VERBOSE=false
SKIP_PYTHON_FIX=false

# Process command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --auto-stage)
            AUTO_STAGE=true
            shift
            ;;
        --verbose)
            VERBOSE=true
            set -x  # Enable bash debugging if verbose
            shift
            ;;
        --skip-python-fix)
            SKIP_PYTHON_FIX=true
            shift
            ;;
        -h|--help)
            echo "Usage: $0 [--auto-stage] [--verbose] [--skip-python-fix]"
            echo "Options:"
            echo "  --auto-stage       Automatically stage all fixed files without prompting"
            echo "  --verbose          Show detailed debugging information"
            echo "  --skip-python-fix  Skip Python-specific fixes (line length, etc.)"
            echo "  -h, --help         Show this help message"
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            echo "Use -h or --help for usage information"
            exit 1
            ;;
    esac
done

echo -e "${PURPLE}🔧 Whitespace Fix & Staging Script${NC}"
echo -e "${BLUE}=================================${NC}"
echo -e "${BLUE}⏰ Started at: $(date)${NC}"

# Check if we're in a git repository
if ! git rev-parse --is-inside-work-tree &>/dev/null; then
    echo -e "${RED}❌ Error: Not inside a git repository${NC}"
    exit 1
fi

echo -e "${BLUE}📍 Current directory: $(pwd)${NC}"
echo -e "${BLUE}🔄 Current branch: $(git branch --show-current)${NC}"

# Function to check if file is binary
is_binary() {
    file --mime "$1" | grep -q "charset=binary"
    return $?
}

# Function to fix whitespace issues
fix_whitespace() {
    echo -e "\n${BLUE}✂️  Fixing whitespace issues...${NC}"

    # Track statistics
    FIXED_COUNT=0
    SKIPPED_COUNT=0
    ERROR_COUNT=0

    # First ensure all files end with newline
    echo -e "${YELLOW}Ensuring files end with newline...${NC}"
    find . -type f \( -name "*.py" -o -name "*.sh" -o -name "*.md" -o -name "*.yaml" -o -name "*.yml" \
        -o -name "*.json" -o -name "*.txt" -o -name "*.html" -o -name "*.css" -o -name "*.js" \
        -o -name "*.c" -o -name "*.cpp" -o -name "*.h" -o -name "*.java" \) \
        -not -path "./.git/*" -not -path "./venv/*" -not -path "./.venv/*" \
        -not -path "./models/*" -not -path "./data/*" \
        -exec sh -c '
            file="$1"
            if [ -f "$file" ] && [ -s "$file" ] && [ "$(tail -c1 "$file" | wc -l)" -eq 0 ]; then
                echo >> "$file"
                echo "  EOL fixed: $file"
                ((FIXED_COUNT++))
            fi
        ' sh {} \;

    # Fix trailing whitespace in code files
    echo -e "${YELLOW}Removing trailing whitespace in code files...${NC}"
    find . -type f \( -name "*.py" -o -name "*.sh" -o -name "*.yaml" -o -name "*.yml" \
        -o -name "*.json" -o -name "*.txt" -o -name "*.html" -o -name "*.css" -o -name "*.js" \
        -o -name "*.c" -o -name "*.cpp" -o -name "*.h" -o -name "*.java" \) \
        -not -path "./.git/*" -not -path "./venv/*" -not -path "./.venv/*" \
        -not -path "./models/*" -not -path "./data/*" \
        -not -name "*.md" \
        > /tmp/whitespace_fix_files.tmp

    while IFS= read -r file; do
        if [ -f "$file" ] && ! is_binary "$file"; then
            # Create backup
            cp "$file" "$file.bak"

            # Remove trailing whitespace
            sed -i 's/[[:space:]]*$//' "$file" 2>/dev/null

            # Check if changes were made
            if cmp -s "$file" "$file.bak"; then
                rm "$file.bak"  # No changes, remove backup
            else
                echo -e "  ${GREEN}Fixed:${NC} $file"
                ((FIXED_COUNT++))
                rm "$file.bak"
            fi
        else
            if [ "$VERBOSE" = true ]; then
                echo -e "  ${YELLOW}Skipped:${NC} $file"
            fi
            ((SKIPPED_COUNT++))
        fi
    done < /tmp/whitespace_fix_files.tmp

    # Handle Markdown files separately (preserve intentional trailing spaces)
    echo -e "${YELLOW}Fixing whitespace in Markdown files (preserving line breaks)...${NC}"
    find . -type f -name "*.md" -not -path "./.git/*" -not -path "./venv/*" -not -path "./.venv/*" \
        -exec sh -c '
            file="$1"
            if [ -f "$file" ]; then
                cp "$file" "$file.tmp"

                # Preserve markdown line breaks (lines ending with exactly two spaces)
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
                ((FIXED_COUNT++))
            fi
        ' sh {} \;

    # Clean up temporary files
    rm -f /tmp/whitespace_fix_files.tmp

    echo -e "\n${GREEN}✅ Whitespace issues fixed${NC}"
    echo -e "${BLUE}📊 Statistics:${NC}"
    echo -e "  - Files fixed: $FIXED_COUNT"
    echo -e "  - Files skipped: $SKIPPED_COUNT"
    echo -e "  - Errors: $ERROR_COUNT"

    return $FIXED_COUNT
}

# Function to fix Python-specific issues
fix_python_issues() {
    if [ "$SKIP_PYTHON_FIX" = true ]; then
        echo -e "${YELLOW}Skipping Python-specific fixes as requested${NC}"
        return 0
    fi

    echo -e "\n${BLUE}🐍 Fixing Python code style issues...${NC}"
    PYTHON_FIXED=0

    # Find Python files
    find . -type f -name "*.py" \
        -not -path "./.git/*" -not -path "./venv/*" -not -path "./.venv/*" \
        > /tmp/python_fix_files.tmp

    while IFS= read -r file; do
        if [ -f "$file" ]; then
            echo "Fixing style issues in $file"

            # Make backup
            cp "$file" "$file.bak"

            # Use autopep8 if available, otherwise fallback to sed
            if command -v autopep8 >/dev/null 2>&1; then
                autopep8 --max-line-length=79 --in-place "$file" 2>/dev/null
                if [ $? -ne 0 ]; then
                    echo -e "${YELLOW}⚠️  autopep8 failed, restoring backup for $file${NC}"
                    mv "$file.bak" "$file"
                else
                    rm "$file.bak"
                    ((PYTHON_FIXED++))
                fi
            else
                # Manual fixes with sed
                # Fix missing whitespace after colon
                sed -i 's/\([{,]\)\([^ ]\)/\1 \2/g' "$file"
                # Fix E713 (not in vs. !=)
                sed -i 's/\(if\|while\) \([^ ]*\) != \(.*\):/\1 \2 not in \3:/g' "$file"
                rm "$file.bak"
                ((PYTHON_FIXED++))
            fi
        fi
    done < /tmp/python_fix_files.tmp

    rm -f /tmp/python_fix_files.tmp
    echo -e "${GREEN}✅ Fixed style issues in $PYTHON_FIXED Python files${NC}"

    return $PYTHON_FIXED
}

# Function to stage changes with confirmation
stage_changes() {
    echo -e "\n${BLUE}📦 Staging fixed files...${NC}"

    if [ "$AUTO_STAGE" = true ]; then
        echo -e "${YELLOW}Auto-staging enabled. Staging all changes...${NC}"
        STAGE_CHANGES="y"
    else
        read -p "Do you want to stage the whitespace fixes? (y/N): " STAGE_CHANGES
    fi

    if [[ "$STAGE_CHANGES" =~ ^[Yy]$ ]]; then
        git add -A
        echo -e "${GREEN}✅ Changes staged${NC}"

        # Show what's staged
        echo -e "${BLUE}📋 Staged changes:${NC}"
        git diff --cached --name-only | head -10
        TOTAL_FILES=$(git diff --cached --name-only | wc -l)
        if [ "$TOTAL_FILES" -gt 10 ]; then
            echo -e "${YELLOW}... and $(($TOTAL_FILES - 10)) more files${NC}"
        fi

        # Check for unstaged changes
        if ! git diff --quiet; then
            echo -e "${YELLOW}⚠️  There are still unstaged changes not fixed by this script${NC}"
            echo -e "   Run 'git status' to see remaining issues"
        fi

        return 0
    else
        echo -e "${YELLOW}Changes not staged. You can review and stage them manually.${NC}"
        return 1
    fi
}

# Main execution
FIXED_COUNT=0

# Fix whitespace issues
fix_whitespace
WHITESPACE_FIXED=$?

# Fix Python issues if not skipped
if [ "$SKIP_PYTHON_FIX" != true ]; then
    fix_python_issues
    PYTHON_FIXED=$?
else
    PYTHON_FIXED=0
fi

# Total fixed count
FIXED_COUNT=$((WHITESPACE_FIXED + PYTHON_FIXED))

# Stage changes if files were fixed
if [ $FIXED_COUNT -gt 0 ]; then
    stage_changes
    STAGED=$?
else
    echo -e "${GREEN}✅ No issues found, nothing to stage${NC}"
    STAGED=0
fi

echo -e "\n${PURPLE}🎉 Fix and stage operation completed!${NC}"
echo -e "${BLUE}📊 Final Summary:${NC}"
echo -e "  - Whitespace issues fixed: $WHITESPACE_FIXED"
echo -e "  - Python issues fixed: $PYTHON_FIXED"
echo -e "  - Total files fixed: $FIXED_COUNT"
echo -e "  - Changes staged: $([ "$STAGED" -eq 0 ] && echo "Yes" || echo "No")"

echo -e "\n${BLUE}🔗 Next steps:${NC}"
if [ "$STAGED" -eq 0 ]; then
    echo -e "1. Commit your changes: git commit -m \"Fix whitespace and style issues\""
    echo -e "2. Push your changes: git push"
else
    echo -e "1. Review changes: git diff"
    echo -e "2. Stage changes manually: git add -A"
    echo -e "3. Commit your changes: git commit -m \"Fix whitespace and style issues\""
    echo -e "4. Push your changes: git push"
fi

echo -e "\n${GREEN}💡 TIP: To fix whitespace AND automatically stage in one step:${NC}"
echo -e "    ${YELLOW}./scripts/git-tools/fix_whitespace.sh --auto-stage${NC}"
