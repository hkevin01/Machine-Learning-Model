#!/bin/bash
# Fix Whitespace Issues for Clean Git Commits
# Removes trailing whitespace, normalizes line endings, and ensures files end with newline

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

echo -e "${PURPLE}🔧 Whitespace Fix Script${NC}"
echo -e "${BLUE}========================${NC}"
echo -e "${BLUE}⏰ Started at: $(date)${NC}"

# Check if we're in a git repository
if ! git rev-parse --is-inside-work-tree &>/dev/null; then
    echo -e "${RED}❌ Error: Not inside a git repository${NC}"
    exit 1
fi

echo -e "${BLUE}📍 Current directory: $(pwd)${NC}"

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

    # Fix trailing whitespace in code files
    echo -e "${YELLOW}Removing trailing whitespace in code files...${NC}"
    find . -type f \( -name "*.py" -o -name "*.sh" -o -name "*.yaml" -o -name "*.yml" \
        -o -name "*.json" -o -name "*.txt" -o -name "*.html" -o -name "*.css" -o -name "*.js" \
        -o -name "*.c" -o -name "*.cpp" -o -name "*.h" -o -name "*.java" \) \
        -not -path "./.git/*" -not -path "./venv/*" -not -path "./.venv/*" \
        -not -path "./models/*" -not -path "./data/*" \
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
            echo -e "  ${YELLOW}Skipped:${NC} $file"
            ((SKIPPED_COUNT++))
        fi
    done < /tmp/whitespace_fix_files.tmp

    # Handle Markdown files separately (preserve intentional trailing spaces)
    echo -e "${YELLOW}\nFixing whitespace in Markdown files (preserving line breaks)...${NC}"
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
            fi
        ' sh {} \;

    # Ensure files end with newline
    echo -e "${YELLOW}\nEnsuring files end with newline...${NC}"
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
            fi
        ' sh {} \;

    # Clean up temporary files
    rm -f /tmp/whitespace_fix_files.tmp

    echo -e "\n${GREEN}✅ Whitespace issues fixed${NC}"
    echo -e "${BLUE}📊 Statistics:${NC}"
    echo -e "  - Files fixed: $FIXED_COUNT"
    echo -e "  - Files skipped: $SKIPPED_COUNT"
    echo -e "  - Errors: $ERROR_COUNT"
}

# Optionally stage the changes
stage_changes() {
    echo -e "\n${BLUE}📦 Staging fixed files...${NC}"

    read -p "Do you want to stage the whitespace fixes? (y/N): " STAGE_CHANGES

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
    else
        echo -e "${YELLOW}Changes not staged. You can review and stage them manually.${NC}"
    fi
}

# Main execution
fix_whitespace
stage_changes

echo -e "\n${PURPLE}🎉 Whitespace fix completed!${NC}"
echo -e "${BLUE}🔗 Next steps:${NC}"
echo -e "1. Review changes: git diff"
echo -e "2. Stage if not already: git add -A"
echo -e "3. Commit: git commit -m \"Fix whitespace issues\""
echo -e "4. Push: git push"
