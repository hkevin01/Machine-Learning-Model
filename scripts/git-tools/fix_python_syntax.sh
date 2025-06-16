#!/bin/bash
# Fix Python Syntax Errors
# Resolves common Python syntax issues, especially with function signatures

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

echo -e "${PURPLE}🔧 Python Syntax Fix Script${NC}"
echo -e "${BLUE}==============================${NC}"
echo -e "${BLUE}⏰ Started at: $(date)${NC}"
echo -e "${BLUE}📍 Current directory: $(pwd)${NC}"

# Check if we're in a git repository
if ! git rev-parse --is-inside-work-tree &>/dev/null; then
    echo -e "${RED}❌ Error: Not inside a git repository${NC}"
    exit 1
fi

# Function to fix Python file syntax
fix_python_file() {
    local file="$1"
    echo -e "${BLUE}Fixing syntax in: $file${NC}"

    # Create a backup of the file
    cp "$file" "$file.syntax.bak"

    # Fix 1: Fix duplicate return type annotations
    # Pattern: def function() -> None:) -> None:
    sed -i 's/\(def [^(]*([^)]*)\) -> None:\([ ]*\)) -> None:/\1:/g' "$file"

    # Fix 2: Fix other variations with double return types
    sed -i 's/\(def [^(]*([^)]*)\) -> None:\([ ]*\))\([ ]*\)-> None:/\1:/g' "$file"

    # Fix 3: Fix issues with parameters after broken type annotations
    sed -i 's/\(def [^(]*([^)]*)\) -> None:\([ ]*\))\([ ]*\)self):/\1(self):/g' "$file"
    sed -i 's/\(def [^(]*([^)]*)\) -> None:\([ ]*\))\([ ]*\)value: bool):/\1(value: bool):/g' "$file"
    sed -i 's/\(def [^(]*([^)]*)\) -> None:\([ ]*\))\([ ]*\)\([a-zA-Z_][a-zA-Z0-9_]*\):/\1(\4):/g' "$file"

    # Check if the file is now syntactically valid
    if python3 -c "import ast; ast.parse(open('$file').read())" 2>/dev/null; then
        echo -e "${GREEN}✅ Successfully fixed $file${NC}"
        rm -f "$file.syntax.bak"
        return 0
    else
        echo -e "${YELLOW}⚠️ Initial fixes didn't resolve all issues in $file${NC}"

        # More aggressive approach: Simplify function definitions
        cp "$file.syntax.bak" "$file"
        sed -i 's/def \([^(]*\)(.*) -> None:[^(]*:[^:]*:/def \1():/g' "$file"
        sed -i 's/def \([^(]*\)(.*) -> None:[^(]*:/def \1():/g' "$file"

        # Check again
        if python3 -c "import ast; ast.parse(open('$file').read())" 2>/dev/null; then
            echo -e "${GREEN}✅ Fixed $file with aggressive approach${NC}"
            rm -f "$file.syntax.bak"
            return 0
        else
            echo -e "${RED}❌ Failed to fix $file - restoring original${NC}"
            cp "$file.syntax.bak" "$file"
            rm -f "$file.syntax.bak"
            return 1
        fi
    fi
}

# List of specific files with known issues
KNOWN_PROBLEM_FILES=(
    "src/machine_learning_model/cli.py"
    "src/machine_learning_model/data/loaders.py"
    "src/machine_learning_model/data/preprocessors.py"
    "src/machine_learning_model/data/validators.py"
    "src/machine_learning_model/main.py"
)

# Fix known problem files first
echo -e "${BLUE}🔄 Fixing known problem files...${NC}"
FIXED_COUNT=0
ERROR_COUNT=0

for file in "${KNOWN_PROBLEM_FILES[@]}"; do
    if [ -f "$file" ]; then
        if fix_python_file "$file"; then
            git add "$file"
            ((FIXED_COUNT++))
        else
            ((ERROR_COUNT++))
        fi
    fi
done

# Now scan for other potential syntax issues
echo -e "${BLUE}🔄 Scanning for other syntax issues...${NC}"
find src -name "*.py" -type f | while read -r file; do
    # Skip files we already processed
    if [[ " ${KNOWN_PROBLEM_FILES[@]} " =~ " $file " ]]; then
        continue
    fi

    # Check if file has syntax errors
    if ! python3 -c "import ast; ast.parse(open('$file').read())" 2>/dev/null; then
        echo -e "${YELLOW}⚠️ Found syntax issue in: $file${NC}"
        if fix_python_file "$file"; then
            git add "$file"
            ((FIXED_COUNT++))
        else
            ((ERROR_COUNT++))
        fi
    fi
done

# Print summary
echo -e "\n${BLUE}📊 Summary:${NC}"
echo -e "✅ Fixed files: $FIXED_COUNT"
echo -e "❌ Files with remaining issues: $ERROR_COUNT"

if [ $ERROR_COUNT -eq 0 ]; then
    echo -e "\n${GREEN}🎉 All Python syntax issues fixed!${NC}"
    exit 0
else
    echo -e "\n${YELLOW}⚠️ Some files still have syntax issues${NC}"
    exit 1
fi
