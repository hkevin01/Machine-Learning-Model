#!/bin/bash
# Fix Python Syntax Errors
# Fixes specific syntax errors in Python files based on common patterns

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

echo -e "${PURPLE}🔧 Python Syntax Error Fix Script${NC}"
echo -e "${BLUE}==============================${NC}"

# Fix CLI file syntax
if [ -f "src/machine_learning_model/cli.py" ]; then
    echo -e "${BLUE}🔧 Fixing CLI file...${NC}"
    cp src/machine_learning_model/cli.py src/machine_learning_model/cli.py.bak

    # Remove unmatched parenthesis
    sed -i 's/^[[:space:]]*):$//' src/machine_learning_model/cli.py
    sed -i 's/def version_callback():value: bool):/def version_callback(value: bool):/g' src/machine_learning_model/cli.py

    # Check if fixed
    if python3 -c "import ast; ast.parse(open('src/machine_learning_model/cli.py').read())" 2>/dev/null; then
        echo -e "${GREEN}✅ CLI file fixed${NC}"
        git add src/machine_learning_model/cli.py
    else
        echo -e "${YELLOW}⚠️ CLI file still has syntax errors${NC}"
    fi
fi

# Fix validators.py
if [ -f "src/machine_learning_model/data/validators.py" ]; then
    echo -e "${BLUE}🔧 Fixing validators.py...${NC}"
    cp src/machine_learning_model/data/validators.py src/machine_learning_model/data/validators.py.bak

    # Fix tuple annotation issue
    sed -i 's/^[[:space:]]*self, df: pd.DataFrame, expected_types: Dict\[str, str\]/    def validate_data_types(self, df: pd.DataFrame, expected_types: Dict[str, str])/' src/machine_learning_model/data/validators.py
    sed -i 's/) -> Dict\[str, Any\]:/def validate_data_types(self, df: pd.DataFrame, expected_types: Dict[str, str]) -> Dict[str, Any]:/g' src/machine_learning_model/data/validators.py

    # Check if fixed
    if python3 -c "import ast; ast.parse(open('src/machine_learning_model/data/validators.py').read())" 2>/dev/null; then
        echo -e "${GREEN}✅ validators.py fixed${NC}"
        git add src/machine_learning_model/data/validators.py
    else
        echo -e "${YELLOW}⚠️ validators.py still has syntax errors${NC}"
    fi
fi

# Fix preprocessors.py
if [ -f "src/machine_learning_model/data/preprocessors.py" ]; then
    echo -e "${BLUE}🔧 Fixing preprocessors.py...${NC}"
    cp src/machine_learning_model/data/preprocessors.py src/machine_learning_model/data/preprocessors.py.bak

    # Fix unmatched parenthesis and function definitions
    sed -i 's/^[[:space:]]*) -> pd.DataFrame:/    def preprocess(self, df: pd.DataFrame, strategy: str = "mean") -> pd.DataFrame:/' src/machine_learning_model/data/preprocessors.py
    sed -i 's/^[[:space:]]*self, df: pd.DataFrame, strategy: str = "mean"/    def preprocess(self, df: pd.DataFrame, strategy: str = "mean")/' src/machine_learning_model/data/preprocessors.py
    sed -i 's/def __init__():self):/def __init__(self):/g' src/machine_learning_model/data/preprocessors.py

    # Check if fixed
    if python3 -c "import ast; ast.parse(open('src/machine_learning_model/data/preprocessors.py').read())" 2>/dev/null; then
        echo -e "${GREEN}✅ preprocessors.py fixed${NC}"
        git add src/machine_learning_model/data/preprocessors.py
    else
        echo -e "${YELLOW}⚠️ preprocessors.py still has syntax errors${NC}"
    fi
fi

# Check final status
ERRORS=0
for file in src/machine_learning_model/cli.py src/machine_learning_model/data/validators.py src/machine_learning_model/data/preprocessors.py; do
    if [ -f "$file" ]; then
        if ! python3 -c "import ast; ast.parse(open('$file').read())" 2>/dev/null; then
            echo -e "${RED}❌ $file still has syntax errors${NC}"
            ((ERRORS++))
        fi
    fi
done

if [ $ERRORS -eq 0 ]; then
    echo -e "${GREEN}✅ All syntax errors fixed!${NC}"
    exit 0
else
    echo -e "${YELLOW}⚠️ $ERRORS files still have syntax errors${NC}"
    exit 1
fi

# Make this script executable
chmod +x "$0"
