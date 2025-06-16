#!/bin/bash
# Fix Python Syntax Errors
# Repairs specific syntax errors in problematic Python files

# Colors
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[0;33m'
RED='\033[0;31m'
NC='\033[0m'

echo -e "${BLUE}🔧 Python Syntax Error Fixer${NC}"
echo -e "${BLUE}==============================${NC}"

# Create backup directory
BACKUP_DIR="/tmp/python_fix_backups_$(date +%s)"
mkdir -p "$BACKUP_DIR"
echo -e "${BLUE}📁 Created backup directory: $BACKUP_DIR${NC}"

# Fix CLI File
CLI_FILE="src/machine_learning_model/cli.py"
if [ -f "$CLI_FILE" ]; then
    echo -e "${BLUE}🔧 Fixing $CLI_FILE...${NC}"
    cp "$CLI_FILE" "$BACKUP_DIR/cli.py"

    # Replace the entire app.callback function that has the syntax error
    sed -i '27,42c\
@app.callback()\
def main(\
    version: Optional[bool] = typer.Option(\
        None,\
        "--version",\
        "-v",\
        callback=version_callback,\
        is_eager=True,\
        help="Show version and exit.",\
    ),\
    verbose: bool = typer.Option(False, "--verbose", help="Enable verbose output."),\
    quiet: bool = typer.Option(False, "--quiet", help="Suppress output.")\
):\
    """\
    machine_learning_model - A Python package for [description]\
    """\
    if verbose:\
        logger.info("Verbose mode enabled")\
    if quiet:\
        logger.remove()' "$CLI_FILE"

    # Check if fixed
    if python3 -c "import ast; ast.parse(open('$CLI_FILE').read())" 2>/dev/null; then
        echo -e "${GREEN}✅ Fixed $CLI_FILE${NC}"
        git add "$CLI_FILE"
    else
        echo -e "${RED}❌ Failed to fix $CLI_FILE${NC}"
    fi
fi

# Fix Validators File
VALIDATORS_FILE="src/machine_learning_model/data/validators.py"
if [ -f "$VALIDATORS_FILE" ]; then
    echo -e "${BLUE}🔧 Fixing $VALIDATORS_FILE...${NC}"
    cp "$VALIDATORS_FILE" "$BACKUP_DIR/validators.py"

    # Replace the function definition with a properly formatted one
    sed -i '12,14c\
    def validate_data_types(self, df: pd.DataFrame, expected_types: Dict[str, str]) -> Dict[str, Any]:\
        """Validate data types of DataFrame columns."""\
        validation_results = {' "$VALIDATORS_FILE"

    # Check if fixed
    if python3 -c "import ast; ast.parse(open('$VALIDATORS_FILE').read())" 2>/dev/null; then
        echo -e "${GREEN}✅ Fixed $VALIDATORS_FILE${NC}"
        git add "$VALIDATORS_FILE"
    else
        echo -e "${RED}❌ Failed to fix $VALIDATORS_FILE${NC}"
    fi
fi

# Fix Preprocessors File
PREPROCESSORS_FILE="src/machine_learning_model/data/preprocessors.py"
if [ -f "$PREPROCESSORS_FILE" ]; then
    echo -e "${BLUE}🔧 Fixing $PREPROCESSORS_FILE...${NC}"
    cp "$PREPROCESSORS_FILE" "$BACKUP_DIR/preprocessors.py"

    # Replace the problematic function definition
    sed -i '20,22c\
    def preprocess(self, df: pd.DataFrame, strategy: str = "mean") -> pd.DataFrame:\
        """Handle missing values in the dataset."""' "$PREPROCESSORS_FILE"

    # Check if fixed
    if python3 -c "import ast; ast.parse(open('$PREPROCESSORS_FILE').read())" 2>/dev/null; then
        echo -e "${GREEN}✅ Fixed $PREPROCESSORS_FILE${NC}"
        git add "$PREPROCESSORS_FILE"
    else
        echo -e "${RED}❌ Failed to fix $PREPROCESSORS_FILE${NC}"
    fi
fi

# Check final status
ERRORS=0
for file in "$CLI_FILE" "$VALIDATORS_FILE" "$PREPROCESSORS_FILE"; do
    if [ -f "$file" ]; then
        if ! python3 -c "import ast; ast.parse(open('$file').read())" 2>/dev/null; then
            echo -e "${RED}❌ $file still has syntax errors${NC}"
            ((ERRORS++))
        fi
    fi
done

if [ $ERRORS -eq 0 ]; then
    echo -e "\n${GREEN}✅ All syntax errors fixed!${NC}"
    echo -e "${BLUE}📋 Summary:${NC}"
    echo -e "✅ Fixed CLI file"
    echo -e "✅ Fixed validators file"
    echo -e "✅ Fixed preprocessors file"
    echo -e "\n${BLUE}🔄 Next steps:${NC}"
    echo -e "1. Try committing with: git commit -m \"Fix Python syntax errors\""
    echo -e "2. If additional errors occur, check the error messages for details"
else
    echo -e "\n${YELLOW}⚠️ $ERRORS files still have syntax errors${NC}"
    echo -e "${BLUE}🔄 You may need to:${NC}"
    echo -e "1. Edit the files manually to fix remaining syntax issues"
    echo -e "2. Use emergency commit: ./scripts/git-tools/emergency_commit.sh \"Fix syntax errors\""
fi

echo -e "${BLUE}💾 Backups stored in: $BACKUP_DIR${NC}"
