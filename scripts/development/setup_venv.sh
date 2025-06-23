#!/bin/bash
# Script to check for and set up a virtual environment for the project

# Enable debugging and show command execution
set -x

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[0;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Force output to terminal
exec > >(tee /dev/tty)
exec 2>&1

echo "Starting script execution..."
echo -e "${BLUE}🔧 Setting up virtual environment for the project...${NC}"

# Get current working directory (project root)
PROJECT_DIR="$(pwd)"
VENV_DIR="$PROJECT_DIR/venv"
REQUIREMENTS_FILE="$PROJECT_DIR/requirements.txt"

echo "Project directory: $PROJECT_DIR"
echo "Venv directory: $VENV_DIR"
echo -e "${BLUE}📍 Project directory: $PROJECT_DIR${NC}"
echo -e "${BLUE}📍 Virtual environment will be created at: $VENV_DIR${NC}"

# Check if script is executable
if [ ! -x "$0" ]; then
    echo "Making script executable..."
    chmod +x "$0"
fi

# Check if venv exists
if [ -d "$VENV_DIR" ]; then
    echo -e "${GREEN}✅ Virtual environment already exists at $VENV_DIR${NC}"
    echo "Listing venv contents:"
    ls -la "$VENV_DIR"
else
    # Create virtual environment
    echo -e "${BLUE}📦 Creating virtual environment...${NC}"
    echo "Running: python3 -m venv $VENV_DIR"
    python3 -m venv "$VENV_DIR" 2>&1
    if [ $? -eq 0 ]; then
        echo -e "${GREEN}✅ Virtual environment created at $VENV_DIR${NC}"
        echo "Listing created venv:"
        ls -la "$VENV_DIR"
    else
        echo -e "${RED}❌ Failed to create virtual environment${NC}"
        echo "Python3 version:"
        python3 --version
        exit 1
    fi
fi

# Activate virtual environment
echo -e "${BLUE}🚀 Activating virtual environment...${NC}"
source "$VENV_DIR/bin/activate"

# Verify activation
if [ "$VIRTUAL_ENV" != "" ]; then
    echo -e "${GREEN}✅ Virtual environment activated: $VIRTUAL_ENV${NC}"
else
    echo -e "${RED}❌ Failed to activate virtual environment${NC}"
    exit 1
fi

# Upgrade pip first
echo -e "${BLUE}📦 Upgrading pip...${NC}"
pip install --upgrade pip
echo -e "${GREEN}✅ Pip upgraded${NC}"

# Install pytest and other essential packages
echo -e "${BLUE}📦 Installing essential packages (pytest, pandas)...${NC}"
pip install pytest pandas
echo -e "${GREEN}✅ Essential packages installed${NC}"

# Install requirements if file exists
if [ -f "$REQUIREMENTS_FILE" ]; then
    echo -e "${BLUE}📦 Installing requirements from $REQUIREMENTS_FILE...${NC}"
    pip install -r "$REQUIREMENTS_FILE"
    if [ $? -eq 0 ]; then
        echo -e "${GREEN}✅ Requirements installed successfully!${NC}"
    else
        echo -e "${YELLOW}⚠️  Some requirements failed to install${NC}"
    fi
else
    echo -e "${YELLOW}⚠️  No requirements.txt file found. Creating basic one...${NC}"
    cat > "$REQUIREMENTS_FILE" << 'EOF'
# Core dependencies
numpy>=1.21.0
pandas>=1.3.0
pytest>=7.4.0
pytest-cov>=4.1.0
black>=23.0.0
isort>=5.12.0
flake8>=6.0.0
mypy>=1.5.0
pre-commit>=3.3.0
EOF
    echo -e "${GREEN}✅ Basic requirements.txt created${NC}"
    pip install -r "$REQUIREMENTS_FILE"
    echo -e "${GREEN}✅ Basic requirements installed${NC}"
fi

# Install development dependencies
echo "Installing development dependencies..."
pip install -r requirements-dev.txt

# Install additional development tools
pip install \
    pytest>=7.0.0 \
    pytest-cov>=4.0.0 \
    black>=23.0.0 \
    isort>=5.12.0 \
    flake8>=6.0.0 \
    pre-commit>=3.0.0

# List installed packages
echo -e "${BLUE}📋 Installed packages:${NC}"
pip list

# Show how to activate venv
echo -e "\n${BLUE}💡 To activate virtual environment in the future:${NC}"
echo -e "${YELLOW}source venv/bin/activate${NC}"

# Show how to run tests
echo -e "\n${BLUE}💡 To run tests:${NC}"
echo -e "${YELLOW}source venv/bin/activate${NC}"
echo -e "${YELLOW}pytest tests/test_data/test_loaders.py -v${NC}"

# Deactivate virtual environment
deactivate
echo -e "${GREEN}✅ Virtual environment setup complete!${NC}"

# Add final confirmation
echo "Script completed successfully!"
echo "Final venv check:"
if [ -d "$VENV_DIR" ]; then
    echo "✅ Virtual environment exists at $VENV_DIR"
    echo "Contents:"
    ls -la "$VENV_DIR/"
else
    echo "❌ Virtual environment not found"
fi

# Fix VS Code settings to show venv folder
echo "🔧 Configuring VS Code to show venv folder..."
VSCODE_SETTINGS_DIR="$PROJECT_DIR/.vscode"
VSCODE_SETTINGS_FILE="$VSCODE_SETTINGS_DIR/settings.json"

# Create .vscode directory if it doesn't exist
mkdir -p "$VSCODE_SETTINGS_DIR"

# Create or update VS Code settings to ensure venv is visible
cat > "$VSCODE_SETTINGS_FILE" << 'EOF'
{
    "python.linting.enabled": true,
    "python.linting.flake8Enabled": true,
    "python.linting.flake8Args": [
        "--max-line-length=88",
        "--extend-ignore=E203,W503"
    ],
    "python.formatting.provider": "black",
    "python.sortImports.args": [
        "--profile=black"
    ],
    "editor.formatOnSave": true,
    "editor.codeActionsOnSave": {
        "source.organizeImports": true
    },
    "files.exclude": {
        "**/__pycache__": true,
        "**/*.pyc": true,
        "**/.pytest_cache": true,
        "**/.coverage": true,
        "**/htmlcov": true,
        "**/.mypy_cache": true,
        "**/venv": true,
        "**/.venv": true,
        "**/node_modules": true,
        "**/.git": true
    }
}
EOF

echo "✅ VS Code settings updated to show venv folder"
echo ""
echo "📋 VS Code Configuration Steps:"
echo "1. Restart VS Code completely (close and reopen)"
echo "2. Press F5 or Ctrl+R to refresh the file explorer"
echo "3. Check if 'venv' folder now appears in the explorer"
echo "4. If still not visible, try View → Command Palette → 'Developer: Reload Window'"
echo ""
echo "🔧 Manual VS Code Setup (if needed):"
echo "1. Press Ctrl+Shift+P (Command Palette)"
echo "2. Type 'Python: Select Interpreter'"
echo "3. Choose './venv/bin/python'"
echo ""
echo "💡 The venv folder should now be visible in VS Code Explorer!"
