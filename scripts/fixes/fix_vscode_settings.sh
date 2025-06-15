#!/bin/bash
# Fix VS Code Settings Issues
# Resolves user settings errors and warnings

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[0;33m'
BLUE='\033[0;34m'
PURPLE='\033[0;35m'
NC='\033[0m' # No Color

echo -e "${PURPLE}🔧 VS Code Settings Fix Script${NC}"
echo -e "${BLUE}================================${NC}"

# Function to find VS Code config directory
find_vscode_config() {
    local possible_paths=(
        "$HOME/.config/Code/User"
        "$HOME/.vscode/User"
        "$HOME/Library/Application Support/Code/User"  # macOS
        "$APPDATA/Code/User"  # Windows
        "$HOME/.config/VSCodium/User"  # VSCodium
    )

    for path in "${possible_paths[@]}"; do
        if [ -d "$path" ]; then
            echo "$path"
            return 0
        fi
    done

    return 1
}

# Function to backup existing settings
backup_settings() {
    local settings_dir="$1"
    local settings_file="$settings_dir/settings.json"

    if [ -f "$settings_file" ]; then
        echo -e "${YELLOW}📦 Backing up existing settings...${NC}"
        cp "$settings_file" "$settings_file.backup.$(date +%Y%m%d_%H%M%S)"
        echo -e "${GREEN}✅ Backup created${NC}"
    fi
}

# Function to validate JSON
validate_json() {
    local file="$1"

    if command -v python3 &> /dev/null; then
        python3 -c "import json; json.load(open('$file'))" 2>/dev/null
        return $?
    elif command -v node &> /dev/null; then
        node -e "JSON.parse(require('fs').readFileSync('$file', 'utf8'))" 2>/dev/null
        return $?
    else
        # Basic validation - check for obvious syntax errors
        if grep -q "^[[:space:]]*{" "$file" && grep -q "}[[:space:]]*$" "$file"; then
            return 0
        else
            return 1
        fi
    fi
}

# Function to fix JSON syntax
fix_json_syntax() {
    local file="$1"
    local temp_file="/tmp/vscode_settings_fixed.json"

    echo -e "${BLUE}🔧 Fixing JSON syntax errors...${NC}"

    # Remove trailing commas and fix common issues
    sed -e 's/,\s*}/}/g' \
        -e 's/,\s*]/]/g' \
        -e 's/\/\/.*$//g' \
        -e '/^[[:space:]]*$/d' \
        "$file" > "$temp_file"

    # Validate the fixed file
    if validate_json "$temp_file"; then
        mv "$temp_file" "$file"
        echo -e "${GREEN}✅ JSON syntax fixed${NC}"
        return 0
    else
        rm -f "$temp_file"
        echo -e "${RED}❌ Could not automatically fix JSON syntax${NC}"
        return 1
    fi
}

# Function to create clean settings
create_clean_settings() {
    local settings_file="$1"

    echo -e "${BLUE}📝 Creating clean VS Code settings...${NC}"

    cat > "$settings_file" << 'EOF'
{
    "python.defaultInterpreterPath": "./venv/bin/python",
    "python.linting.enabled": true,
    "python.linting.pylintEnabled": true,
    "python.linting.flake8Enabled": true,
    "python.linting.mypyEnabled": true,
    "python.formatting.provider": "black",
    "python.formatting.blackArgs": ["--line-length=88"],
    "python.sortImports.args": ["--profile", "black"],
    "editor.formatOnSave": true,
    "editor.codeActionsOnSave": {
        "source.organizeImports": true
    },
    "files.exclude": {
        "**/__pycache__": true,
        "**/*.pyc": true,
        "**/venv": true,
        "**/.venv": true,
        "**/.pytest_cache": true,
        "**/.mypy_cache": true,
        "**/models/legacy/MODELS": true
    },
    "search.exclude": {
        "**/venv": true,
        "**/.venv": true,
        "**/models/legacy": true
    },
    "python.analysis.autoImportCompletions": true,
    "python.analysis.autoSearchPaths": true,
    "python.analysis.diagnosticMode": "workspace",
    "python.analysis.typeCheckingMode": "basic",
    "git.ignoreLimitWarning": true,
    "workbench.colorTheme": "Default Dark+",
    "terminal.integrated.defaultProfile.linux": "bash"
}
EOF

    echo -e "${GREEN}✅ Clean settings created${NC}"
}

# Function to fix workspace settings
fix_workspace_settings() {
    local workspace_settings=".vscode/settings.json"

    echo -e "${BLUE}🏢 Checking workspace settings...${NC}"

    if [ ! -d ".vscode" ]; then
        mkdir -p ".vscode"
        echo -e "${GREEN}✅ Created .vscode directory${NC}"
    fi

    if [ -f "$workspace_settings" ]; then
        if ! validate_json "$workspace_settings"; then
            echo -e "${YELLOW}⚠️  Workspace settings have JSON errors${NC}"
            backup_settings ".vscode"

            if ! fix_json_syntax "$workspace_settings"; then
                create_clean_workspace_settings
            fi
        else
            echo -e "${GREEN}✅ Workspace settings are valid${NC}"
        fi
    else
        create_clean_workspace_settings
    fi
}

# Function to create clean workspace settings
create_clean_workspace_settings() {
    local workspace_settings=".vscode/settings.json"

    echo -e "${BLUE}📝 Creating clean workspace settings...${NC}"

    cat > "$workspace_settings" << 'EOF'
{
    "python.defaultInterpreterPath": "./venv/bin/python",
    "python.analysis.extraPaths": ["./src"],
    "python.testing.pytestEnabled": true,
    "python.testing.pytestArgs": ["tests"],
    "python.testing.unittestEnabled": false,
    "mypy.dmypyExecutable": "./venv/bin/dmypy",
    "files.associations": {
        "*.py": "python"
    },
    "python.linting.pylintArgs": [
        "--load-plugins=pylint_django",
        "--django-settings-module=myproject.settings"
    ]
}
EOF

    echo -e "${GREEN}✅ Clean workspace settings created${NC}"
}

# Function to fix file permissions
fix_permissions() {
    local settings_dir="$1"

    echo -e "${BLUE}🔐 Fixing file permissions...${NC}"

    # Make sure the directory exists and is writable
    if [ -d "$settings_dir" ]; then
        chmod 755 "$settings_dir"
        if [ -f "$settings_dir/settings.json" ]; then
            chmod 644 "$settings_dir/settings.json"
        fi
        echo -e "${GREEN}✅ Permissions fixed${NC}"
    fi
}

# Function to clear VS Code cache
clear_vscode_cache() {
    echo -e "${BLUE}🗑️  Clearing VS Code cache...${NC}"

    local cache_paths=(
        "$HOME/.config/Code/CachedExtensions"
        "$HOME/.config/Code/logs"
        "$HOME/.config/Code/CachedData"
        "$HOME/.vscode/extensions/.obsolete"
    )

    for cache_path in "${cache_paths[@]}"; do
        if [ -d "$cache_path" ]; then
            rm -rf "$cache_path"
            echo -e "${YELLOW}Cleared: $cache_path${NC}"
        fi
    done

    echo -e "${GREEN}✅ Cache cleared${NC}"
}

# Main execution
main() {
    echo -e "${BLUE}🔍 Detecting VS Code configuration...${NC}"

    # Find VS Code config directory
    VSCODE_CONFIG_DIR=$(find_vscode_config)

    if [ $? -ne 0 ]; then
        echo -e "${RED}❌ Could not find VS Code configuration directory${NC}"
        echo -e "${YELLOW}💡 Please ensure VS Code is installed${NC}"
        exit 1
    fi

    echo -e "${GREEN}✅ Found VS Code config: $VSCODE_CONFIG_DIR${NC}"

    # Create directory if it doesn't exist
    if [ ! -d "$VSCODE_CONFIG_DIR" ]; then
        mkdir -p "$VSCODE_CONFIG_DIR"
        echo -e "${GREEN}✅ Created config directory${NC}"
    fi

    local settings_file="$VSCODE_CONFIG_DIR/settings.json"

    # Fix permissions first
    fix_permissions "$VSCODE_CONFIG_DIR"

    # Check if settings file exists and is valid
    if [ -f "$settings_file" ]; then
        echo -e "${BLUE}📋 Checking existing settings...${NC}"

        if validate_json "$settings_file"; then
            echo -e "${GREEN}✅ User settings are valid${NC}"
        else
            echo -e "${YELLOW}⚠️  User settings have JSON errors${NC}"
            backup_settings "$VSCODE_CONFIG_DIR"

            if ! fix_json_syntax "$settings_file"; then
                echo -e "${YELLOW}Creating new clean settings...${NC}"
                create_clean_settings "$settings_file"
            fi
        fi
    else
        echo -e "${YELLOW}📝 No user settings found, creating new ones...${NC}"
        create_clean_settings "$settings_file"
    fi

    # Fix workspace settings
    fix_workspace_settings

    # Clear cache to resolve any lingering issues
    clear_vscode_cache

    # Final validation
    echo -e "\n${BLUE}🔍 Final validation...${NC}"

    if validate_json "$settings_file" && validate_json ".vscode/settings.json"; then
        echo -e "${GREEN}✅ All settings are now valid${NC}"
    else
        echo -e "${RED}❌ Some settings still have issues${NC}"
        exit 1
    fi

    echo -e "\n${PURPLE}🎉 VS Code settings fix completed!${NC}"
    echo -e "${BLUE}📋 Summary:${NC}"
    echo -e "✅ User settings: $settings_file"
    echo -e "✅ Workspace settings: .vscode/settings.json"
    echo -e "✅ Permissions fixed"
    echo -e "✅ Cache cleared"

    echo -e "\n${BLUE}🔄 Next steps:${NC}"
    echo -e "1. Restart VS Code"
    echo -e "2. Reload the Python extension"
    echo -e "3. Select Python interpreter: Ctrl+Shift+P -> Python: Select Interpreter"
    echo -e "4. Choose: ./venv/bin/python"
}

# Run the main function
main "$@"
