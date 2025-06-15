#!/bin/bash
# Fix MyPy Daemon Executable Issue (Shell Version)
# This script resolves the warning:
# "The mypy daemon executable ('dmypy') was not found on your PATH."

set -e

echo "🔧 MyPy Daemon Fix Script (Shell Version)"
echo "========================================="

# Step 1: Detect or create virtual environment
VENV_DIR="venv"
if [ -d "$VENV_DIR" ]; then
    echo "✅ Found virtual environment: $VENV_DIR"
else
    echo "🐍 Creating new virtual environment: $VENV_DIR"
    python3 -m venv "$VENV_DIR"
fi

if [ -f "$VENV_DIR/bin/activate" ]; then
    ACTIVATE_SCRIPT="$VENV_DIR/bin/activate"
    PYTHON="$VENV_DIR/bin/python"
    PIP="$VENV_DIR/bin/pip"
    DMYPY="$VENV_DIR/bin/dmypy"
else
    ACTIVATE_SCRIPT="$VENV_DIR/Scripts/activate"
    PYTHON="$VENV_DIR/Scripts/python.exe"
    PIP="$VENV_DIR/Scripts/pip.exe"
    DMYPY="$VENV_DIR/Scripts/dmypy.exe"
fi

echo "📦 Upgrading pip and installing/upgrading mypy..."
"$PYTHON" -m pip install --upgrade pip
"$PIP" install --upgrade mypy

# Step 2: Verify dmypy is accessible
if [ -x "$DMYPY" ]; then
    echo "✅ dmypy found at: $DMYPY"
else
    echo "❌ dmypy not found in venv. Something went wrong."
    exit 1
fi

# Step 3: Create VS Code settings if needed
VSCODE_DIR=".vscode"
SETTINGS_FILE="$VSCODE_DIR/settings.json"
mkdir -p "$VSCODE_DIR"

cat > "$SETTINGS_FILE" <<EOF
{
  "python.linting.mypyEnabled": true,
  "python.linting.enabled": true,
  "mypy.dmypyExecutable": "$DMYPY",
  "python.defaultInterpreterPath": "$PYTHON"
}
EOF
echo "✅ VS Code settings updated: $SETTINGS_FILE"

# Step 4: Create mypy.ini config if not present
if [ ! -f "mypy.ini" ]; then
cat > mypy.ini <<EOF
[mypy]
python_version = 3.8
warn_return_any = True
warn_unused_configs = True
disallow_untyped_defs = True
disallow_incomplete_defs = True
check_untyped_defs = True
disallow_untyped_decorators = True
no_implicit_optional = True
warn_redundant_casts = True
warn_unused_ignores = True
warn_no_return = True
warn_unreachable = True
strict_equality = True

[mypy-tests.*]
disallow_untyped_defs = False

[mypy-setup]
ignore_errors = True

[mypy-numpy.*]
ignore_missing_imports = True

[mypy-pandas.*]
ignore_missing_imports = True

[mypy-matplotlib.*]
ignore_missing_imports = True

[mypy-sklearn.*]
ignore_missing_imports = True

[mypy-scipy.*]
ignore_missing_imports = True
EOF
    echo "✅ Created mypy.ini configuration file"
else
    echo "ℹ️  mypy.ini already exists"
fi

# Step 5: Add mypy to requirements-dev.txt if not present
REQ_FILE="requirements-dev.txt"
if [ -f "$REQ_FILE" ]; then
    if ! grep -q "mypy" "$REQ_FILE"; then
        echo -e "\n# Type checking\nmypy>=1.0.0" >> "$REQ_FILE"
        echo "✅ Added mypy to $REQ_FILE"
    else
        echo "✅ mypy already in $REQ_FILE"
    fi
else
    echo -e "# Development dependencies\nmypy>=1.0.0\npytest>=6.0.0\nblack>=21.0.0\nflake8>=3.9.0" > "$REQ_FILE"
    echo "✅ Created $REQ_FILE with mypy"
fi

# Step 6: Test dmypy
echo "🧪 Testing dmypy..."
source "$ACTIVATE_SCRIPT"
dmypy --help >/dev/null 2>&1 && echo "✅ dmypy is working correctly" || echo "⚠️  dmypy test failed"

echo
echo "🎉 MyPy daemon fix completed!"
echo "🔄 Next steps:"
echo "1. source $ACTIVATE_SCRIPT"
echo "2. Restart VS Code"
echo "3. Try running: dmypy --help"
echo "4. Enjoy type checking with mypy daemon!"
