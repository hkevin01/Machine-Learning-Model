#!/bin/bash
# Generic Python Project Structure Generator (Shell Version)
# Creates a standard Python project structure for ML projects

set -e

PROJECT_NAME="${1:-$(basename "$PWD")}"
BASE_PATH="${2:-.}"
PACKAGE_NAME=$(echo "$PROJECT_NAME" | tr '[:upper:]' '[:lower:]' | tr ' -' '__')

echo "Creating Python project: $PROJECT_NAME"
echo "Location: $(realpath "$BASE_PATH/$PROJECT_NAME")"

PROJECT_PATH="$BASE_PATH/$PROJECT_NAME"
if [ "$PROJECT_NAME" = "$(basename "$PWD")" ] && [ "$BASE_PATH" = "." ]; then
    PROJECT_PATH="$PWD"
    echo "Using current directory as project root: $PROJECT_NAME"
else
    mkdir -p "$PROJECT_PATH"
fi

# Remove duplicate/legacy model folders
for d in model Model MODEL Models MODELS "#file:MODELS" "file:MODELS" "#file_MODELS" "#file-models" "#MODELS"; do
    if [ -d "$PROJECT_PATH/$d" ] && [ "$d" != "models" ]; then
        echo "Backing up and removing duplicate folder: $d"
        mkdir -p "$PROJECT_PATH/models/legacy/$d"
        mv "$PROJECT_PATH/$d"/* "$PROJECT_PATH/models/legacy/$d/" 2>/dev/null || true
        rm -rf "$PROJECT_PATH/$d"
    fi
done

# Directory structure
DIRS=(
    "src/$PACKAGE_NAME"
    "tests"
    "docs"
    "scripts"
    "logs"
    ".github/workflows"
    ".copilot"
    "data/raw"
    "data/processed"
    "data/interim"
    "data/external"
    "data/features"
    "models/trained"
    "models/checkpoints"
    "models/experiments"
    "models/metadata"
    "models/legacy"
    "notebooks"
    "config"
    "examples"
    "assets"
)
for d in "${DIRS[@]}"; do
    mkdir -p "$PROJECT_PATH/$d"
done

# Touch basic files
touch "$PROJECT_PATH/README.md"
touch "$PROJECT_PATH/requirements.txt"
touch "$PROJECT_PATH/requirements-dev.txt"
touch "$PROJECT_PATH/pyproject.toml"
touch "$PROJECT_PATH/.pre-commit-config.yaml"
touch "$PROJECT_PATH/Makefile"
touch "$PROJECT_PATH/CHANGELOG.md"
touch "$PROJECT_PATH/CONTRIBUTING.md"
touch "$PROJECT_PATH/Dockerfile"
touch "$PROJECT_PATH/docker-compose.yml"
touch "$PROJECT_PATH/.gitignore"
touch "$PROJECT_PATH/.env.example"
touch "$PROJECT_PATH/LICENSE"
touch "$PROJECT_PATH/scripts/README.md"
touch "$PROJECT_PATH/data/README.md"
touch "$PROJECT_PATH/models/README.md"
touch "$PROJECT_PATH/models/legacy/README.md"
touch "$PROJECT_PATH/notebooks/README.md"
touch "$PROJECT_PATH/config/README.md"
touch "$PROJECT_PATH/examples/README.md"
touch "$PROJECT_PATH/docs/README.md"
touch "$PROJECT_PATH/docs/project_structure.md"
touch "$PROJECT_PATH/docs/folder_purposes.md"
touch "$PROJECT_PATH/docs/ml_workflow.md"
touch "$PROJECT_PATH/tests/__init__.py"
touch "$PROJECT_PATH/tests/conftest.py"
touch "$PROJECT_PATH/.copilot/config.json"
touch "$PROJECT_PATH/.github/workflows/ci.yml"
touch "$PROJECT_PATH/src/$PACKAGE_NAME/__init__.py"
touch "$PROJECT_PATH/src/$PACKAGE_NAME/main.py"
touch "$PROJECT_PATH/src/$PACKAGE_NAME/cli.py"
touch "$PROJECT_PATH/tests/test_${PACKAGE_NAME}.py"

echo "✅ Successfully created Python project: $PROJECT_NAME"
echo "📁 Location: $(realpath "$PROJECT_PATH")"
echo "🚀 Next steps:"
echo "1. cd \"$PROJECT_PATH\""
echo "2. python -m venv venv"
echo "3. source venv/bin/activate"
echo "4. pip install -r requirements.txt"
echo "5. make install-dev  # or pip install -e '.[dev]'"
echo "6. pre-commit install"
echo "7. git init && git add . && git commit -m 'Initial commit'"
echo "8. make test        # Run tests"
echo "9. make format      # Format code"
echo "10. make check      # Run all checks"
