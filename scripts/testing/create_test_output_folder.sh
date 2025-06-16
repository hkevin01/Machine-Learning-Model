#!/bin/bash
# Script to create and initialize test-output folder structure

# Colors
GREEN='\033[0;32m'
BLUE='\033[0;34m'
PURPLE='\033[0;35m'
NC='\033[0m'

echo -e "${PURPLE}🔧 Creating test-output folder structure${NC}"
echo -e "${BLUE}==============================${NC}"

# Create test-output directory and subdirectories
mkdir -p test-output/{logs,reports,coverage,artifacts}

# Create .gitkeep files to ensure directories are tracked
touch test-output/.gitkeep
touch test-output/logs/.gitkeep
touch test-output/reports/.gitkeep
touch test-output/coverage/.gitkeep
touch test-output/artifacts/.gitkeep

# Create README for the test-output folder
cat > test-output/README.md << 'EOF'
# Test Output Directory

This directory contains logs and output from various test runs and development tools.

## Structure

- `logs/` - Log files from test runs, applications, and tools
- `reports/` - Test reports, coverage reports, and analysis outputs
- `coverage/` - Code coverage reports and data
- `artifacts/` - Build artifacts, generated files, and temporary outputs

## Usage

This folder is automatically populated by:
- Test runners (pytest, unittest)
- Coverage tools
- Linting and formatting tools
- Build processes
- Development scripts

Files in this directory are typically temporary and can be safely deleted.
EOF

echo -e "${GREEN}✅ Test-output folder structure created${NC}"
echo -e "${BLUE}📁 Created directories:${NC}"
echo -e "  - test-output/"
echo -e "  - test-output/logs/"
echo -e "  - test-output/reports/"
echo -e "  - test-output/coverage/"
echo -e "  - test-output/artifacts/"
echo -e "${GREEN}✅ Ready for storing test logs and output${NC}"
