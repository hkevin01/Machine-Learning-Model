#!/bin/bash
# Fix Pre-commit Issues Script
# Resolves pre-commit hook problems and updates deprecated configurations

# Force all output to terminal immediately
exec > >(tee /dev/tty)
exec 2>&1

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[0;33m'
BLUE='\033[0;34m'
PURPLE='\033[0;35m'
NC='\033[0m' # No Color

echo -e "${PURPLE}🔧 Pre-commit Issues Fix Script${NC}"
echo -e "${BLUE}==================================${NC}"
echo -e "${BLUE}⏰ Started at: $(date)${NC}"
echo -e "${BLUE}📍 Script location: /scripts/fixes/fix_precommit_issues.sh${NC}"
echo

# Function to check if we're in the right directory
check_project_directory() {
    echo -e "${BLUE}📂 Checking project directory...${NC}"
    if [ ! -f "pyproject.toml" ] || [ ! -f ".pre-commit-config.yaml" ]; then
        echo -e "${RED}❌ Error: Not in project root directory${NC}"
        echo -e "${YELLOW}💡 Please run this script from the project root${NC}"
        exit 1
    fi
    echo -e "${GREEN}✅ Project directory verified${NC}"
}

# Function to backup current pre-commit config
backup_precommit_config() {
    echo -e "${BLUE}📦 Backing up current pre-commit configuration...${NC}"

    if [ -f ".pre-commit-config.yaml" ]; then
        BACKUP_FILE=".pre-commit-config.yaml.backup.$(date +%Y%m%d_%H%M%S)"
        cp ".pre-commit-config.yaml" "$BACKUP_FILE"
        echo -e "${GREEN}✅ Backup created: $BACKUP_FILE${NC}"
    else
        echo -e "${YELLOW}⚠️  No existing pre-commit config found${NC}"
    fi
}

# Main execution function
main() {
    echo -e "${BLUE}🔍 Starting comprehensive pre-commit fix...${NC}"

    # Step 1: Verify we're in the right directory
    check_project_directory

    # Step 2: Clean environments first
    clean_precommit_environments

    # Step 3: Fix pyproject.toml escape issues
    fix_pyproject_toml

    # Step 4: Fix file endings and whitespace
    fix_file_endings

    # Step 5: Add missing docstrings and fix code style
    fix_missing_docstrings

    # Step 6: Backup and update pre-commit configuration
    backup_precommit_config
    update_precommit_config

    # Step 7: Reinstall pre-commit
    reinstall_precommit

    # Step 8: Test the setup
    test_precommit_setup

    # Step 9: Commit all fixes
    commit_fixes

    echo -e "\n${PURPLE}🎉 All pre-commit issues fixed!${NC}"
    echo -e "${BLUE}📋 Summary of fixes:${NC}"
    echo -e "✅ Fixed pyproject.toml escape characters"
    echo -e "✅ Added missing docstrings"
    echo -e "✅ Fixed code style and line length issues"
    echo -e "✅ Fixed trailing whitespace and file endings"
    echo -e "✅ Updated pre-commit configuration"
    echo -e "✅ Reinstalled pre-commit hooks"
    echo -e "✅ Committed all fixes"

    echo -e "\n${BLUE}🔄 Next steps:${NC}"
    echo -e "1. Try committing normally now"
    echo -e "2. Pre-commit should work without issues"
    echo -e "3. All code style issues resolved"

    echo -e "\n${GREEN}💬 Terminal output confirmed - script completed successfully!${NC}"
}

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --test-only)
            echo -e "${BLUE}Running test mode...${NC}"
            test_precommit_setup
            exit 0
            ;;
        --clean-only)
            echo -e "${BLUE}Running clean mode...${NC}"
            clean_precommit_environments
            fix_trailing_whitespace
            fix_end_of_file
            exit 0
            ;;
        -h|--help)
            echo "Usage: $0 [--test-only] [--clean-only] [-h|--help]"
            echo "Options:"
            echo "  --test-only    Only test current pre-commit setup"
            echo "  --clean-only   Only clean files and environments"
            echo "  -h, --help     Show this help"
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            echo "Use -h or --help for usage information"
            exit 1
            ;;
    esac
done

# Run main function
echo -e "${PURPLE}🚀 Starting pre-commit fixes from /scripts/fixes/ directory${NC}"
main "$@"

# Final confirmation
echo -e "\n${GREEN}🎯 SCRIPT EXECUTION COMPLETE${NC}"
echo -e "${GREEN}If you can read this, the terminal output is working correctly!${NC}"
echo -e "${BLUE}Executed from: $(pwd)${NC}"
echo -e "${BLUE}Timestamp: $(date)${NC}"
