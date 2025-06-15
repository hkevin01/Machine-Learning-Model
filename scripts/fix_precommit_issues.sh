```bash
#!/bin/bash
# Fix Pre-commit Issues Script
# Resolves pre-commit hook problems and updates deprecated configurations

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[0;33m'
BLUE='\033[0;34m'
PURPLE='\033[0;35m'
NC='\033[0m' # No Color

echo -e "${PURPLE}🔧 Pre-commit Issues Fix Script${NC}"
echo -e "${BLUE}==================================${NC}"

# Function to check if we're in the right directory
check_project_directory() {
    if [ ! -f "pyproject.toml" ] || [ ! -f ".pre-commit-config.yaml" ]; then
        echo -e "${RED}❌ Error: Not in project root directory${NC}"
        echo -e "${YELLOW}💡 Please run this script from the project root${NC}"
        exit 1
    fi
}

# Function to backup current pre-commit config
backup_precommit_config() {
    echo -e "${BLUE}📦 Backing up current pre-commit configuration...${NC}"
    # ...existing code...
}

# Function to update pre-commit configuration
update_precommit_config() {
    echo -e "${BLUE}📝 Updating pre-commit configuration...${NC}"
    # ...existing code...
}

# Function to clean pre-commit environments
clean_precommit_environments() {
    echo -e "${BLUE}🗑️  Cleaning pre-commit environments...${NC}"
    # ...existing code...
}

# Function to fix trailing whitespace issues
fix_trailing_whitespace() {
    echo -e "${BLUE}✂️  Fixing trailing whitespace issues...${NC}"
    # ...existing code...
}

# Function to fix end of file issues
fix_end_of_file() {
    echo -e "${BLUE}📄 Fixing end of file issues...${NC}"
    # ...existing code...
}

# Function to reinstall pre-commit
reinstall_precommit() {
    echo -e "${BLUE}🔄 Reinstalling pre-commit...${NC}"
    # ...existing code...
}

# Function to run pre-commit autoupdate
update_precommit_repos() {
    echo -e "${BLUE}🔄 Updating pre-commit repositories...${NC}"
    # ...existing code...
}

# Function to test pre-commit setup
test_precommit_setup() {
    echo -e "${BLUE}🧪 Testing pre-commit setup...${NC}"
    # ...existing code...
}

# Function to configure git user if needed
configure_git_user() {
    echo -e "${BLUE}👤 Checking git user configuration...${NC}"
    # ...existing code...
}

# Function to stage and commit the fixes
commit_fixes() {
    echo -e "${BLUE}💾 Committing pre-commit configuration fixes...${NC}"
    # ...existing code...
}

# Main execution function
main() {
    echo -e "${BLUE}🔍 Starting pre-commit issues diagnosis and fix...${NC}"
    # Step 1: Verify we're in the right directory
    check_project_directory
    # Step 2: Configure git user if needed
    configure_git_user
    # Step 3: Backup current configuration
    backup_precommit_config
    # Step 4: Clean environments
    clean_precommit_environments
    # Step 5: Fix file issues
    fix_trailing_whitespace
    fix_end_of_file
    # Step 6: Update pre-commit configuration
    update_precommit_config
    # Step 7: Reinstall pre-commit
    reinstall_precommit
    # Step 8: Update repositories
    update_precommit_repos
    # Step 9: Test the setup
    test_precommit_setup
    # Step 10: Commit the fixes
    commit_fixes
    echo -e "\n${PURPLE}🎉 Pre-commit issues fix completed!${NC}"
    echo -e "${BLUE}📋 Summary of changes:${NC}"
    echo -e "✅ Updated pre-commit hooks to latest versions"
    echo -e "✅ Fixed deprecated stage names warning"
    echo -e "✅ Cleaned up trailing whitespace"
    echo -e "✅ Fixed end-of-file issues"
    echo -e "✅ Reinstalled pre-commit with fresh environments"
    echo -e "✅ Committed configuration fixes"
    echo -e "\n${BLUE}🔄 Next steps:${NC}"
    echo -e "1. Try committing your changes normally"
    echo -e "2. Pre-commit hooks should now work properly"
    echo -e "3. If issues persist, run: pre-commit run --all-files"
    echo -e "4. For emergency commits, use: ./scripts/commit_bypass.sh"
}

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --test-only)
            echo "Running test mode..."
            test_precommit_setup
            exit 0
            ;;
        --clean-only)
            echo "Running clean mode..."
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
main "$@"
```
