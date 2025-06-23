# Scripts Directory

This directory contains utility scripts for development, testing, and project management.

## Directory Structure

scripts/
├── development/           # Development environment setup
│   ├── create_python_project.sh  # Project template generator
│   ├── make_scripts_executable.sh # Make all scripts executable
│   ├── setup_tests.sh     # Test environment setup
│   ├── setup_venv.sh      # Virtual environment setup
│   └── README.md          # Development scripts documentation
├── fixes/                 # Issue resolution and troubleshooting
│   ├── fix_isort_connection.sh    # Fix isort connection issues
│   ├── fix_precommit_issues.sh    # Fix pre-commit hook problems
│   ├── fix_vscode_settings.sh     # Fix VS Code configuration
│   └── README.md          # Fix scripts documentation
├── git-tools/             # Git workflow utilities
│   ├── cleanup_backups.sh # Clean up backup files
│   ├── commit_bypass.sh   # Emergency commit bypass
│   ├── disable_precommit.sh # Temporarily disable pre-commit
│   ├── emergency_commit.sh # Emergency commit with fixes
│   ├── fix_and_stage.sh   # Fix and stage changes
│   ├── fix_git_commits.sh # Fix git commit issues
│   ├── fix_git_unstaged_commits.sh # Fix unstaged commits
│   ├── fix_precommit_config.sh # Fix pre-commit configuration
│   ├── fix_precommit_config_staging.sh # Fix staging issues
│   ├── fix_precommit_yaml.sh # Fix YAML configuration
│   ├── fix_python_errors.sh # Fix Python syntax errors
│   ├── fix_python_syntax.sh # Fix Python syntax issues
│   ├── fix_specific_syntax_errors.sh # Fix specific syntax errors
│   ├── fix_syntax_errors.sh # Fix general syntax errors
│   ├── fix_whitespace.sh  # Fix whitespace issues
│   └── README.md          # Git tools documentation
├── testing/               # Testing utilities
│   ├── create_test_output_folder.sh # Create test output directories
│   ├── quick_test.sh      # Quick test runner
│   ├── run_tests.sh       # Test execution script
│   └── README.md          # Testing scripts documentation
├── run_comprehensive_tests.sh # Comprehensive test suite
├── setup_virtualenv.sh    # Virtual environment setup
└── README.md              # This file

## Quick Reference

| Category | Purpose | Key Scripts |
|----------|---------|-------------|
| Development | Environment setup and project creation | setup_venv.sh, create_python_project.sh |
| Testing | Test execution and validation | run_comprehensive_tests.sh, quick_test.sh |
| Git Tools | Git workflow and commit management | commit_bypass.sh, fix_git_commits.sh |
| Fixes | Issue resolution and troubleshooting | fix_isort_connection.sh, fix_precommit_issues.sh |

## Common Workflows

### Initial Setup
```bash
# Set up virtual environment
./scripts/development/setup_venv.sh

# Make all scripts executable
./scripts/development/make_scripts_executable.sh

# Run comprehensive tests
./scripts/run_comprehensive_tests.sh
```

### Development Workflow
```bash
# Quick test during development
./scripts/testing/quick_test.sh

# Fix common issues
./scripts/fixes/fix_precommit_issues.sh

# Emergency commit if needed
./scripts/git-tools/emergency_commit.sh
```

### Troubleshooting
```bash
# Fix isort connection issues
./scripts/fixes/fix_isort_connection.sh

# Fix VS Code settings
./scripts/fixes/fix_vscode_settings.sh

# Clean up backup files
./scripts/git-tools/cleanup_backups.sh
```

## Script Categories

### Development Scripts
- **setup_venv.sh**: Creates and configures virtual environment
- **create_python_project.sh**: Generates new Python project templates
- **make_scripts_executable.sh**: Makes all scripts executable

### Testing Scripts
- **run_comprehensive_tests.sh**: Runs full test suite with coverage
- **quick_test.sh**: Fast test execution for development
- **run_tests.sh**: Standard test runner

### Git Tools
- **commit_bypass.sh**: Emergency commit bypass for urgent changes
- **fix_git_commits.sh**: Fixes common git commit issues
- **cleanup_backups.sh**: Removes backup files created by fix scripts

### Fix Scripts
- **fix_isort_connection.sh**: Resolves isort connection issues
- **fix_precommit_issues.sh**: Fixes pre-commit hook problems
- **fix_vscode_settings.sh**: Configures VS Code for optimal development

## Best Practices

1. **Always run scripts from project root**
2. **Check script documentation before use**
3. **Use fix scripts when encountering issues**
4. **Keep virtual environment activated during development**
5. **Run comprehensive tests before committing**

## Troubleshooting

If scripts fail:
1. Check if virtual environment is activated
2. Ensure all dependencies are installed
3. Run relevant fix scripts
4. Check script documentation for specific issues

For more detailed information, see the README files in each subdirectory.
