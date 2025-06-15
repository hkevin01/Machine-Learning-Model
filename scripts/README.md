# Scripts Directory

This directory contains utility scripts organized by purpose for the Machine Learning Model project.

## Directory Structure

scripts/
├── development/              # Development environment setup
│   ├── setup_venv.sh         # Virtual environment setup
│   ├── setup_tests.sh        # Test environment setup
│   └── README.md             # Development scripts documentation
├── testing/                  # Testing and validation scripts
│   ├── run_tests.sh          # Comprehensive test runner
│   ├── test_data_pipeline.sh  # Data pipeline integration tests
│   ├── quick_test.sh         # Fast validation tests
│   ├── test_mypy_fix.sh      # MyPy daemon validation
│   └── README.md             # Testing scripts documentation
├── fixes/                    # Issue resolution scripts
│   ├── fix_mypy_daemon.sh    # Fix MyPy daemon issues
│   ├── fix_precommit_issues.sh # Fix pre-commit problems
│   ├── fix_vscode_settings.sh # Fix VS Code configuration
│   └── README.md             # Fix scripts documentation
├── git-tools/                # Git-related utilities
│   ├── fix_git_commits.sh    # Fix git commit issues
│   ├── fix_git_unstaged_commits.sh # Handle unstaged changes
│   ├── commit_bypass.sh      # Emergency commit bypass
│   └── README.md             # Git tools documentation
└── README.md                 # This file

## Testing
- **Fix Issues**
- **Git Tools**

## Subfolder Documentation
Each subfolder contains its own README.md with detailed documentation:

- **development/README.md**: Environment setup scripts, virtual environment configuration, and development dependencies
- **testing/README.md**: Test execution, coverage reporting, and validation utilities
- **fixes/README.md**: Issue resolution for MyPy, pre-commit, and VS Code problems
- **git-tools/README.md**: Git workflow management, commit utilities, and repository maintenance

## Script Categories
| Category     | Purpose                              | Key Scripts                                        |
| ------------ | ------------------------------------ | -------------------------------------------------- |
| development/ | Environment setup and configuration  | setup_venv.sh, setup_tests.sh                      |
| testing/     | Test execution and validation        | run_tests.sh, quick_test.sh, test_data_pipeline.sh |
| fixes/       | Issue resolution and troubleshooting | fix_mypy_daemon.sh, fix_precommit_issues.sh        |
| git-tools/   | Git workflow and commit utilities    | fix_git_commits.sh, commit_bypass.sh               |

## Usage Guidelines
- **Run from project root**: All scripts are designed to be run from the project root directory
- **Make executable**: Use `chmod +x scripts/category/script.sh` if needed
- **Check output**: All scripts provide detailed terminal output
- **Follow dependencies**: Some scripts depend on others (e.g., `setup_venv.sh` before tests)
- **Read subfolder docs**: Check specific README files for detailed usage instructions

## Common Workflows
- Initial Project Setup
- Development Cycle
- Troubleshooting

## Common Issues and Solutions
| Issue                    | Script to Run                 | Subfolder Docs        |
| ------------------------ | ----------------------------- | --------------------- |
| MyPy daemon not found    | fixes/fix_mypy_daemon.sh      | fixes/README.md       |
| Pre-commit hooks failing | fixes/fix_precommit_issues.sh | fixes/README.md       |
| VS Code settings errors  | fixes/fix_vscode_settings.sh  | fixes/README.md       |
| Git commit problems      | git-tools/fix_git_commits.sh  | git-tools/README.md   |
| Test failures            | testing/run_tests.sh          | testing/README.md     |
| Environment issues       | development/setup_venv.sh     | development/README.md |

## Exit Codes
All scripts use standard exit codes:

- **0**: Success
- **1**: Failure

Check terminal output and subfolder documentation for details

For detailed usage instructions and troubleshooting, see the README.md files in each subfolder.
