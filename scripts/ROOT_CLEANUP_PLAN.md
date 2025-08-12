Root Folder Cleanup Plan - Phase 2
=====================================

The initial cleanup phase successfully moved all functional scripts to organized subdirectories. However, deprecation stubs are still present at the root level. This phase will complete the cleanup by moving all remaining Python files and stubs to appropriate locations.

## Current Root Python Files Status

**Deprecation Stubs (should be moved to scripts/legacy/):**


- demo_pyqt6_gui.py → scripts/legacy/demo_pyqt6_gui.py
- quick_test_agent.py → scripts/legacy/quick_test_agent.py
- run_gui.py → scripts/legacy/run_gui.py
- run_gui_pyqt6.py → scripts/legacy/run_gui_pyqt6.py
- run_gui_pyqt6_only.py → scripts/legacy/run_gui_pyqt6_only.py
- test_algorithm_database.py → scripts/legacy/test_algorithm_database.py
- test_complete_gui.py → scripts/legacy/test_complete_gui.py
- test_enhanced_results.py → scripts/legacy/test_enhanced_results.py
- test_gui_headless.py → scripts/legacy/test_gui_headless.py
- test_imports.py → scripts/legacy/test_imports.py
- test_pyqt6_gui.py → scripts/legacy/test_pyqt6_gui.py


**Shell Scripts (should be moved to scripts/legacy/):**

- activate_venv.sh → scripts/legacy/activate_venv.sh
- run.sh → scripts/legacy/run.sh
- run_agent.sh → scripts/legacy/run_agent.sh
- run_pyqt6_demo.sh → scripts/legacy/run_pyqt6_demo.sh
- test_agent_workflow.sh → scripts/legacy/test_agent_workflow.sh


**Batch Files:**


- run_agent.bat → scripts/legacy/run_agent.bat

**Documentation Stubs (already have redirect content):**

- PROJECT_CLEANUP_SUMMARY.md → Already points to docs/, can be removed or moved to scripts/legacy/
- PYQT6_GUI_GUIDE.md → Already points to docs/, can be removed or moved to scripts/legacy/

## Planned Actions

1. **Create scripts/legacy/ directory** for all backward-compatibility stubs
2. **Move all Python stub files** to scripts/legacy/
3. **Move all shell script stubs** to scripts/legacy/
4. **Update any remaining references** in documentation
5. **Create a single ROOT_MIGRATION_GUIDE.md** at root explaining the new structure
6. **Clean root to contain only essential project files**


## Final Root Structure Target

After cleanup, root should contain only:

```
Machine Learning Model/
├── .devcontainer/          # Development container config
├── .github/               # GitHub workflows and templates
├── .vscode/               # VS Code settings
├── config/                # Configuration files
├── data/                  # Data directories
├── docs/                  # Documentation
├── examples/              # Example scripts
├── models/                # Model storage
├── notebooks/             # Jupyter notebooks
├── scripts/               # All utility scripts (organized by category)
├── src/                   # Source code
├── tests/                 # Test suite
├── .env.example           # Environment template
├── .gitignore            # Git ignore rules
├── .pre-commit-config.yaml # Pre-commit hooks
├── CHANGELOG.md          # Version history
├── CONTRIBUTING.md       # Contribution guidelines
├── Dockerfile            # Docker build files
├── Dockerfile.gui
├── LICENSE               # License file
├── Makefile              # Build automation
├── README.md             # Project documentation
├── docker-compose.yml    # Docker composition
├── dvc.yaml              # Data version control
├── mkdocs.yml            # Documentation build
├── pyproject.toml        # Python project config
├── requirements.txt      # Dependencies
├── requirements-dev.txt  # Development dependencies
├── requirements.in       # Source requirements
└── requirements-dev.in   # Source dev requirements
```

## Benefits

1. **Clean professional root** with only essential project metadata
2. **Backward compatibility preserved** in scripts/legacy/
3. **Clear organization** - utilities in scripts/, source in src/, docs in docs/
4. **Easier navigation** - no confusion between stubs and actual files
5. **Better tooling support** - IDEs and tools work better with clean structure

This completes the root folder cleanup while preserving all functionality through organized subdirectories.
