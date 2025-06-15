# Development Scripts

Scripts for setting up and configuring the development environment.

## Scripts

### setup_venv.sh
**Purpose**: Create and configure virtual environment with all dependencies

**Usage**:
```bash
./scripts/development/setup_venv.sh
```

**What it does**:
- Creates Python virtual environment in `venv/`
- Installs pip and essential packages (pytest, pandas)
- Installs project requirements from `requirements.txt`
- Configures VS Code settings to recognize venv
- Creates symbolic links for easy access

**Output**: Detailed progress with success/failure indicators

### setup_tests.sh
**Purpose**: Set up testing environment and dependencies

**Usage**:
```bash
./scripts/development/setup_tests.sh
```

**What it does**:
- Installs test-specific dependencies
- Creates test data directories
- Sets up pytest configuration
- Prepares test fixtures

## Prerequisites

- Python 3.8+
- pip package manager
- Git repository initialized

## Troubleshooting

| Issue               | Solution                                  |
| ------------------- | ----------------------------------------- |
| Permission denied   | Run `chmod +x scripts/development/*.sh`   |
| Python not found    | Install Python 3.8+ or check PATH         |
| pip fails           | Try `python -m pip install --upgrade pip` |
| venv creation fails | Check disk space and permissions          |

## Environment Variables

The scripts may set these environment variables:
- `VIRTUAL_ENV`: Path to activated virtual environment
- `PYTHONPATH`: Python module search paths

## Files Created

After running setup scripts, you'll have:
- `venv/` - Virtual environment directory
- `.vscode/settings.json` - VS Code configuration
- `requirements.txt` - Basic requirements (if not present)
