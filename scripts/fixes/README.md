# Fix Scripts

Issue resolution and troubleshooting utilities.

## Scripts

### fix_mypy_daemon.sh
**Purpose**: Resolve MyPy daemon executable issues

**Common Issue**: "The mypy daemon executable ('dmypy') was not found"

**Usage**:
```bash
./scripts/fixes/fix_mypy_daemon.sh
```

**Solutions**:
- Installs/upgrades mypy in virtual environment
- Configures VS Code settings with absolute paths
- Creates symbolic links for dmypy access
- Tests daemon functionality

### fix_precommit_issues.sh
**Purpose**: Fix pre-commit hook problems and configuration

**Common Issues**:
- Deprecated stage names warnings
- Trailing whitespace failures
- Missing docstrings
- Configuration syntax errors

**Usage**:
```bash
./scripts/fixes/fix_precommit_issues.sh
```

**Solutions**:
- Updates pre-commit configuration to latest versions
- Fixes pyproject.toml escape character issues
- Adds missing docstrings
- Resolves code style violations

### fix_vscode_settings.sh
**Purpose**: Repair VS Code configuration issues

**Common Issue**: "Unable to write into user settings"

**Usage**:
```bash
./scripts/fixes/fix_vscode_settings.sh
```

**Solutions**:
- Fixes JSON syntax errors in settings
- Repairs file permissions
- Creates clean configuration files
- Clears corrupted cache

## When to Use

| Symptom                     | Script                    | Expected Result        |
| --------------------------- | ------------------------- | ---------------------- |
| MyPy not working in VS Code | `fix_mypy_daemon.sh`      | Type checking enabled  |
| Pre-commit hooks failing    | `fix_precommit_issues.sh` | Clean commits possible |
| VS Code errors on startup   | `fix_vscode_settings.sh`  | VS Code runs normally  |

## Post-Fix Actions

After running fix scripts:

1. **MyPy**: Restart VS Code completely
2. **Pre-commit**: Try normal commit process
3. **VS Code**: Reload window and select Python interpreter

## Backup Policy

All fix scripts create backups:
- Timestamped backup files (`.backup.YYYYMMDD_HHMMSS`)
- Original configurations preserved
- Safe to run multiple times

## Manual Fallbacks

If scripts fail:
- **MyPy**: `pip install mypy[dmypy]`
- **Pre-commit**: `pre-commit clean && pre-commit install`
- **VS Code**: Delete `.vscode/settings.json` and restart
