# Fix Scripts Documentation

This directory contains scripts to resolve common development environment issues.

## Available Fix Scripts

### fix_isort_connection.sh
**Purpose**: Resolve isort connection issues and clear caches

**Common Issue**: "Connection refused" errors with isort

**Usage**:
```bash
./scripts/fixes/fix_isort_connection.sh
```

**What it does**:
- Kills hanging isort processes
- Clears isort cache
- Reinstalls isort if needed

### fix_precommit_issues.sh
**Purpose**: Resolve pre-commit hook issues

**Common Issue**: Pre-commit hooks failing or not working

**Usage**:
```bash
./scripts/fixes/fix_precommit_issues.sh
```

**What it does**:
- Reinstalls pre-commit hooks
- Clears pre-commit cache
- Updates hook configurations

### fix_vscode_settings.sh
**Purpose**: Configure VS Code for optimal Python development

**Usage**:
```bash
./scripts/fixes/fix_vscode_settings.sh
```

**What it does**:
- Sets up Python interpreter path
- Configures linting (flake8)
- Sets up formatting (black)
- Configures import sorting (isort)
- Sets up testing (pytest)

## Common Issues and Solutions

| Issue | Script | Result |
|-------|--------|--------|
| isort connection errors | `fix_isort_connection.sh` | isort working properly |
| Pre-commit hooks failing | `fix_precommit_issues.sh` | Hooks working correctly |
| VS Code not configured | `fix_vscode_settings.sh` | Optimal VS Code setup |

## Troubleshooting Tips

1. **General**: Restart VS Code completely after running fix scripts
2. **isort**: Clear cache and restart if issues persist
3. **Pre-commit**: Run `pre-commit clean` if hooks are corrupted

## Manual Fixes

If scripts don't resolve issues:

1. **isort**: `pip install --force-reinstall isort`
2. **Pre-commit**: `pre-commit clean && pre-commit install`
3. **VS Code**: Delete `.vscode/settings.json` and recreate
