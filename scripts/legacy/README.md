# Legacy Scripts Directory

This directory contains backward-compatibility stubs that were previously located at the project root. These files provide deprecation warnings and redirect to the new organized locations.

## Purpose

- **Backward Compatibility**: Ensures existing scripts and workflows continue to work
- **Migration Guidance**: Provides clear deprecation warnings with new file locations
- **Clean Root**: Allows the project root to contain only essential project metadata

## File Types

### Python Stubs
- GUI launcher stubs that redirect to `scripts/gui/`
- Test script stubs that redirect to `scripts/testing/`
- Agent script stubs that redirect to `scripts/agent/`

### Shell Script Stubs
- Environment activation stubs that redirect to `scripts/env/`
- Docker script stubs that redirect to `scripts/docker/`
- Agent workflow stubs that redirect to `scripts/agent/`

### Documentation Stubs
- Documentation file stubs that redirect to `docs/`

## Usage

These files are intended for backward compatibility only. **For new development, always use the files in their proper organized locations:**

- GUI scripts: `scripts/gui/`
- Agent scripts: `scripts/agent/`
- Test scripts: `scripts/testing/`
- Environment scripts: `scripts/env/`
- Docker scripts: `scripts/docker/`
- Documentation: `docs/`

## Migration Timeline

These stubs may be removed in a future version once all users and CI systems have migrated to the new file locations. Always use the organized locations for new development.

## Finding the New Location

If you're looking for a specific file:
1. Check the deprecation warning message for the new location
2. Look in the appropriate `scripts/` subdirectory
3. Refer to the main README.md for the current project structure
