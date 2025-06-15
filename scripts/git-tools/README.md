# Git Tools

Git workflow utilities and commit management scripts.

## Scripts

### fix_git_commits.sh
**Purpose**: Comprehensive git commit issue resolution

**Usage**:
```bash
./scripts/git-tools/fix_git_commits.sh
# or with custom message
./scripts/git-tools/fix_git_commits.sh -m "Custom commit message"
```

**Features**:
- Cleans up large files and problematic paths
- Updates .gitignore automatically
- Stages files safely
- Handles pre-commit issues
- Option to skip remote push (`--no-push`)

### fix_git_unstaged_commits.sh
**Purpose**: Handle unstaged changes and prepare for commit

**Usage**:
```bash
./scripts/git-tools/fix_git_unstaged_commits.sh
```

**Features**:
- Fixes end-of-file issues
- Removes trailing whitespace
- Stages changes for VS Code Source Control
- Prepares files for manual commit

### commit_bypass.sh
**Purpose**: Emergency commit bypassing pre-commit hooks

**Usage**:
```bash
./scripts/git-tools/commit_bypass.sh "Emergency commit message"
# or with options
./scripts/git-tools/commit_bypass.sh "Fix urgent bug" --no-sync
```

**Features**:
- Bypasses broken pre-commit hooks
- Excludes large files automatically
- Emergency push to remote
- Customizable commit messages

## Workflow Integration

### Normal Development
1. Make changes
2. Run `fix_git_commits.sh` to prepare
3. Use VS Code Source Control or `git commit`

### When Pre-commit Broken
1. Run `commit_bypass.sh` for emergency
2. Fix pre-commit with `../fixes/fix_precommit_issues.sh`
3. Return to normal workflow

### VS Code Integration
1. Run `fix_git_unstaged_commits.sh`
2. Use VS Code Source Control panel
3. Review and commit through UI

## File Management

### Automatic Exclusions
- Large files (>10MB)
- Binary executables
- Virtual environment files
- Cache directories
- Model files (*.pkl, *.h5, etc.)

### Safe Staging
Only stages known-safe directories:
- `src/`, `tests/`, `docs/`, `scripts/`
- Configuration files
- Documentation files

## Exit Codes

- `0`: Success
- `1`: Git errors or missing repository
- Check output for specific error details

## Safety Features

- Creates .gitignore entries for large files
- Backs up configurations before changes
- Shows what will be committed before proceeding
- Provides rollback instructions
