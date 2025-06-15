# Testing Scripts

Comprehensive testing utilities for the Machine Learning Model project.

## Scripts

### run_tests.sh
**Purpose**: Comprehensive test runner with coverage reporting

**Usage**:
```bash
./scripts/testing/run_tests.sh
```

**Features**:
- Runs all pytest unit tests
- Generates coverage reports (HTML and terminal)
- Tests data loaders, preprocessors, and validators
- Enforces 80% coverage threshold

### quick_test.sh
**Purpose**: Fast validation of core functionality

**Usage**:
```bash
./scripts/testing/quick_test.sh
```

**Features**:
- Syntax checking for Python files
- Import testing
- Basic functionality validation
- Quick pytest on key tests
- Completes in under 30 seconds

### test_data_pipeline.sh
**Purpose**: Integration testing of complete data processing workflow

**Usage**:
```bash
./scripts/testing/test_data_pipeline.sh
```

**Features**:
- End-to-end pipeline testing
- Multiple dataset validation
- Preprocessing workflow testing
- Data validation pipeline testing

### test_mypy_fix.sh
**Purpose**: Validate MyPy daemon installation and functionality

**Usage**:
```bash
./scripts/testing/test_mypy_fix.sh
```

**Features**:
- Tests dmypy executable
- Validates VS Code configuration
- Tests type checking functionality

## Test Categories

- **Unit Tests**: Individual component testing
- **Integration Tests**: Component interaction testing
- **Pipeline Tests**: End-to-end workflow testing
- **Validation Tests**: Tool and configuration testing

## Coverage Reports

After running `run_tests.sh`, coverage reports are available:
- **Terminal**: Immediate feedback
- **HTML**: `htmlcov/data/index.html` for detailed view

## Prerequisites

- Virtual environment activated
- pytest and dependencies installed
- Test data files present

## Test Data Requirements

Required data files for testing:
- `data/raw/classification/iris/iris.csv`
- `data/raw/classification/wine/wine.csv`
- `data/raw/regression/housing/california_housing.csv`
- `data/raw/clustering/customers/mall_customers.csv`

## Exit Codes

All scripts use standard exit codes:
- `0`: Success
- `1`: Failure
- Check terminal output for details
