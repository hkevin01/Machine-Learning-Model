# Testing Scripts

This directory contains scripts for running tests and validating the project.

## Available Scripts

### run_tests.sh
**Purpose**: Run the complete test suite with coverage reporting

**Usage**:
```bash
./scripts/testing/run_tests.sh
```

**What it does**:
- Runs pytest with coverage
- Generates HTML coverage report
- Saves test results to test-outputs/

### quick_test.sh
**Purpose**: Fast test execution for development workflow

**Usage**:
```bash
./scripts/testing/quick_test.sh
```

**What it does**:
- Runs tests without coverage (faster)
- Provides quick feedback during development
- Exits on first failure

### create_test_output_folder.sh
**Purpose**: Create test output directories

**Usage**:
```bash
./scripts/testing/create_test_output_folder.sh
```

**What it does**:
- Creates test-outputs/ directory structure
- Sets up subdirectories for reports and coverage
- Ensures proper permissions

## Test Output Structure

```
test-outputs/
├── reports/
│   ├── pytest.log
│   ├── flake8.log
│   ├── black.log
│   └── isort.log
└── coverage/
    ├── index.html
    ├── coverage.xml
    └── coverage.txt
```

## Running Tests

### Full Test Suite
```bash
# From project root
./scripts/testing/run_tests.sh
```

### Quick Development Tests
```bash
# Fast feedback during development
./scripts/testing/quick_test.sh
```

### Manual Test Execution
```bash
# Run specific test file
pytest tests/test_data/test_loaders.py -v

# Run with coverage
pytest --cov=src --cov-report=html

# Run specific test function
pytest tests/test_data/test_loaders.py::TestDataLoaders::test_load_iris_dataset -v
```

## Coverage Reports

After running tests, coverage reports are available:

- **HTML Report**: `test-outputs/coverage/index.html`
- **XML Report**: `test-outputs/coverage/coverage.xml`
- **Text Report**: `test-outputs/coverage/coverage.txt`

## Test Configuration

Tests are configured in:
- `pytest.ini`: Pytest configuration
- `pyproject.toml`: Coverage and test settings
- `conftest.py`: Shared test fixtures

## Best Practices

1. **Run quick tests during development**
2. **Run full suite before committing**
3. **Check coverage reports regularly**
4. **Add tests for new features**
5. **Keep test output directories clean**

## Troubleshooting

### Common Issues

**Tests not found**:
```bash
# Ensure you're in the project root
pwd
# Should show: /path/to/Machine Learning Model

# Check test discovery
pytest --collect-only
```

**Coverage not working**:
```bash
# Install coverage
pip install pytest-cov

# Run with explicit coverage
pytest --cov=src --cov-report=term-missing
```

**Import errors**:
```bash
# Ensure virtual environment is activated
source venv/bin/activate

# Install package in development mode
pip install -e .
```

## Integration with CI/CD

The test scripts are designed to work with CI/CD pipelines:

- Exit codes indicate success/failure
- Reports are saved to test-outputs/
- Coverage thresholds can be enforced
- Test results are machine-readable
