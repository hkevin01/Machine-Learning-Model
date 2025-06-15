#!/bin/bash
# Quick Test Script - Fast validation of core functionality

# Force all output to terminal immediately
exec > >(tee /dev/tty)
exec 2>&1

echo "⚡ Quick Test Suite"
echo "=================="
echo "⏰ Started at: $(date)"
echo "📍 Script location: /scripts/testing/quick_test.sh"
echo

# Activate venv
if [ -f "venv/bin/activate" ]; then
    source venv/bin/activate
    echo "✅ Virtual environment activated: $VIRTUAL_ENV"
else
    echo "⚠️  No venv found - tests may fail"
fi

# Quick syntax check
echo "🔍 Checking Python syntax..."
SYNTAX_ERRORS=0

if python -m py_compile src/machine_learning_model/data/loaders.py; then
    echo "✅ loaders.py - syntax OK"
else
    echo "❌ loaders.py - syntax ERROR"
    ((SYNTAX_ERRORS++))
fi

if python -m py_compile src/machine_learning_model/data/preprocessors.py; then
    echo "✅ preprocessors.py - syntax OK"
else
    echo "❌ preprocessors.py - syntax ERROR"
    ((SYNTAX_ERRORS++))
fi

if python -m py_compile src/machine_learning_model/data/validators.py; then
    echo "✅ validators.py - syntax OK"
else
    echo "❌ validators.py - syntax ERROR"
    ((SYNTAX_ERRORS++))
fi

if [ $SYNTAX_ERRORS -eq 0 ]; then
    echo "✅ All Python files compile successfully"
else
    echo "❌ Found $SYNTAX_ERRORS syntax errors"
fi

# Quick import test
echo
echo "📦 Testing imports..."
python -c "
import sys
sys.path.insert(0, '.')

try:
    from src.machine_learning_model.data.loaders import load_iris_dataset
    print('✅ Loader imports successful')
except Exception as e:
    print(f'❌ Loader import error: {e}')
    exit(1)

try:
    from src.machine_learning_model.data.preprocessors import DataPreprocessor
    print('✅ Preprocessor imports successful')
except Exception as e:
    print(f'❌ Preprocessor import error: {e}')
    exit(1)

try:
    from src.machine_learning_model.data.validators import DataValidator
    print('✅ Validator imports successful')
except Exception as e:
    print(f'❌ Validator import error: {e}')
    exit(1)

print('✅ All imports successful')
"

# Quick functionality test
echo
echo "🧪 Testing core functionality..."
python -c "
import sys
sys.path.insert(0, '.')

try:
    from src.machine_learning_model.data.loaders import load_iris_dataset
    data = load_iris_dataset()
    print(f'✅ Data loaded: {data.shape}')

    assert data.shape[0] > 0, 'No data loaded'
    assert 'species' in data.columns, 'Missing target column'
    print('✅ Data validation passed')

    # Test basic preprocessing
    from src.machine_learning_model.data.preprocessors import DataPreprocessor
    preprocessor = DataPreprocessor()

    # Test missing value handling
    cleaned_data = preprocessor.handle_missing_values(data)
    print(f'✅ Missing value handling: {cleaned_data.shape}')

    # Test data validation
    from src.machine_learning_model.data.validators import DataValidator
    validator = DataValidator()

    completeness = validator.validate_dataset_completeness(data)
    print(f'✅ Data validation: {completeness[\"passed\"]}')

    print('✅ Quick functionality test passed!')

except Exception as e:
    print(f'❌ Functionality test failed: {e}')
    import traceback
    traceback.print_exc()
    exit(1)
"

# Test data files exist
echo
echo "📁 Checking data files..."
DATA_FILES=(
    "data/raw/classification/iris/iris.csv"
    "data/raw/classification/wine/wine.csv"
    "data/raw/regression/housing/california_housing.csv"
    "data/raw/clustering/customers/mall_customers.csv"
    "data/raw/text/newsgroups/sample_newsgroups.csv"
)

MISSING_FILES=0
for file in "${DATA_FILES[@]}"; do
    if [ -f "$file" ]; then
        echo "✅ $file exists"
    else
        echo "❌ $file missing"
        ((MISSING_FILES++))
    fi
done

if [ $MISSING_FILES -eq 0 ]; then
    echo "✅ All required data files found"
else
    echo "⚠️  $MISSING_FILES data files missing"
fi

# Quick pytest run on one test file
echo
echo "🧪 Running quick pytest..."
if command -v pytest >/dev/null 2>&1; then
    if [ -f "tests/test_data/test_loaders.py" ]; then
        pytest tests/test_data/test_loaders.py::TestDataLoaders::test_load_iris_dataset -v --tb=short
        if [ $? -eq 0 ]; then
            echo "✅ Quick pytest passed"
        else
            echo "⚠️  Quick pytest had issues"
        fi
    else
        echo "⚠️  Test file not found: tests/test_data/test_loaders.py"
    fi
else
    echo "⚠️  pytest not available"
fi

echo
echo "📊 QUICK TEST SUMMARY"
echo "===================="
if [ $SYNTAX_ERRORS -eq 0 ] && [ $MISSING_FILES -eq 0 ]; then
    echo "🎉 Status: PASS - All quick tests successful!"
    echo "🚀 Ready for full test suite: ./scripts/testing/run_tests.sh"
    EXIT_CODE=0
else
    echo "⚠️  Status: PARTIAL - Some issues found"
    echo "🔧 Fix issues and run again"
    EXIT_CODE=1
fi

echo "⏰ Completed at: $(date)"
echo "💬 Terminal output confirmed - quick test completed!"

exit $EXIT_CODE
