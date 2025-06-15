#!/bin/bash
# Comprehensive Test Runner Script
# Runs pytest unit tests for data processing utilities

echo "🧪 Test Runner Script for ML Model Project"
echo "==========================================="

# Activate virtual environment
if [ -f "venv/bin/activate" ]; then
    source venv/bin/activate
    echo "✅ Virtual environment activated"
else
    echo "❌ Virtual environment not found"
    exit 1
fi

# Install test dependencies
echo "📦 Installing test dependencies..."
pip install pytest pytest-cov pytest-mock coverage

# Run comprehensive tests
echo "🚀 Running pytest unit tests..."

# Test data loading utilities
echo "📊 Testing data loading utilities..."
python -m pytest tests/test_data/test_loaders.py -v --cov=src/machine_learning_model/data/loaders --cov-report=term-missing

# Test preprocessing utilities
echo "🔧 Testing preprocessing utilities..."
python -m pytest tests/test_data/test_preprocessors.py -v --cov=src/machine_learning_model/data/preprocessors --cov-report=term-missing

# Test validation utilities
echo "✅ Testing validation utilities..."
python -m pytest tests/test_data/test_validators.py -v --cov=src/machine_learning_model/data/validators --cov-report=term-missing

# Run all data tests together
echo "🎯 Running all data processing tests..."
python -m pytest tests/test_data/ -v --cov=src/machine_learning_model/data --cov-report=html --cov-report=term-missing --cov-fail-under=80

# Generate coverage report
echo "📈 Generating coverage report..."
coverage report --include="src/machine_learning_model/data/*"
coverage html --include="src/machine_learning_model/data/*" --directory=htmlcov/data

echo
echo "🎉 Test execution completed!"
echo "📋 Test Results Summary:"
echo "✅ Data loaders tested"
echo "✅ Data preprocessors tested"
echo "✅ Data validators tested"
echo "📊 Coverage report: htmlcov/data/index.html"
