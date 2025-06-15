#!/bin/bash
# Script to set up test files and folders for data processing tests

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[0;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

echo -e "${BLUE}🔧 Setting up test files and folders for data processing tests...${NC}"

# Base directories
PROJECT_DIR="/home/kevin/Projects/AI MODELS/Machine Learning Model"
TESTS_DIR="$PROJECT_DIR/tests/test_data"
SRC_DIR="$PROJECT_DIR/src/machine_learning_model/data"

# Create necessary directories
mkdir -p "$TESTS_DIR"
mkdir -p "$SRC_DIR"

# Create __init__.py files
echo -e "${BLUE}📂 Creating __init__.py files...${NC}"
touch "$PROJECT_DIR/tests/__init__.py"
touch "$TESTS_DIR/__init__.py"
touch "$PROJECT_DIR/src/machine_learning_model/__init__.py"
touch "$SRC_DIR/__init__.py"

# Create test_loaders.py
echo -e "${BLUE}📝 Creating test_loaders.py...${NC}"
cat > "$TESTS_DIR/test_loaders.py" << 'EOF'
# filepath: /home/kevin/Projects/AI MODELS/Machine Learning Model/tests/test_data/test_loaders.py
import pytest
import pandas as pd
from src.machine_learning_model.data.loaders import load_iris_dataset, load_wine_dataset, load_california_housing

class TestDataLoaders:
    """Test suite for data loading functions"""

    def test_load_iris_dataset(self):
        """Test loading the Iris dataset"""
        data = load_iris_dataset()
        assert isinstance(data, pd.DataFrame), "Iris dataset should be a DataFrame"
        assert not data.empty, "Iris dataset should not be empty"
        assert set(data.columns) == {"sepal_length", "sepal_width", "petal_length", "petal_width", "species"}

    def test_load_wine_dataset(self):
        """Test loading the Wine dataset"""
        data = load_wine_dataset()
        assert isinstance(data, pd.DataFrame), "Wine dataset should be a DataFrame"
        assert not data.empty, "Wine dataset should not be empty"
        assert "alcohol" in data.columns, "Wine dataset should have 'alcohol' column"

    def test_load_california_housing(self):
        """Test loading the California Housing dataset"""
        data = load_california_housing()
        assert isinstance(data, pd.DataFrame), "California Housing dataset should be a DataFrame"
        assert not data.empty, "California Housing dataset should not be empty"
        assert "median_house_value" in data.columns, "California Housing dataset should have 'median_house_value' column"

    def test_load_invalid_file(self):
        """Test loading an invalid file"""
        with pytest.raises(FileNotFoundError):
            pd.read_csv("non_existent_file.csv")

    def test_load_empty_file(self, tmp_path):
        """Test loading an empty file"""
        empty_file = tmp_path / "empty.csv"
        empty_file.touch()  # Create an empty file
        with pytest.raises(pd.errors.EmptyDataError):
            pd.read_csv(empty_file)
EOF

# Create loaders.py
echo -e "${BLUE}📝 Creating loaders.py...${NC}"
cat > "$SRC_DIR/loaders.py" << 'EOF'
# filepath: /home/kevin/Projects/AI MODELS/Machine Learning Model/src/machine_learning_model/data/loaders.py
import pandas as pd

def load_iris_dataset():
    """Load the Iris dataset"""
    return pd.read_csv("data/raw/classification/iris/iris.csv")

def load_wine_dataset():
    """Load the Wine dataset"""
    return pd.read_csv("data/raw/classification/wine/wine.csv")

def load_california_housing():
    """Load the California Housing dataset"""
    return pd.read_csv("data/raw/regression/housing/california_housing.csv")
EOF

echo -e "${GREEN}✅ Test files and folders set up successfully!${NC}"
