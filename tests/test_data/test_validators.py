"""Tests for data validation utilities."""

import numpy as np
import pandas as pd
import pytest

from src.machine_learning_model.data.validators import (
    DataValidator,
    validate_ml_dataset,
)


class TestDataValidator:
    """Test suite for DataValidator class."""

    @pytest.fixture
    def sample_data(self):
        """Create sample data for testing."""
        return pd.DataFrame(
            {
                "feature1": [1, 2, 3, 4, 5, 100],  # Contains outlier
                "feature2": [2, 4, 6, 8, 10, 12],
                "feature3": [1.1, 2.2, np.nan, 4.4, 5.5, 6.6],  # Contains missing value
                "target": ["A", "B", "A", "B", "A", "B"],
            }
        )

    def test_validate_data_types(self, sample_data):
        """Test data type validation."""
        validator = DataValidator()
        expected_types = {
            "feature1": "int",
            "feature2": "int",
            "feature3": "float",
            "target": "object",
        }

        result = validator.validate_data_types(sample_data, expected_types)

        assert isinstance(result, dict), "Should return dictionary"
        assert "passed" in result, "Should have 'passed' key"
        assert "summary" in result, "Should have 'summary' key"

    def test_validate_target_distribution(self, sample_data):
        """Test target distribution validation."""
        validator = DataValidator()
        result = validator.validate_target_distribution(sample_data["target"])

        assert isinstance(result, dict), "Should return dictionary"
        assert "distribution" in result, "Should have distribution info"
        assert "class_balance" in result, "Should have class balance info"
        assert result["distribution"]["A"] == 3, "Should count class A correctly"
        assert result["distribution"]["B"] == 3, "Should count class B correctly"

    def test_detect_outliers(self, sample_data):
        """Test outlier detection."""
        validator = DataValidator()
        result = validator.detect_outliers(sample_data)

        assert isinstance(result, dict), "Should return dictionary"
        assert "total_outliers" in result, "Should have total outlier count"
        assert "outliers_by_column" in result, "Should have column-wise outliers"
        assert result["total_outliers"] > 0, "Should detect the outlier (100)"

    def test_validate_dataset_completeness(self, sample_data):
        """Test dataset completeness validation."""
        validator = DataValidator()
        result = validator.validate_dataset_completeness(sample_data)

        assert isinstance(result, dict), "Should return dictionary"
        assert "statistics" in result, "Should have statistics"
        assert result["statistics"]["total_rows"] == 6, "Should count rows correctly"
        assert (
            result["statistics"]["total_columns"] == 4
        ), "Should count columns correctly"

    def test_validate_column_names(self, sample_data):
        """Test column name validation."""
        validator = DataValidator()
        required_columns = ["feature1", "feature2", "target"]
        result = validator.validate_column_names(sample_data, required_columns)

        assert isinstance(result, dict), "Should return dictionary"
        assert result["passed"], "Should pass validation"
        assert len(result["missing_columns"]) == 0, "Should have no missing columns"

    def test_validate_column_names_missing(self, sample_data):
        """Test column name validation with missing columns."""
        validator = DataValidator()
        required_columns = ["feature1", "missing_column", "target"]
        result = validator.validate_column_names(sample_data, required_columns)

        assert not result["passed"], "Should fail validation"
        assert (
            "missing_column" in result["missing_columns"]
        ), "Should detect missing column"


class TestValidateMLDataset:
    """Test suite for validate_ml_dataset function."""

    @pytest.fixture
    def good_dataset(self):
        """Create a good quality dataset."""
        return pd.DataFrame(
            {
                "feature1": range(1, 21),
                "feature2": range(2, 42, 2),
                "feature3": np.random.normal(0, 1, 20),
                "target": ["A"] * 10 + ["B"] * 10,
            }
        )

    @pytest.fixture
    def poor_dataset(self):
        """Create a poor quality dataset with issues."""
        return pd.DataFrame(
            {
                "feature1": [1, 2, np.nan, np.nan, 5],  # Many missing values
                "feature2": [1, 1, 1, 1, 1],  # No variance
                "target": ["A", "A", "A", "A", "B"],  # Imbalanced
            }
        )

    def test_validate_ml_dataset_good(self, good_dataset):
        """Test validation with good quality dataset."""
        result = validate_ml_dataset(
            good_dataset,
            target_column="target",
            required_columns=["feature1", "feature2", "feature3", "target"],
        )

        assert isinstance(result, dict), "Should return dictionary"
        assert "overall_passed" in result, "Should have overall pass status"
        assert "validations" in result, "Should have validation details"
        assert "summary" in result, "Should have summary"

    def test_validate_ml_dataset_poor(self, poor_dataset):
        """Test validation with poor quality dataset."""
        result = validate_ml_dataset(
            poor_dataset, target_column="target", min_samples=3
        )

        assert isinstance(result, dict), "Should return dictionary"
        assert result["summary"]["total_warnings"] > 0, "Should have warnings"

    def test_validate_ml_dataset_missing_target(self, good_dataset):
        """Test validation with missing target column."""
        result = validate_ml_dataset(good_dataset, target_column="missing_target")

        # Should handle missing target gracefully
        assert isinstance(result, dict), "Should return dictionary"
