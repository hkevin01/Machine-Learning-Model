"""Tests for data preprocessing utilities."""

import numpy as np
import pandas as pd
import pytest

from src.machine_learning_model.data.preprocessors import (
    DataPreprocessor,
    quick_preprocess,
)


class TestDataPreprocessor:
    """Test suite for DataPreprocessor class."""

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

    def test_handle_missing_values_mean(self, sample_data):
        """Test handling missing values with mean strategy."""
        preprocessor = DataPreprocessor()
        result = preprocessor.handle_missing_values(sample_data, strategy="mean")

        assert not result.isnull().any().any(), "Should not have any missing values"
        assert len(result) == len(sample_data), "Should maintain same number of rows"

    def test_handle_missing_values_drop(self, sample_data):
        """Test handling missing values by dropping rows."""
        preprocessor = DataPreprocessor()
        result = preprocessor.handle_missing_values(sample_data, strategy="drop")

        assert not result.isnull().any().any(), "Should not have any missing values"
        assert len(result) < len(sample_data), "Should have fewer rows after dropping"

    def test_normalize_features_standard(self, sample_data):
        """Test standard normalization of features."""
        preprocessor = DataPreprocessor()
        features = sample_data[["feature1", "feature2"]]
        result = preprocessor.normalize_features(features, method="standard")

        assert isinstance(result, pd.DataFrame), "Should return DataFrame"
        assert result.shape == features.shape, "Should maintain same shape"
        # Check if approximately standardized (mean ~0, std ~1)
        assert abs(result.mean().mean()) < 0.1, "Mean should be close to 0"

    def test_encode_categorical_variables(self, sample_data):
        """Test encoding of categorical variables."""
        preprocessor = DataPreprocessor()
        encoded_y, mapping = preprocessor.encode_categorical_variables(
            sample_data["target"]
        )

        assert isinstance(encoded_y, np.ndarray), "Should return numpy array"
        assert isinstance(mapping, dict), "Should return mapping dictionary"
        assert len(encoded_y) == len(sample_data), "Should maintain same length"
        assert set(encoded_y) == { 0, 1}, "Should have binary encoding for two classes"

    def test_split_train_test(self, sample_data):
        """Test train-test split functionality."""
        preprocessor = DataPreprocessor()
        X = sample_data[["feature1", "feature2"]]
        y = sample_data["target"]

        X_train, X_test, y_train, y_test = preprocessor.split_train_test(
            X, y, test_size=0.3
        )

        assert len(X_train) + len(X_test) == len(X), "Should split all data"
        assert len(y_train) + len(y_test) == len(y), "Should split all targets"
        assert len(X_train) == len(y_train), "Training sets should match"
        assert len(X_test) == len(y_test), "Test sets should match"

    def test_detect_outliers(self, sample_data):
        """Test outlier detection."""
        preprocessor = DataPreprocessor()
        features = sample_data[["feature1", "feature2"]]
        outliers = preprocessor.detect_outliers(features)

        assert isinstance(outliers, pd.DataFrame), "Should return DataFrame"
        assert outliers.shape == features.shape, "Should have same shape as input"
        # feature1 has value 100 which should be detected as outlier
        assert outliers["feature1"].any(), "Should detect outlier in feature1"


class TestQuickPreprocess:
    """Test suite for quick_preprocess function."""

    @pytest.fixture
    def sample_dataset(self):
        """Create sample dataset for testing."""
        return pd.DataFrame(
            {
                "feature1": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
                "feature2": [2, 4, 6, 8, 10, 12, 14, 16, 18, 20],
                "feature3": [1.1, 2.2, 3.3, 4.4, 5.5, 6.6, 7.7, 8.8, 9.9, 10.0],
                "target": ["A", "B", "A", "B", "A", "B", "A", "B", "A", "B"],
            }
        )

    def test_quick_preprocess_basic(self, sample_dataset):
        """Test basic quick preprocessing."""
        X_train, X_test, y_train, y_test = quick_preprocess(
            sample_dataset, target_column="target", test_size=0.2
        )

        assert len(X_train) + len(X_test) == len(
            sample_dataset
        ), "Should split all data"
        assert isinstance(X_train, pd.DataFrame), "Should return DataFrame for features"
        assert len(X_train.columns) == 3, "Should have 3 feature columns"

    def test_quick_preprocess_no_normalization(self, sample_dataset):
        """Test quick preprocessing without normalization."""
        X_train, X_test, y_train, y_test = quick_preprocess(
            sample_dataset, target_column="target", normalize=False
        )

        # Values should not be normalized (should be original scale)
        assert X_train["feature1"].max() > 5, "Should maintain original scale"

    def test_quick_preprocess_with_missing_values(self):
        """Test quick preprocessing with missing values."""
        df_with_missing = pd.DataFrame(
            {
                "feature1": [1, 2, np.nan, 4, 5],
                "feature2": [2, 4, 6, np.nan, 10],
                "target": ["A", "B", "A", "B", "A"],
            }
        )

        X_train, X_test, y_train, y_test = quick_preprocess(
            df_with_missing, target_column="target", handle_missing="mean"
        )

        # Should not have any missing values
        assert not pd.concat([X_train, X_test]).isnull().any().any()
