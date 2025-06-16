"""Test suite for data loading functions."""

import pytest
import pandas as pd
from src.machine_learning_model.data.loaders import (
    load_iris_dataset,
    load_wine_dataset,
    load_california_housing,
)


class TestDataLoaders:
    """Test suite for data loading functions."""

    def test_load_iris_dataset(self):
        """Test loading the Iris dataset."""
        data = load_iris_dataset()
        assert isinstance(data, pd.DataFrame), "Should be DataFrame"
        assert not data.empty, "Should not be empty"
        expected_cols = {
            "sepal_length",
            "sepal_width",
            "petal_length",
            "petal_width",
            "species",
        }
        assert set(data.columns) == expected_cols

    def test_load_wine_dataset(self):
        """Test loading the Wine dataset."""
        data = load_wine_dataset()
        assert isinstance(data, pd.DataFrame), "Should be DataFrame"
        assert not data.empty, "Should not be empty"
        assert "alcohol" in data.columns, "Should have alcohol column"

    def test_load_california_housing(self):
        """Test loading the California Housing dataset."""
        data = load_california_housing()
        assert isinstance(data, pd.DataFrame), "Should be DataFrame"
        assert not data.empty, "Should not be empty"
        assert "median_house_value" in data.columns

    def test_load_invalid_file(self):
        """Test loading an invalid file."""
        with pytest.raises(FileNotFoundError):
            pd.read_csv("non_existent_file.csv")

    def test_load_empty_file(self, tmp_path):
        """Test loading an empty file."""
        empty_file = tmp_path / "empty.csv"
        empty_file.touch()  # Create an empty file
        with pytest.raises(pd.errors.EmptyDataError):
            pd.read_csv(empty_file)
