import pandas as pd
import pytest

from src.machine_learning_model.data.loaders import (
    load_california_housing,
    load_iris_dataset,
    load_wine_dataset,
)


class TestDataLoaders:
    """Test suite for data loading functions"""

    def test_load_iris_dataset(self):
        """Test loading the Iris dataset"""
        data = load_iris_dataset()
        assert isinstance(data, pd.DataFrame), "Iris dataset should be a DataFrame"
        assert not data.empty, "Iris dataset should not be empty"
        assert set(data.columns) == {
            "sepal_length",
            "sepal_width",
            "petal_length",
            "petal_width",
            "species",
        }

    def test_load_wine_dataset(self):
        """Test loading the Wine dataset"""
        data = load_wine_dataset()
        assert isinstance(data, pd.DataFrame), "Wine dataset should be a DataFrame"
        assert not data.empty, "Wine dataset should not be empty"
        assert "alcohol" in data.columns, "Wine dataset should have 'alcohol' column"

    def test_load_california_housing(self):
        """Test loading the California Housing dataset"""
        data = load_california_housing()
        assert isinstance(
            data, pd.DataFrame
        ), "California Housing dataset should be a DataFrame"
        assert not data.empty, "California Housing dataset should not be empty"
        assert (
            "median_house_value" in data.columns
        ), "California Housing dataset should have 'median_house_value' column"

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
