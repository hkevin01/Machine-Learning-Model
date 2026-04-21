"""
Module: data.loaders
Purpose: Dataset loading utilities for the machine learning pipeline.
         Provides thin wrappers around pandas CSV reads so the rest of
         the codebase is decoupled from raw file paths.
Assumptions: CSV files exist at the expected relative paths under
             data/raw/. Run scripts/prepare_sample_data.py to generate
             them if missing.
Failure Modes: FileNotFoundError if the CSV is absent; propagated to
               caller so the pipeline step can emit a clear error.
"""

import pandas as pd


def load_iris_dataset() -> pd.DataFrame:
    """
    Purpose:    Load the Fisher Iris classification benchmark dataset.
    Returns:    DataFrame with 150 rows × 5 columns
                (sepal_length, sepal_width, petal_length, petal_width, species).
    Precond:    data/raw/classification/iris/iris.csv exists.
    Postcond:   Returns a valid DataFrame; raises FileNotFoundError otherwise.
    Error:      FileNotFoundError propagated to caller — not silenced here.
    """
    return pd.read_csv("data/raw/classification/iris/iris.csv")


def load_wine_dataset() -> pd.DataFrame:
    """
    Purpose:    Load the UCI Wine quality classification dataset.
    Returns:    DataFrame with wine chemical-property features and quality label.
    Precond:    data/raw/classification/wine/wine.csv exists.
    Postcond:   Returns a valid DataFrame; raises FileNotFoundError otherwise.
    Error:      FileNotFoundError propagated to caller — not silenced here.
    """
    return pd.read_csv("data/raw/classification/wine/wine.csv")


def load_california_housing() -> pd.DataFrame:
    """
    Purpose:    Load the California Housing regression benchmark dataset.
    Returns:    DataFrame with 8 numeric feature columns and a median_house_value
                target column.
    Precond:    data/raw/regression/housing/california_housing.csv exists.
    Postcond:   Returns a valid DataFrame; raises FileNotFoundError otherwise.
    Error:      FileNotFoundError propagated to caller — not silenced here.
    """
    return pd.read_csv("data/raw/regression/housing/california_housing.csv")
