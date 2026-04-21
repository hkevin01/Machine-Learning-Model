"""
Module: data.loaders
Purpose: Dataset loading utilities for the machine learning pipeline.
         Provides thin wrappers that first check for local CSV files and
         fall back to sklearn's bundled datasets, saving a copy for future
         runs so the pipeline is fully reproducible offline.
Assumptions: sklearn is available (it is a core dependency).
             Write permission to data/raw/ for initial dataset creation.
Failure Modes: OSError if data directory is not writable and CSV absent.
"""

from pathlib import Path

import pandas as pd
from sklearn import datasets as _sk_datasets


def _ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def load_iris_dataset() -> pd.DataFrame:
    """
    Purpose:    Load the Fisher Iris classification benchmark dataset.
    Returns:    DataFrame with 150 rows × 5 columns
                (sepal_length, sepal_width, petal_length, petal_width, species).
    Postcond:   Returns a valid DataFrame. Creates the CSV on first call if absent.
    """
    csv_path = Path("data/raw/classification/iris/iris.csv")
    if csv_path.exists():
        return pd.read_csv(csv_path)

    _ensure_dir(csv_path.parent)
    raw = _sk_datasets.load_iris(as_frame=True)
    df = raw.frame.copy()
    df.columns = [
        "sepal_length", "sepal_width", "petal_length", "petal_width", "species"
    ]
    # Replace numeric target with readable names
    target_names = raw.target_names
    df["species"] = df["species"].map(lambda v: target_names[int(v)])
    df.to_csv(csv_path, index=False)
    return df


def load_wine_dataset() -> pd.DataFrame:
    """
    Purpose:    Load the UCI Wine classification dataset.
    Returns:    DataFrame with wine chemical-property features and a target column.
    Postcond:   Returns a valid DataFrame. Creates the CSV on first call if absent.
    """
    csv_path = Path("data/raw/classification/wine/wine.csv")
    if csv_path.exists():
        return pd.read_csv(csv_path)

    _ensure_dir(csv_path.parent)
    raw = _sk_datasets.load_wine(as_frame=True)
    df = raw.frame.copy()
    df.to_csv(csv_path, index=False)
    return df


def load_california_housing() -> pd.DataFrame:
    """
    Purpose:    Load the California Housing regression benchmark dataset.
    Returns:    DataFrame with 8 numeric feature columns plus median_house_value.
    Postcond:   Returns a valid DataFrame. Creates the CSV on first call if absent.
    """
    csv_path = Path("data/raw/regression/housing/california_housing.csv")
    if csv_path.exists():
        return pd.read_csv(csv_path)

    _ensure_dir(csv_path.parent)
    raw = _sk_datasets.fetch_california_housing(as_frame=True)
    df = raw.frame.copy()
    # Rename target column to match expected name
    df.rename(columns={"MedHouseVal": "median_house_value"}, inplace=True)
    # Normalise column names to lowercase_underscore
    df.columns = [c.lower() for c in df.columns]
    df.to_csv(csv_path, index=False)
    return df

