"""
Module: data.preprocessors
Purpose: Stateful data preprocessing for ML pipelines.
         Provides fit-once / transform-many semantics for scaling,
         encoding, imputation, outlier detection, and train-test splits.
         A single DataPreprocessor instance should be created per experiment
         and reused so fitted scalers/encoders are consistent across splits.
Assumptions: Input DataFrames contain only numeric columns for scaling
             operations. Categorical encoding targets 1-D label arrays.
Failure Modes: ValueError on unknown strategy/method names.
               TypeError if non-DataFrame inputs are passed to numeric ops.
"""

from typing import Optional, Tuple, Union

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, MinMaxScaler, StandardScaler


class DataPreprocessor:
    """
    Purpose:    Stateful transformer that wraps common preprocessing steps.
                Maintains fitted scaler and label-encoder references so
                the same transformations can be applied to validation and
                test splits without data leakage.
    Constraints: One instance per experiment; do not share across projects.
    """

    def __init__(self):
        """
        Purpose:   Initialise with no fitted state.
        Postcond:  self.scaler and self.label_encoder are None; the first
                   call to normalize_features or encode_categorical_variables
                   triggers fitting.
        """
        self.scaler = None
        self.label_encoder = None

    def handle_missing_values(
        self, df: pd.DataFrame, strategy: str = "mean"
    ) -> pd.DataFrame:
        """
        Purpose:  Impute or drop missing values in a DataFrame.
        Inputs:   df       — source DataFrame, may contain NaN.
                  strategy — 'mean' | 'median' | 'mode' | 'drop'.
        Returns:  New DataFrame with NaN handled per strategy; original
                  is not mutated.
        Error:    ValueError on unknown strategy.
        """
        if strategy == "mean":
            numeric_cols = df.select_dtypes(include=[np.number]).columns
            fill_values = {col: df[col].mean() for col in numeric_cols}
            return df.fillna(fill_values)
        elif strategy == "median":
            numeric_cols = df.select_dtypes(include=[np.number]).columns
            fill_values = {col: df[col].median() for col in numeric_cols}
            return df.fillna(fill_values)
        elif strategy == "mode":
            fill_values = {col: df[col].mode().iloc[0] for col in df.columns if not df[col].mode().empty}
            return df.fillna(fill_values)
        elif strategy == "drop":
            return df.dropna()
        else:
            raise ValueError(f"Unknown strategy: {strategy}")

    def normalize_features(
        self, X: pd.DataFrame, method: str = "standard"
    ) -> pd.DataFrame:
        """
        Purpose:    Scale numeric features to a standard range.
                    Fits the scaler on first call; subsequent calls apply
                    the already-fitted transform (prevents data leakage).
        Inputs:     X      — numeric feature DataFrame.
                    method — 'standard' (zero-mean unit-variance) |
                             'minmax' (0-1 range).
        Returns:    Scaled DataFrame preserving original column names and index.
        Precond:    X must contain only numeric columns.
        Side-effect: Mutates self.scaler on first call.
        Error:      ValueError on unknown method.
        """
        if method == "standard":
            if self.scaler is None:
                self.scaler = StandardScaler()
                scaled_data = self.scaler.fit_transform(X)
            else:
                scaled_data = self.scaler.transform(X)
        elif method == "minmax":
            if self.scaler is None:
                self.scaler = MinMaxScaler()
                scaled_data = self.scaler.fit_transform(X)
            else:
                scaled_data = self.scaler.transform(X)
        else:
            raise ValueError(f"Unknown normalization method: {method}")

        return pd.DataFrame(scaled_data, columns=X.columns, index=X.index)

    def encode_categorical_variables(self, y: pd.Series) -> Tuple[np.ndarray, dict]:
        """
        Purpose:    Ordinal-encode a categorical target Series.
                    Fits the LabelEncoder on first call; subsequent calls
                    apply the fitted mapping (prevents unseen-class errors
                    on test data when labels are consistent).
        Inputs:     y — 1-D pandas Series of string or integer class labels.
        Returns:    (encoded_array, mapping_dict) where mapping_dict maps
                    original class name → integer code.
        Side-effect: Mutates self.label_encoder on first call.
        """
        if self.label_encoder is None:
            self.label_encoder = LabelEncoder()
            encoded_y = self.label_encoder.fit_transform(y)
        else:
            encoded_y = self.label_encoder.transform(y)

        # Build human-readable class→code mapping for downstream display
        mapping = dict(
            zip(
                self.label_encoder.classes_,
                np.array(
                    self.label_encoder.transform(self.label_encoder.classes_)
                ).tolist(),
            )
        )

        return np.asarray(encoded_y), mapping

    def split_train_test(
        self,
        X: pd.DataFrame,
        y: Union[pd.Series, np.ndarray],
        test_size: float = 0.2,
        random_state: int = 42,
    ) -> Tuple[
        pd.DataFrame,
        pd.DataFrame,
        Union[pd.Series, np.ndarray],
        Union[pd.Series, np.ndarray],
    ]:
        """
        Purpose:  Partition features and labels into training and test sets.
        Inputs:   X            — feature DataFrame.
                  y            — label array or Series, same length as X.
                  test_size    — fraction for test set; range (0, 1).
                  random_state — seed for reproducibility.
        Returns:  (X_train, X_test, y_train, y_test) tuple.
        Precond:  len(X) == len(y), test_size ∈ (0, 1).
        """
        return tuple(
            train_test_split(X, y, test_size=test_size, random_state=random_state)
        )

    def detect_outliers(
        self, X: pd.DataFrame, method: str = "iqr", threshold: float = 1.5
    ) -> pd.DataFrame:
        """
        Purpose:  Produce a boolean mask identifying outlier cells.
        Inputs:   X         — numeric DataFrame.
                  method    — 'iqr' (interquartile range fence).
                  threshold — IQR multiplier for fence width (default 1.5
                              is Tukey's conventional fence).
        Returns:  Boolean DataFrame; True where a cell is an outlier.
        Error:    ValueError on unknown method.
        """
        if method == "iqr":
            Q1 = X.quantile(0.25)
            Q3 = X.quantile(0.75)
            IQR = Q3 - Q1
            outliers = (X < (Q1 - threshold * IQR)) | (X > (Q3 + threshold * IQR))
            return outliers
        else:
            raise ValueError(f"Unknown outlier detection method: {method}")

    def remove_outliers(
        self, df: pd.DataFrame, columns: Optional[list] = None, method: str = "iqr"
    ) -> pd.DataFrame:
        """
        Purpose:  Drop rows that contain outliers in any of the target columns.
        Inputs:   df      — source DataFrame.
                  columns — column names to inspect; defaults to all numeric
                            columns if None.
                  method  — passed through to detect_outliers.
        Returns:  Filtered DataFrame (subset of rows); original not mutated.
        """
        if columns is None:
            columns = df.select_dtypes(include=[np.number]).columns.tolist()

        outliers = self.detect_outliers(df[columns], method=method)
        # Retain only rows where no inspected column is an outlier
        mask = ~outliers.any(axis=1)
        return df[mask]


def quick_preprocess(
    df: pd.DataFrame,
    target_column: str,
    test_size: float = 0.2,
    normalize: bool = True,
    handle_missing: str = "mean",
    remove_outliers: bool = False,
) -> Tuple[
    pd.DataFrame,
    pd.DataFrame,
    Union[pd.Series, np.ndarray],
    Union[pd.Series, np.ndarray],
]:
    """Quick preprocessing pipeline for common ML tasks."""
    preprocessor = DataPreprocessor()

    # Separate features and target
    X = df.drop(columns=[target_column])
    y = df[target_column]

    # Handle missing values
    if handle_missing:
        X = preprocessor.handle_missing_values(X, strategy=handle_missing)

    # Remove outliers if requested
    if remove_outliers:
        combined = pd.concat([X, y], axis=1)
        combined_clean = preprocessor.remove_outliers(combined)
        X = combined_clean.drop(columns=[target_column])
        y = combined_clean[target_column]

    # Normalize features if requested
    if normalize:
        numeric_columns = X.select_dtypes(include=[np.number]).columns
        X[numeric_columns] = preprocessor.normalize_features(X[numeric_columns])

    # Encode categorical target if needed
    if y.dtype == "object":
        y, _ = preprocessor.encode_categorical_variables(y)

    # Split the data
    return preprocessor.split_train_test(X, y, test_size=test_size)
