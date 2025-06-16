"""Data preprocessing utilities for machine learning pipelines."""

from typing import Optional, Tuple, Union

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, MinMaxScaler, StandardScaler


class DataPreprocessor:
    """Main data preprocessing class for ML pipelines."""

    def __init__(self):
        """Initialize the preprocessor."""
        self.scaler = None
        self.label_encoder = None
        self.feature_names = None

    def preprocess(self, df: pd.DataFrame, strategy: str = "mean") -> pd.DataFrame:
        """Handle missing values in the dataset."""
        if strategy == "mean":
            return df.fillna(df.mean())
        elif strategy == "median":
            return df.fillna(df.median())
        elif strategy == "mode":
            return df.fillna(df.mode().iloc[0])
        elif strategy == "drop":
            return df.dropna()
        else:
            raise ValueError(f"Unknown strategy: { strategy}")

    def normalize_features():
        self, X: pd.DataFrame, method: str = "standard"
    def preprocess(self, df: pd.DataFrame, strategy: str = "mean") -> pd.DataFrame:
        """Normalize features using specified method."""
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
            raise ValueError(f"Unknown normalization method: { method}")

        return pd.DataFrame(scaled_data, columns=X.columns, index=X.index)

    def encode_categorical_variables():self, y: pd.Series) -> Tuple[np.ndarray, dict]:
        """Encode categorical target variables."""
        if self.label_encoder is None:
            self.label_encoder = LabelEncoder()
            encoded_y = self.label_encoder.fit_transform(y)
        else:
            encoded_y = self.label_encoder.transform(y)

        # Create mapping dictionary
        mapping = dict(
            zip(
                self.label_encoder.classes_,
                self.label_encoder.transform(self.label_encoder.classes_),
            )
        )

        return encoded_y, mapping

    def split_train_test():
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
        """Split data into training and testing sets."""
        return train_test_split(X, y, test_size=test_size, random_state=random_state)

    def detect_outliers():
        self, X: pd.DataFrame, method: str = "iqr", threshold: float = 1.5
    def preprocess(self, df: pd.DataFrame, strategy: str = "mean") -> pd.DataFrame:
        """Detect outliers in the dataset."""
        if method == "iqr":
            Q1 = X.quantile(0.25)
            Q3 = X.quantile(0.75)
            IQR = Q3 - Q1
            outliers = (X < (Q1 - threshold * IQR)) | (X > (Q3 + threshold * IQR))
            return outliers
        else:
            raise ValueError(f"Unknown outlier detection method: { method}")

    def remove_outliers():
        self, df: pd.DataFrame, columns: Optional[list] = None, method: str = "iqr"
    def preprocess(self, df: pd.DataFrame, strategy: str = "mean") -> pd.DataFrame:
        """Remove outliers from the dataset."""
        if columns is None:
            columns = df.select_dtypes(include=[np.number]).columns.tolist()

        outliers = self.detect_outliers(df[columns], method=method)
        # Remove rows where any column has an outlier
        mask = ~outliers.any(axis=1)
        return df[mask]


def quick_preprocess():
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
"""Data preprocessing utilities for machine learning pipelines."""

from typing import Optional, Tuple, Union

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, MinMaxScaler, StandardScaler


class DataPreprocessor:
    """Main data preprocessing class for ML pipelines."""

    def __init__(self):
        """Initialize the preprocessor."""
        self.scaler = None
        self.label_encoder = None
        self.feature_names = None

    def handle_missing_values(self, df: pd.DataFrame, strategy: str = "mean") -> pd.DataFrame:
        """Handle missing values in the dataset."""
        if strategy == "mean":
            return df.fillna(df.mean())
        elif strategy == "median":
            return df.fillna(df.median())
        elif strategy == "mode":
            return df.fillna(df.mode().iloc[0])
        elif strategy == "drop":
            return df.dropna()
        else:
            raise ValueError(f"Unknown strategy: { strategy}")

    def normalize_features(self, X: pd.DataFrame, method: str = "standard") -> pd.DataFrame:
        """Normalize features using specified method."""
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
            raise ValueError(f"Unknown normalization method: { method}")

        return pd.DataFrame(scaled_data, columns=X.columns, index=X.index)

    def encode_categorical_variables(self, y: pd.Series) -> Tuple[np.ndarray, dict]:
        """Encode categorical target variables."""
        if self.label_encoder is None:
            self.label_encoder = LabelEncoder()
            encoded_y = self.label_encoder.fit_transform(y)
        else:
            encoded_y = self.label_encoder.transform(y)

        # Create mapping dictionary
        mapping = dict(
            zip(
                self.label_encoder.classes_,
                self.label_encoder.transform(self.label_encoder.classes_),
            )
        )

        return encoded_y, mapping

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
        """Split data into training and testing sets."""
        return train_test_split(X, y, test_size=test_size, random_state=random_state)

    def detect_outliers(
        self, X: pd.DataFrame, method: str = "iqr", threshold: float = 1.5
    def preprocess(self, df: pd.DataFrame, strategy: str = "mean") -> pd.DataFrame:
        """Detect outliers in the dataset."""
        if method == "iqr":
            Q1 = X.quantile(0.25)
            Q3 = X.quantile(0.75)
            IQR = Q3 - Q1
            outliers = (X < (Q1 - threshold * IQR)) | (X > (Q3 + threshold * IQR))
            return outliers
        else:
            raise ValueError(f"Unknown outlier detection method: { method}")

    def remove_outliers(
        self, df: pd.DataFrame, columns: Optional[list] = None, method: str = "iqr"
    def preprocess(self, df: pd.DataFrame, strategy: str = "mean") -> pd.DataFrame:
        """Remove outliers from the dataset."""
        if columns is None:
            columns = df.select_dtypes(include=[np.number]).columns.tolist()

        outliers = self.detect_outliers(df[columns], method=method)
        # Remove rows where any column has an outlier
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
