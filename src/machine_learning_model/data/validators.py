"""Data validation utilities for machine learning datasets."""

from dataclasses import dataclass
from enum import Enum
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd


class DataType(Enum):
    """Supported data types for validation."""
    NUMERICAL = "numerical"
    CATEGORICAL = "categorical"
    TEXT = "text"
    DATETIME = "datetime"


@dataclass
class ValidationResult:
    """Result of a validation check."""
    is_valid: bool
    message: str
    details: Optional[Dict[str, Any]] = None


class DataValidator:
    """Comprehensive data validation for ML datasets."""

    def __init__(self):
        self.validation_results: List[ValidationResult] = []

    def validate_dataset(self, df: pd.DataFrame, expected_schema: Dict[str, DataType]) -> List[ValidationResult]:
        """Validate a dataset against expected schema."""
        results = []
        
        # Basic structure validation
        results.extend(self._validate_structure(df))
        
        # Schema validation
        results.extend(self._validate_schema(df, expected_schema))
        
        # Data quality validation
        results.extend(self._validate_data_quality(df))
        
        # Statistical validation
        results.extend(self._validate_statistics(df))
        
        self.validation_results = results
        return results

    def _validate_structure(self, df: pd.DataFrame) -> List[ValidationResult]:
        """Validate basic dataset structure."""
        results = []
        
        # Check if DataFrame is empty
        if df.empty:
            results.append(ValidationResult(
                is_valid=False,
                message="Dataset is empty",
                details={"shape": df.shape}
            ))
        else:
            results.append(ValidationResult(
                is_valid=True,
                message=f"Dataset has {len(df)} rows and {len(df.columns)} columns",
                details={"shape": df.shape}
            ))
        
        # Check for duplicate rows
        duplicate_count = df.duplicated().sum()
        if duplicate_count > 0:
            results.append(ValidationResult(
                is_valid=False,
                message=f"Found {duplicate_count} duplicate rows",
                details={"duplicate_count": duplicate_count}
            ))
        else:
            results.append(ValidationResult(
                is_valid=True,
                message="No duplicate rows found"
            ))
        
        return results

    def _validate_schema(self, df: pd.DataFrame, expected_schema: Dict[str, DataType]) -> List[ValidationResult]:
        """Validate dataset schema against expected types."""
        results = []
        
        for column, expected_type in expected_schema.items():
            if column not in df.columns:
                results.append(ValidationResult(
                    is_valid=False,
                    message=f"Missing required column: {column}",
                    details={"missing_column": column}
                ))
                continue
            
            # Validate data type
            if expected_type == DataType.NUMERICAL:
                if not pd.api.types.is_numeric_dtype(df[column]):
                    results.append(ValidationResult(
                        is_valid=False,
                        message=f"Column {column} should be numerical but is {df[column].dtype}",
                        details={"column": column, "actual_type": str(df[column].dtype)}
                    ))
                else:
                    results.append(ValidationResult(
                        is_valid=True,
                        message=f"Column {column} is correctly typed as numerical"
                    ))
            
            elif expected_type == DataType.CATEGORICAL:
                if not pd.api.types.is_categorical_dtype(df[column]) and not pd.api.types.is_object_dtype(df[column]):
                    results.append(ValidationResult(
                        is_valid=False,
                        message=f"Column {column} should be categorical but is {df[column].dtype}",
                        details={"column": column, "actual_type": str(df[column].dtype)}
                    ))
                else:
                    results.append(ValidationResult(
                        is_valid=True,
                        message=f"Column {column} is correctly typed as categorical"
                    ))
        
        return results

    def _validate_data_quality(self, df: pd.DataFrame) -> List[ValidationResult]:
        """Validate data quality metrics."""
        results = []
        
        # Check for missing values
        missing_data = df.isnull().sum()
        total_missing = missing_data.sum()
        
        if total_missing > 0:
            missing_percentage = (total_missing / (len(df) * len(df.columns))) * 100
            results.append(ValidationResult(
                is_valid=False,
                message=f"Found {total_missing} missing values ({missing_percentage:.2f}%)",
                details={"missing_counts": missing_data.to_dict(), "total_missing": total_missing}
            ))
        else:
            results.append(ValidationResult(
                is_valid=True,
                message="No missing values found"
            ))
        
        # Check for infinite values in numerical columns
        numerical_columns = df.select_dtypes(include=[np.number]).columns
        infinite_counts = {}
        
        for col in numerical_columns:
            infinite_count = np.isinf(df[col]).sum()
            if infinite_count > 0:
                infinite_counts[col] = infinite_count
        
        if infinite_counts:
            results.append(ValidationResult(
                is_valid=False,
                message=f"Found infinite values in columns: {list(infinite_counts.keys())}",
                details={"infinite_counts": infinite_counts}
            ))
        else:
            results.append(ValidationResult(
                is_valid=True,
                message="No infinite values found in numerical columns"
            ))
        
        return results

    def _validate_statistics(self, df: pd.DataFrame) -> List[ValidationResult]:
        """Validate statistical properties of the dataset."""
        results = []
        
        numerical_columns = df.select_dtypes(include=[np.number]).columns
        
        for col in numerical_columns:
            # Check for zero variance (constant columns)
            if df[col].var() == 0:
                results.append(ValidationResult(
                    is_valid=False,
                    message=f"Column {col} has zero variance (constant values)",
                    details={"column": col, "value": df[col].iloc[0]}
                ))
            else:
                results.append(ValidationResult(
                    is_valid=True,
                    message=f"Column {col} has non-zero variance"
                ))
            
            # Check for extreme outliers (beyond 3 standard deviations)
            mean_val = df[col].mean()
            std_val = df[col].std()
            outliers = df[col][(df[col] < mean_val - 3 * std_val) | (df[col] > mean_val + 3 * std_val)]
            
            if len(outliers) > 0:
                results.append(ValidationResult(
                    is_valid=False,
                    message=f"Column {col} has {len(outliers)} extreme outliers",
                    details={"column": col, "outlier_count": len(outliers), "outlier_values": outliers.tolist()}
                ))
        
        return results

    def get_summary(self) -> Dict[str, Any]:
        """Get a summary of all validation results."""
        if not self.validation_results:
            return {"status": "No validation performed"}
        
        total_checks = len(self.validation_results)
        passed_checks = sum(1 for result in self.validation_results if result.is_valid)
        failed_checks = total_checks - passed_checks
        
        return {
            "total_checks": total_checks,
            "passed_checks": passed_checks,
            "failed_checks": failed_checks,
            "success_rate": passed_checks / total_checks if total_checks > 0 else 0,
            "is_valid": failed_checks == 0,
            "failed_messages": [result.message for result in self.validation_results if not result.is_valid]
        }


def validate_iris_dataset(df: pd.DataFrame) -> List[ValidationResult]:
    """Validate Iris dataset specifically."""
    expected_schema = {
        "sepal_length": DataType.NUMERICAL,
        "sepal_width": DataType.NUMERICAL,
        "petal_length": DataType.NUMERICAL,
        "petal_width": DataType.NUMERICAL,
        "species": DataType.CATEGORICAL
    }
    
    validator = DataValidator()
    return validator.validate_dataset(df, expected_schema)


def validate_wine_dataset(df: pd.DataFrame) -> List[ValidationResult]:
    """Validate Wine dataset specifically."""
    expected_schema = {
        "alcohol": DataType.NUMERICAL,
        "malic_acid": DataType.NUMERICAL,
        "ash": DataType.NUMERICAL,
        "alcalinity_of_ash": DataType.NUMERICAL,
        "magnesium": DataType.NUMERICAL,
        "total_phenols": DataType.NUMERICAL,
        "flavanoids": DataType.NUMERICAL,
        "nonflavanoid_phenols": DataType.NUMERICAL,
        "proanthocyanins": DataType.NUMERICAL,
        "color_intensity": DataType.NUMERICAL,
        "hue": DataType.NUMERICAL,
        "od280_od315_of_diluted_wines": DataType.NUMERICAL,
        "proline": DataType.NUMERICAL,
        "target": DataType.CATEGORICAL
    }
    
    validator = DataValidator()
    return validator.validate_dataset(df, expected_schema)


def validate_housing_dataset(df: pd.DataFrame) -> List[ValidationResult]:
    """Validate Housing dataset specifically."""
    expected_schema = {
        "longitude": DataType.NUMERICAL,
        "latitude": DataType.NUMERICAL,
        "housing_median_age": DataType.NUMERICAL,
        "total_rooms": DataType.NUMERICAL,
        "total_bedrooms": DataType.NUMERICAL,
        "population": DataType.NUMERICAL,
        "households": DataType.NUMERICAL,
        "median_income": DataType.NUMERICAL,
        "median_house_value": DataType.NUMERICAL
    }
    
    validator = DataValidator()
    return validator.validate_dataset(df, expected_schema)
    return validator.validate_dataset(df, expected_schema)
    return validator.validate_dataset(df, expected_schema)
