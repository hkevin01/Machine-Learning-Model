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

    # ------------------------------------------------------------------
    # Methods expected by the test suite (test_data/test_validators.py)
    # ------------------------------------------------------------------

    def validate_data_types(
        self, df: pd.DataFrame, expected_types: Dict[str, str]
    ) -> Dict[str, Any]:
        """
        Validate that each column has the expected dtype kind.
        expected_types maps column name → one of 'int', 'float', 'object', etc.
        Returns {'passed': bool, 'summary': {...}}.
        """
        results: Dict[str, Any] = {}
        all_passed = True
        for col, expected in expected_types.items():
            if col not in df.columns:
                results[col] = {"passed": False, "reason": "column missing"}
                all_passed = False
                continue
            actual = str(df[col].dtype)
            # Accept if the expected kind appears anywhere in the dtype string
            passed = expected in actual or (
                expected == "int" and pd.api.types.is_integer_dtype(df[col])
            ) or (
                expected == "float" and pd.api.types.is_float_dtype(df[col])
            ) or (
                expected == "object" and pd.api.types.is_object_dtype(df[col])
            )
            results[col] = {"passed": passed, "actual": actual, "expected": expected}
            if not passed:
                all_passed = False
        return {"passed": all_passed, "summary": results}

    def validate_target_distribution(self, target: pd.Series) -> Dict[str, Any]:
        """
        Compute class distribution and balance ratio for a target Series.
        Returns {'distribution': {class: count}, 'class_balance': float}.
        """
        counts = target.value_counts().to_dict()
        max_count = max(counts.values()) if counts else 1
        min_count = min(counts.values()) if counts else 1
        balance = min_count / max_count if max_count > 0 else 1.0
        return {"distribution": counts, "class_balance": balance}

    def detect_outliers(self, df: pd.DataFrame) -> Dict[str, Any]:
        """
        Detect outliers using IQR for all numeric columns.
        Returns {'total_outliers': int, 'outliers_by_column': {col: [indices]}}.
        """
        numeric = df.select_dtypes(include=[np.number])
        outliers_by_col: Dict[str, list] = {}
        for col in numeric.columns:
            q1 = numeric[col].quantile(0.25)
            q3 = numeric[col].quantile(0.75)
            iqr = q3 - q1
            mask = (numeric[col] < q1 - 1.5 * iqr) | (numeric[col] > q3 + 1.5 * iqr)
            outliers_by_col[col] = list(numeric[col][mask].index)
        total = sum(len(v) for v in outliers_by_col.values())
        return {"total_outliers": total, "outliers_by_column": outliers_by_col}

    def validate_dataset_completeness(self, df: pd.DataFrame) -> Dict[str, Any]:
        """
        Check for missing values and report basic shape statistics.
        Returns {'statistics': {'total_rows', 'total_columns', 'missing_values',
                                'missing_percentage', 'complete_rows'}}.
        """
        missing = int(df.isnull().sum().sum())
        total_cells = df.shape[0] * df.shape[1]
        complete_rows = int((~df.isnull().any(axis=1)).sum())
        return {
            "statistics": {
                "total_rows": df.shape[0],
                "total_columns": df.shape[1],
                "missing_values": missing,
                "missing_percentage": (missing / total_cells * 100) if total_cells else 0.0,
                "complete_rows": complete_rows,
            }
        }

    def validate_column_names(
        self, df: pd.DataFrame, required_columns: List[str]
    ) -> Dict[str, Any]:
        """
        Check that all required columns are present in df.
        Returns {'passed': bool, 'missing_columns': [str], 'extra_columns': [str]}.
        """
        present = set(df.columns)
        required = set(required_columns)
        missing = list(required - present)
        extra = list(present - required)
        return {
            "passed": len(missing) == 0,
            "missing_columns": missing,
            "extra_columns": extra,
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


def validate_ml_dataset(
    df: pd.DataFrame,
    target_column: str,
    required_columns: Optional[List[str]] = None,
    min_samples: int = 10,
) -> Dict[str, Any]:
    """
    Comprehensive ML dataset validation combining all checks.

    Parameters
    ----------
    df               : Input DataFrame to validate.
    target_column    : Name of the target/label column.
    required_columns : List of column names that must be present.
                       Defaults to all columns in df.
    min_samples      : Minimum number of rows required.

    Returns
    -------
    dict with keys:
      'overall_passed' : bool
      'validations'    : dict of individual check results
      'summary'        : {'total_checks', 'passed_checks', 'failed_checks',
                          'total_warnings'}
    """
    validator = DataValidator()
    validations: Dict[str, Any] = {}
    warnings = 0

    # --- column presence ---
    cols_to_check = required_columns if required_columns is not None else list(df.columns)
    col_result = validator.validate_column_names(df, cols_to_check)
    validations["column_names"] = col_result
    if not col_result["passed"]:
        warnings += 1

    # --- target column exists ---
    target_present = target_column in df.columns
    validations["target_present"] = {"passed": target_present}
    if not target_present:
        warnings += 1

    # --- completeness ---
    completeness = validator.validate_dataset_completeness(df)
    validations["completeness"] = completeness
    if completeness["statistics"]["missing_values"] > 0:
        warnings += 1

    # --- minimum samples ---
    enough_samples = len(df) >= min_samples
    validations["min_samples"] = {"passed": enough_samples, "actual": len(df), "required": min_samples}
    if not enough_samples:
        warnings += 1

    # --- target distribution (only when target exists) ---
    if target_present:
        dist = validator.validate_target_distribution(df[target_column])
        validations["target_distribution"] = dist
        # Warn if heavily imbalanced (balance ratio < 0.1)
        if dist["class_balance"] < 0.1:
            warnings += 1

    # --- outliers (numeric features only, excluding target) ---
    feature_df = df.drop(columns=[target_column], errors="ignore")
    if not feature_df.select_dtypes(include=[np.number]).empty:
        outlier_result = validator.detect_outliers(feature_df)
        validations["outliers"] = outlier_result
        if outlier_result["total_outliers"] > 0:
            warnings += 1

    # --- variance check (constant columns) ---
    numeric_df = df.select_dtypes(include=[np.number])
    zero_var_cols = [c for c in numeric_df.columns if numeric_df[c].var() == 0]
    validations["variance"] = {"zero_variance_columns": zero_var_cols, "passed": len(zero_var_cols) == 0}
    if zero_var_cols:
        warnings += 1

    total_checks = len(validations)
    passed_checks = sum(
        1 for v in validations.values()
        if isinstance(v, dict) and v.get("passed", True)
    )

    overall_passed = (
        col_result["passed"]
        and target_present
        and enough_samples
    )

    return {
        "overall_passed": overall_passed,
        "validations": validations,
        "summary": {
            "total_checks": total_checks,
            "passed_checks": passed_checks,
            "failed_checks": total_checks - passed_checks,
            "total_warnings": warnings,
        },
    }

