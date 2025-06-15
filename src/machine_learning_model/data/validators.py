"""Data validation utilities for machine learning pipelines."""

from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd


class DataValidator:
    """Data validation class for ML pipelines."""

    def validate_data_types(
        self, df: pd.DataFrame, expected_types: Dict[str, str]
    ) -> Dict[str, Any]:
        """Validate data types of DataFrame columns."""
        validation_results = {
            "passed": True,
            "errors": [],
            "warnings": [],
            "summary": {},
        }

        for column, expected_type in expected_types.items():
            if column not in df.columns:
                validation_results["errors"].append(
                    f"Column '{ column}' not found in DataFrame"
                )
                validation_results["passed"] = False
                continue

            actual_type = str(df[column].dtype)
            validation_results["summary"][column] = {
                "expected": expected_type,
                "actual": actual_type,
                "match": expected_type in actual_type,
            }

            if expected_type not in actual_type:
                validation_results["warnings"].append(
                    f"Column '{ column}': expected { expected_type}, got { actual_type}"
                )

        return validation_results

    def validate_target_distribution(
        self, y: pd.Series, min_samples_per_class: int = 5
    ) -> Dict[str, Any]:
        """Validate target variable distribution."""
        validation_results = {
            "passed": True,
            "errors": [],
            "warnings": [],
            "distribution": {},
            "class_balance": {},
        }

        # Get value counts
        value_counts = y.value_counts()
        validation_results["distribution"] = value_counts.to_dict()

        # Check minimum samples per class
        insufficient_classes = value_counts[value_counts < min_samples_per_class]
        if len(insufficient_classes) > 0:
            validation_results["errors"].extend(
                [
                    f"Class '{ cls}' has only { count} samples (minimum: { min_samples_per_class})"
                    for cls, count in insufficient_classes.items()
                ]
            )
            validation_results["passed"] = False

        # Calculate class balance
        total_samples = len(y)
        for class_name, count in value_counts.items():
            percentage = (count / total_samples) * 100
            validation_results["class_balance"][class_name] = {
                "count": count,
                "percentage": percentage,
            }

            # Warn about severe imbalance
            if percentage < 5:
                validation_results["warnings"].append(
                    f"Class '{ class_name}' represents only { percentage:.1f}% of data"
                )

        return validation_results

    def detect_outliers(
        self, df: pd.DataFrame, columns: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """Detect outliers in numeric columns."""
        if columns is None:
            columns = df.select_dtypes(include=[np.number]).columns.tolist()

        outlier_results = {
            "total_outliers": 0,
            "outliers_by_column": {},
            "outlier_indices": set(),
        }

        for column in columns:
            if column not in df.columns:
                continue

            Q1 = df[column].quantile(0.25)
            Q3 = df[column].quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR

            outliers = df[(df[column] < lower_bound) | (df[column] > upper_bound)]
            outlier_indices = outliers.index.tolist()

            outlier_results["outliers_by_column"][column] = {
                "count": len(outliers),
                "percentage": (len(outliers) / len(df)) * 100,
                "indices": outlier_indices,
                "bounds": {"lower": lower_bound, "upper": upper_bound},
            }

            outlier_results["outlier_indices"].update(outlier_indices)

        outlier_results["total_outliers"] = len(outlier_results["outlier_indices"])
        outlier_results["outlier_indices"] = list(outlier_results["outlier_indices"])

        return outlier_results

    def validate_dataset_completeness(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Validate dataset completeness and quality."""
        validation_results = {
            "passed": True,
            "errors": [],
            "warnings": [],
            "statistics": {
                "total_rows": len(df),
                "total_columns": len(df.columns),
                "missing_values": {},
                "duplicate_rows": 0,
                "data_types": {},
            },
        }

        # Check for missing values
        missing_counts = df.isnull().sum()
        # Removed unused variable: total_missing

        for column, missing_count in missing_counts.items():
            if missing_count > 0:
                percentage = (missing_count / len(df)) * 100
                validation_results["statistics"]["missing_values"][column] = {
                    "count": missing_count,
                    "percentage": percentage,
                }

                if percentage > 50:
                    validation_results["errors"].append(
                        f"Column '{ column}' has { percentage:.1f}% missing values"
                    )
                    validation_results["passed"] = False
                elif percentage > 10:
                    validation_results["warnings"].append(
                        f"Column '{ column}' has { percentage:.1f}% missing values"
                    )

        # Check for duplicate rows
        duplicate_count = df.duplicated().sum()
        validation_results["statistics"]["duplicate_rows"] = duplicate_count
        if duplicate_count > 0:
            validation_results["warnings"].append(
                f"Found { duplicate_count} duplicate rows"
            )

        # Data type summary
        validation_results["statistics"][
            "data_types"
        ] = df.dtypes.value_counts().to_dict()

        # Empty dataset check
        if len(df) == 0:
            validation_results["errors"].append("Dataset is empty")
            validation_results["passed"] = False

        return validation_results

    def validate_column_names(
        self, df: pd.DataFrame, required_columns: List[str]
    ) -> Dict[str, Any]:
        """Validate that required columns exist in the DataFrame."""
        validation_results = {
            "passed": True,
            "errors": [],
            "missing_columns": [],
            "extra_columns": [],
            "column_mapping": {},
        }

        # Check for missing required columns
        missing_columns = [col for col in required_columns if col not in df.columns]
        validation_results["missing_columns"] = missing_columns

        if missing_columns:
            validation_results["errors"].extend(
                [f"Required column '{ col}' not found" for col in missing_columns]
            )
            validation_results["passed"] = False

        # Find extra columns
        extra_columns = [col for col in df.columns if col not in required_columns]
        validation_results["extra_columns"] = extra_columns

        # Create column mapping
        for col in df.columns:
            validation_results["column_mapping"][col] = {
                "required": col in required_columns,
                "dtype": str(df[col].dtype),
                "non_null_count": df[col].count(),
            }

        return validation_results


def validate_ml_dataset(
    df: pd.DataFrame,
    target_column: str,
    required_columns: Optional[List[str]] = None,
    min_samples: int = 10,
    max_missing_percentage: float = 20.0,
) -> Dict[str, Any]:
    """Comprehensive validation for ML datasets."""
    validator = DataValidator()

    validation_results = {
        "overall_passed": True,
        "validations": {},
        "summary": {"total_errors": 0, "total_warnings": 0},
    }

    # Dataset completeness
    completeness = validator.validate_dataset_completeness(df)
    validation_results["validations"]["completeness"] = completeness
    if not completeness["passed"]:
        validation_results["overall_passed"] = False

    # Column validation
    if required_columns:
        column_validation = validator.validate_column_names(df, required_columns)
        validation_results["validations"]["columns"] = column_validation
        if not column_validation["passed"]:
            validation_results["overall_passed"] = False

    # Target validation
    if target_column in df.columns:
        target_validation = validator.validate_target_distribution(df[target_column])
        validation_results["validations"]["target"] = target_validation
        if not target_validation["passed"]:
            validation_results["overall_passed"] = False

    # Outlier detection
    outlier_detection = validator.detect_outliers(df)
    validation_results["validations"]["outliers"] = outlier_detection

    # Calculate summary
    for validation_name, validation_data in validation_results["validations"].items():
        if "errors" in validation_data:
            validation_results["summary"]["total_errors"] += len(
                validation_data["errors"]
            )
        if "warnings" in validation_data:
            validation_results["summary"]["total_warnings"] += len(
                validation_data["warnings"]
            )

    return validation_results
