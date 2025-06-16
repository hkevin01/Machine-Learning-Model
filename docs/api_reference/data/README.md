# Data Module API

This directory contains API documentation for the data loading, preprocessing, and validation utilities.

## Modules

- [**loaders.py**](loaders.md) - Functions to load various datasets
  - Built-in dataset loaders
  - Synthetic data generators
  - External data importers

- [**preprocessors.py**](preprocessors.md) - Data preprocessing utilities
  - `DataPreprocessor` class
  - Missing value handling
  - Feature normalization
  - Categorical encoding
  - Outlier detection

- [**validators.py**](validators.md) - Data validation utilities
  - `DataValidator` class
  - Dataset completeness checking
  - Target distribution analysis
  - Schema validation

## Key Classes and Functions

### Data Loading

```python
from machine_learning_model.data.loaders import load_iris_dataset, load_wine_dataset

# Load datasets
iris = load_iris_dataset()
wine = load_wine_dataset()
```

### Data Preprocessing

```python
from machine_learning_model.data.preprocessors import DataPreprocessor, quick_preprocess

# Quick preprocessing
X_train, X_test, y_train, y_test = quick_preprocess(
    data,
    target_column='target',
    normalize=True
)

# Step-by-step preprocessing
preprocessor = DataPreprocessor()
X_clean = preprocessor.handle_missing_values(X)
X_normalized = preprocessor.normalize_features(X_clean)
```

### Data Validation

```python
from machine_learning_model.data.validators import validate_ml_dataset

# Validate dataset
results = validate_ml_dataset(
    data,
    target_column='target',
    required_columns=['feature1', 'feature2']
)

if results['overall_passed']:
    print("Dataset validation passed!")
else:
    print(f"Found {results['summary']['total_errors']} errors")
```

## Integration with ML Workflow

The data module is typically the first step in the machine learning workflow:

1. **Load data** using the loaders module
2. **Validate data** using the validators module
3. **Preprocess data** using the preprocessors module
4. **Train models** using the processed data

For complete code examples, see the individual module documentation.
