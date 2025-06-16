# API Reference

This directory contains comprehensive API documentation for the Machine Learning Model package.

## Module Structure

- [**data/**](data/) - Data loading and processing utilities
  - [loaders.py](data/loaders.md) - Functions to load datasets
  - [preprocessors.py](data/preprocessors.md) - Data preprocessing classes and functions
  - [validators.py](data/validators.md) - Data validation utilities

- [**supervised/**](supervised/) - Supervised learning algorithms
  - [decision_tree.py](supervised/decision_tree.md) - Decision Tree implementation
  - [random_forest.py](supervised/random_forest.md) - Random Forest implementation
  - [svm.py](supervised/svm.md) - Support Vector Machines
  - [xgboost_model.py](supervised/xgboost.md) - XGBoost wrapper

- [**unsupervised/**](unsupervised/) - Unsupervised learning algorithms
  - [kmeans.py](unsupervised/kmeans.md) - K-means clustering
  - [dbscan.py](unsupervised/dbscan.md) - DBSCAN clustering
  - [pca.py](unsupervised/pca.md) - Principal Component Analysis

- [**semi_supervised/**](semi_supervised/) - Semi-supervised learning algorithms
  - [label_propagation.py](semi_supervised/label_propagation.md) - Label Propagation
  - [semi_supervised_svm.py](semi_supervised/semi_supervised_svm.md) - S3VM

- [**evaluation/**](evaluation/) - Model evaluation utilities
  - [metrics.py](evaluation/metrics.md) - Performance metrics
  - [cross_validation.py](evaluation/cross_validation.md) - Cross-validation tools

- [**utils/**](utils/) - General purpose utilities
  - [visualization.py](utils/visualization.md) - Plotting and visualization tools
  - [feature_selection.py](utils/feature_selection.md) - Feature selection methods

## Usage Examples

```python
# Data loading and preprocessing
from machine_learning_model.data.loaders import load_iris_dataset
from machine_learning_model.data.preprocessors import quick_preprocess

# Load data
data = load_iris_dataset()
X_train, X_test, y_train, y_test = quick_preprocess(data, target_column='species')

# Model training
from machine_learning_model.supervised.random_forest import RandomForestClassifier
model = RandomForestClassifier(n_estimators=100)
model.fit(X_train, y_train)

# Prediction and evaluation
predictions = model.predict(X_test)
accuracy = model.evaluate(X_test, y_test)
print(f"Accuracy: {accuracy:.4f}")
```

## Common Interfaces

All models follow a consistent interface with these core methods:
- `fit(X, y)`: Train the model
- `predict(X)`: Make predictions
- `evaluate(X, y)`: Assess model performance

See individual module documentation for details on additional methods and parameters.
