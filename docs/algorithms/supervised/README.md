# Supervised Learning Algorithms

This directory contains documentation for supervised learning algorithms - those that learn from labeled training data to predict outcomes on unseen data.

## Available Algorithms

- [**Decision Trees**](decision_trees.md) - Tree-based models that split data based on feature values
  - Classification and regression
  - Intuitive and easy to interpret
  - CART (Classification and Regression Trees) implementation

- [**Random Forest**](random_forest.md) - Ensemble of decision trees
  - Reduced overfitting compared to individual trees
  - Feature importance estimation
  - Robust to outliers and noise

- [**Support Vector Machines (SVM)**](svm.md) - Finds optimal hyperplane to separate classes
  - Multiple kernel functions
  - Effective in high-dimensional spaces
  - Works for both classification and regression

- [**XGBoost**](xgboost.md) - Gradient boosting implementation
  - High performance on structured/tabular data
  - Regularization to prevent overfitting
  - Efficient handling of missing values

## Common Interface

All supervised learning algorithms implement this common interface:

```python
# Train the model
model.fit(X_train, y_train)

# Make predictions
predictions = model.predict(X_test)

# Evaluate model performance
score = model.evaluate(X_test, y_test)
```

## When to Use

- **Decision Trees**: When interpretability is important
- **Random Forest**: When you need better performance than decision trees and robustness to overfitting
- **SVM**: For high-dimensional data with clear separation
- **XGBoost**: For tabular data when you need state-of-the-art performance

## References

See individual algorithm pages for specific references and academic papers.
