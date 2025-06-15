# Random Forest

## Overview

Random Forest is an ensemble learning method that operates by constructing multiple decision trees during training and outputting the class (classification) or mean prediction (regression) of the individual trees. This approach combines the simplicity of decision trees with the power of ensemble methods.

## Principles

Random Forest works through the following mechanisms:

1. **Bootstrap Aggregating (Bagging)**: Creates multiple datasets by sampling with replacement
2. **Random Feature Selection**: Each tree considers only a subset of features for splitting
3. **Ensemble Averaging**: Combines predictions from all trees to make a final prediction
4. **Out-of-Bag Evaluation**: Uses samples not selected during bagging to estimate performance

## Advantages

- Reduces overfitting compared to individual decision trees
- Provides excellent accuracy for many problems
- Handles large datasets with higher dimensionality effectively
- Maintains accuracy even when a large portion of data is missing
- Provides estimates of feature importance

## Limitations

- Less interpretable than individual decision trees
- Computationally more intensive than simple models
- May overfit on noisy datasets
- Not optimal for linear relationships (more complex than necessary)
- Requires more memory and computational resources

## Implementation

Our implementation builds upon our decision tree implementation, creating an ensemble of trees with randomized feature selection.

```python
from machine_learning_model.supervised.random_forest import RandomForestClassifier, RandomForestRegressor

# Classification example
clf = RandomForestClassifier(n_estimators=100, max_depth=10)
clf.fit(X_train, y_train)
predictions = clf.predict(X_test)
accuracy = clf.evaluate(X_test, y_test)

# Regression example
regressor = RandomForestRegressor(n_estimators=100, max_depth=10)
regressor.fit(X_train, y_train)
predictions = regressor.predict(X_test)
mse = regressor.evaluate(X_test, y_test)

# Feature importance
importances = clf.feature_importances_
```

## Hyperparameters

| Parameter           | Description                                   | Default                                          |
| ------------------- | --------------------------------------------- | ------------------------------------------------ |
| `n_estimators`      | Number of trees in the forest                 | `100`                                            |
| `max_depth`         | Maximum depth of each tree                    | `None` (unlimited)                               |
| `min_samples_split` | Minimum samples required to split a node      | `2`                                              |
| `min_samples_leaf`  | Minimum samples required at a leaf node       | `1`                                              |
| `max_features`      | Number of features to consider for best split | `'sqrt'` (classification), `'auto'` (regression) |
| `bootstrap`         | Whether to use bootstrap samples              | `True`                                           |
| `oob_score`         | Whether to use out-of-bag samples             | `False`                                          |

## Feature Importance

Random Forest provides a natural way to measure feature importance:

```python
importances = clf.feature_importances_
indices = np.argsort(importances)[::-1]

# Print feature ranking
for f in range(X.shape[1]):
    print(f"{f + 1}. {feature_names[indices[f]]} ({importances[indices[f]]})")
```

## Use Cases

- Credit scoring
- Disease prediction
- Customer churn prediction
- Stock price prediction
- Image classification
- Recommendation systems

## References

- Breiman, L. (2001). Random Forests. Machine Learning, 45(1), 5-32.
- Ho, T. K. (1995). Random Decision Forests. Proceedings of the 3rd International Conference on Document Analysis and Recognition.
