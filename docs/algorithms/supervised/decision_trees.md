# Decision Trees

## Overview

Decision Trees are versatile machine learning algorithms used for both classification and regression tasks. They create a model that predicts the value of a target variable by learning simple decision rules inferred from the data features.

## Principles

A Decision Tree recursively partitions the feature space, creating a tree-like structure of decisions:

1. **Root Node**: Represents the entire dataset
2. **Decision Nodes**: Points where the data is split based on feature values
3. **Leaf Nodes**: Terminal nodes that provide the prediction

## Advantages

- Intuitive and easy to interpret
- Requires little data preprocessing
- Can handle both numerical and categorical data
- Automatically performs feature selection
- Handles non-linear relationships well

## Limitations

- Prone to overfitting with complex trees
- Can be unstable (small variations in data might result in a completely different tree)
- Biased toward features with more levels (when dealing with categorical variables)
- May create biased trees if classes are imbalanced

## Implementation

Our implementation uses the CART (Classification and Regression Trees) algorithm with Gini impurity for classification and mean squared error for regression.

```python
from machine_learning_model.supervised.decision_tree import DecisionTreeClassifier, DecisionTreeRegressor

# Classification example
clf = DecisionTreeClassifier(max_depth=5, min_samples_split=2)
clf.fit(X_train, y_train)
predictions = clf.predict(X_test)
accuracy = clf.evaluate(X_test, y_test)

# Regression example
regressor = DecisionTreeRegressor(max_depth=5, min_samples_split=2)
regressor.fit(X_train, y_train)
predictions = regressor.predict(X_test)
mse = regressor.evaluate(X_test, y_test)
```

## Hyperparameters

| Parameter           | Description                                   | Default                                             |
| ------------------- | --------------------------------------------- | --------------------------------------------------- |
| `max_depth`         | Maximum depth of the tree                     | `None` (unlimited)                                  |
| `min_samples_split` | Minimum samples required to split a node      | `2`                                                 |
| `min_samples_leaf`  | Minimum samples required at a leaf node       | `1`                                                 |
| `max_features`      | Number of features to consider for best split | `None` (all features)                               |
| `criterion`         | Function to measure split quality             | `'gini'` for classification, `'mse'` for regression |

## Visualization

Our implementation provides methods to visualize the trained decision tree:

```python
clf.visualize(feature_names=['feature1', 'feature2', ...])
```

## Use Cases

- Customer churn prediction
- Loan default risk assessment
- Disease diagnosis
- Price prediction

## References

- Breiman, L., Friedman, J., Olshen, R., & Stone, C. (1984). Classification and Regression Trees.
- Quinlan, J. R. (1986). Induction of decision trees. Machine Learning, 1(1), 81-106.
