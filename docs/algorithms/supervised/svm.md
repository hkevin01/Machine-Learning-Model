# Support Vector Machines (SVM)

## Overview

Support Vector Machines (SVMs) are powerful supervised learning models used for classification, regression, and outlier detection. SVMs find the optimal hyperplane that maximizes the margin between different classes in the feature space.

## Principles

SVM operates on these key principles:

1. **Maximum Margin Hyperplane**: Finds the hyperplane with the maximum margin between classes
2. **Support Vectors**: Data points closest to the hyperplane that influence its position
3. **Kernel Trick**: Transforms input data into higher-dimensional space to make it separable
4. **Soft Margin**: Allows for some misclassification to achieve better generalization

## Advantages

- Effective in high-dimensional spaces
- Memory efficient as it uses only a subset of training points (support vectors)
- Versatile through different kernel functions
- Robust against overfitting in high-dimensional spaces
- Effective when the number of features is greater than the number of samples

## Limitations

- Doesn't directly provide probability estimates
- Poor performance with overlapping classes
- Sensitive to the choice of kernel and regularization parameters
- Computationally intensive for large datasets
- Requires careful feature scaling

## Implementation

Our implementation supports various kernels and both classification and regression tasks.

```python
from machine_learning_model.supervised.svm import SVMClassifier, SVMRegressor

# Classification example
clf = SVMClassifier(kernel='rbf', C=1.0)
clf.fit(X_train, y_train)
predictions = clf.predict(X_test)
accuracy = clf.evaluate(X_test, y_test)

# Regression example
regressor = SVMRegressor(kernel='rbf', C=1.0, epsilon=0.1)
regressor.fit(X_train, y_train)
predictions = regressor.predict(X_test)
mse = regressor.evaluate(X_test, y_test)
```

## Hyperparameters

| Parameter | Description                                         | Default   |
| --------- | --------------------------------------------------- | --------- |
| `C`       | Regularization parameter (inverse of strength)      | `1.0`     |
| `kernel`  | Kernel type: 'linear', 'poly', 'rbf', 'sigmoid'     | `'rbf'`   |
| `degree`  | Degree of polynomial kernel                         | `3`       |
| `gamma`   | Kernel coefficient for 'rbf', 'poly', and 'sigmoid' | `'scale'` |
| `coef0`   | Independent term in kernel function                 | `0.0`     |
| `epsilon` | Epsilon in the epsilon-SVR model (regression only)  | `0.1`     |

## Kernel Functions

SVMs support various kernel functions for different types of data:

1. **Linear**: `K(x, y) = x^T y`
   - Best for linearly separable data

2. **Polynomial**: `K(x, y) = (gamma * x^T y + coef0)^degree`
   - Good for nonlinear boundaries with specific structure

3. **RBF (Gaussian)**: `K(x, y) = exp(-gamma * ||x - y||^2)`
   - Versatile, works well for most problems

4. **Sigmoid**: `K(x, y) = tanh(gamma * x^T y + coef0)`
   - Similar to neural networks

## Use Cases

- Text classification
- Image classification
- Gene expression classification
- Face detection
- Handwriting recognition
- Anomaly detection

## References

- Cortes, C., & Vapnik, V. (1995). Support-Vector Networks. Machine Learning, 20(3), 273-297.
- Schölkopf, B., & Smola, A. J. (2002). Learning with Kernels: Support Vector Machines, Regularization, Optimization, and Beyond.
