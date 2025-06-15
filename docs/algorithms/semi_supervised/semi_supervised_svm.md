# Semi-Supervised Support Vector Machines (S3VM)

## Overview

Semi-Supervised Support Vector Machines (S3VM), also known as Transductive SVM, extend traditional SVMs to leverage unlabeled data in addition to labeled data. They find a decision boundary that both separates labeled data points and passes through low-density regions of the input space.

## Principles

S3VM operates through these key mechanisms:

1. **Low-Density Separation**: Decision boundary should pass through regions with few data points
2. **Transductive Learning**: Unlabeled data is directly incorporated into the training process
3. **Iterative Refinement**: Gradually improve the decision boundary using both labeled and unlabeled data
4. **Self-Training**: Bootstrap by assigning pseudo-labels to unlabeled data based on current model

## Advantages

- Makes effective use of both labeled and unlabeled data
- Inherits the strengths of traditional SVMs
- Can work well with limited labeled data
- Often outperforms supervised SVMs when labeled data is scarce
- Maintains good generalization capabilities

## Limitations

- More complex optimization problem than standard SVM
- Can converge to poor local optima
- Sensitive to initial conditions
- Computationally more intensive than supervised SVM
- Requires the "cluster assumption" to hold (classes form well-separated clusters)

## Implementation

Our implementation uses an iterative approach with gradually increasing influence of unlabeled data:

```python
from machine_learning_model.semi_supervised.semi_supervised_svm import S3VM

# Basic usage
model = S3VM(kernel='rbf', C=1.0)
# Provide labeled and unlabeled data (unlabeled marked with -1)
y_train = [0, 1, -1, -1, -1, -1, 0, 1, -1, -1]
model.fit(X_train, y_train)
predicted_labels = model.predict(X_test)

# Access assigned labels for unlabeled training data
transductive_labels = model.transduction_

# Decision function values
decision_values = model.decision_function(X_test)
```

## Hyperparameters

| Parameter   | Description                                         | Default   |
| ----------- | --------------------------------------------------- | --------- |
| `C`         | Regularization parameter for labeled data           | `1.0`     |
| `C_star`    | Regularization parameter for unlabeled data         | `0.1`     |
| `kernel`    | Kernel function: 'linear', 'poly', 'rbf', 'sigmoid' | `'rbf'`   |
| `gamma`     | Kernel coefficient for 'rbf', 'poly', and 'sigmoid' | `'scale'` |
| `degree`    | Degree of polynomial kernel                         | `3`       |
| `coef0`     | Independent term in kernel function                 | `0.0`     |
| `max_iter`  | Maximum number of iterations                        | `100`     |
| `annealing` | Whether to use annealing strategy                   | `True`    |

## Annealing Strategy

S3VM typically employs an annealing strategy to avoid poor local optima:

1. Start with a small value of `C_star` (low influence of unlabeled data)
2. Train the model and assign pseudo-labels to unlabeled data
3. Gradually increase `C_star` and retrain using the pseudo-labels
4. Continue until `C_star` reaches the target value

## Use Cases

- Text classification with limited labeled examples
- Protein sequence classification
- Image recognition with partially labeled datasets
- Spam detection
- Medical diagnosis with limited labeled cases
- Intrusion detection in network security

## References

- Vapnik, V., & Sterin, A. (1977). On structural risk minimization or overall risk in a problem of pattern recognition. Automation and Remote Control, 10(3), 1495-1503.
- Joachims, T. (1999). Transductive inference for text classification using support vector machines. ICML, 99, 200-209.
- Chapelle, O., Sindhwani, V., & Keerthi, S. S. (2008). Optimization techniques for semi-supervised support vector machines. Journal of Machine Learning Research, 9, 203-233.
