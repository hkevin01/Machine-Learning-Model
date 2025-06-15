# Label Propagation

## Overview

Label Propagation is a semi-supervised learning algorithm that assigns labels to previously unlabeled data points based on the similarity between data points. It propagates label information from labeled data points to unlabeled data points through a similarity graph.

## Principles

Label Propagation operates through these key mechanisms:

1. **Graph Construction**: Build a similarity graph where nodes are data points and edges represent similarity
2. **Label Initialization**: Assign known labels to labeled data points and initialize unlabeled points
3. **Propagation**: Iteratively update labels of unlabeled points based on neighbors' labels
4. **Clamping**: Keep the original labels of labeled data fixed during propagation
5. **Convergence**: Continue until labels stabilize or maximum iterations reached

## Advantages

- Makes effective use of both labeled and unlabeled data
- Works well with limited labeled data
- Intuitive approach based on assumption that similar points have similar labels
- Doesn't require complex optimization
- Can adapt to the manifold structure of the data

## Limitations

- Sensitive to the choice of similarity metric and graph construction
- May propagate errors if the similarity graph is poorly constructed
- Struggles with datasets that violate the manifold assumption
- Computationally intensive for large datasets
- Requires sufficient connectivity between labeled and unlabeled data

## Implementation

Our implementation supports various kernel functions and propagation methods:

```python
from machine_learning_model.semi_supervised.label_propagation import LabelPropagation

# Basic usage
model = LabelPropagation(kernel='rbf', gamma=10)
# Provide labeled and unlabeled data (unlabeled marked with -1)
y_train = [0, 1, -1, -1, -1, -1, 2, 2, -1, -1]
model.fit(X_train, y_train)
predicted_labels = model.predict(X_test)

# Access label distributions
label_distributions = model.label_distributions_

# Transduction (get labels for the unlabeled training data)
transduced_labels = model.transduction_
```

## Hyperparameters

| Parameter     | Description                        | Default |
| ------------- | ---------------------------------- | ------- |
| `kernel`      | Kernel function: 'knn', 'rbf'      | `'rbf'` |
| `gamma`       | Parameter for rbf kernel           | `20`    |
| `n_neighbors` | Number of neighbors for knn kernel | `7`     |
| `max_iter`    | Maximum number of iterations       | `1000`  |
| `tol`         | Tolerance for convergence          | `1e-3`  |
| `n_jobs`      | Number of parallel jobs            | `None`  |

## Kernel Functions

1. **RBF Kernel**:
   - Computes similarity based on radial basis function
   - Controlled by `gamma` parameter
   - Higher gamma means more local influence

2. **KNN Kernel**:
   - Computes similarity based on k nearest neighbors
   - Controlled by `n_neighbors` parameter
   - More robust to the manifold structure

## Use Cases

- Text classification with limited labeled documents
- Image recognition with partially labeled datasets
- Sentiment analysis
- Protein function prediction
- Social network analysis
- Web page classification

## References

- Zhu, X., & Ghahramani, Z. (2002). Learning from labeled and unlabeled data with label propagation. Technical Report CMU-CALD-02-107, Carnegie Mellon University.
- Bengio, Y., Delalleau, O., & Le Roux, N. (2006). Label propagation and quadratic criterion. Semi-supervised learning, 10, 193-216.
