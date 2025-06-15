# Principal Component Analysis (PCA)

## Overview

Principal Component Analysis (PCA) is a dimensionality reduction technique that transforms a dataset into a new coordinate system, where the new axes (principal components) maximize the variance of the data. It's widely used for feature extraction, noise filtering, and visualization of high-dimensional data.

## Principles

PCA operates through these key steps:

1. **Standardization**: Normalize the data to have zero mean and unit variance
2. **Covariance Matrix Computation**: Calculate how features vary with respect to each other
3. **Eigendecomposition**: Compute eigenvectors and eigenvalues of the covariance matrix
4. **Principal Components**: Eigenvectors represent directions of maximum variance (principal components)
5. **Dimensionality Reduction**: Project data onto the top k eigenvectors to reduce dimensions

## Advantages

- Reduces dimensionality without losing too much information
- Removes correlated features, reducing multicollinearity
- Can help visualize high-dimensional data
- Reduces computational complexity for subsequent algorithms
- Helps mitigate the curse of dimensionality
- Reduces noise in the data

## Limitations

- Only captures linear relationships between variables
- Sensitive to scaling of the original features
- May lose important information if inappropriate number of components selected
- Principal components can be difficult to interpret
- Not suitable when variance doesn't correlate with the information content

## Implementation

Our implementation provides methods for both transformation and inverse transformation:

```python
from machine_learning_model.unsupervised.pca import PCA

# Basic usage - retain 2 components
pca = PCA(n_components=2)
X_reduced = pca.fit_transform(X)

# Explained variance
explained_variance_ratio = pca.explained_variance_ratio_
cumulative_variance = explained_variance_ratio.cumsum()

# Specify variance retention instead of components
pca = PCA(n_components=0.95)  # Retain 95% of variance
X_reduced = pca.fit_transform(X)

# Inverse transform - project back to original space
X_reconstructed = pca.inverse_transform(X_reduced)

# Access components and mean
components = pca.components_
mean = pca.mean_
```

## Hyperparameters

| Parameter      | Description                                                         | Default                 |
| -------------- | ------------------------------------------------------------------- | ----------------------- |
| `n_components` | Number of components to keep, or variance retention threshold (0-1) | `None` (all components) |
| `whiten`       | Whether to whiten the data (divide by sqrt of eigenvalues)          | `False`                 |
| `svd_solver`   | SVD solver to use: 'auto', 'full', 'arpack', 'randomized'           | `'auto'`                |
| `random_state` | Random seed for randomized SVD solver                               | `None`                  |

## Selecting Number of Components

Two common approaches for selecting the number of components:

1. **Explained Variance Ratio**:
```python
# Plot explained variance
plt.plot(range(1, len(pca.explained_variance_ratio_) + 1),
         pca.explained_variance_ratio_.cumsum())
plt.xlabel('Number of Components')
plt.ylabel('Cumulative Explained Variance')
plt.title('Explained Variance vs. Components')
```

2. **Scree Plot**:
```python
# Plot eigenvalues
plt.plot(range(1, len(pca.explained_variance_) + 1),
         pca.explained_variance_)
plt.xlabel('Number of Components')
plt.ylabel('Eigenvalue (Variance)')
plt.title('Scree Plot')
```

## Use Cases

- Image compression
- Feature extraction
- Noise reduction
- Data visualization
- Speeding up machine learning algorithms
- Anomaly detection
- Face recognition (Eigenfaces)

## References

- Hotelling, H. (1933). Analysis of a complex of statistical variables into principal components. Journal of Educational Psychology, 24(6), 417–441.
- Jolliffe, I. T. (2002). Principal Component Analysis. Springer-Verlag.
