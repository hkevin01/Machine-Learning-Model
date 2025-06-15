# K-means Clustering

## Overview

K-means clustering is an unsupervised learning algorithm that partitions data into K distinct clusters based on distance to the centroid of each cluster. It's a simple yet powerful method for discovering groups and patterns in data without predefined labels.

## Principles

K-means operates through these steps:

1. **Initialization**: Randomly select K points as initial centroids
2. **Assignment**: Assign each data point to the nearest centroid, forming K clusters
3. **Update**: Recalculate centroids as the mean of all points in the cluster
4. **Iteration**: Repeat assignment and update steps until convergence or maximum iterations

## Advantages

- Simplicity and ease of implementation
- Scales well to large datasets
- Linear time complexity O(n⋅K⋅d⋅i) where n is data points, K is clusters, d is dimensions, i is iterations
- Guaranteed to converge (though potentially to a local optimum)
- Works well when clusters are spherical and similarly sized

## Limitations

- Requires specifying K in advance
- Sensitive to initial centroid placement
- Can converge to local optima
- Limited to spherical clusters
- Sensitive to outliers
- Struggles with clusters of different sizes and densities

## Implementation

Our implementation supports various initialization methods and provides both fit/predict and fit_predict interfaces:

```python
from machine_learning_model.unsupervised.kmeans import KMeans

# Basic usage
kmeans = KMeans(n_clusters=3, init='k-means++')
clusters = kmeans.fit_predict(X)

# Separate fit and predict
kmeans = KMeans(n_clusters=5, max_iter=300)
kmeans.fit(X)
clusters = kmeans.predict(X_new)

# Access cluster centers
centers = kmeans.cluster_centers_

# Inertia (sum of squared distances to nearest centroid)
inertia = kmeans.inertia_
```

## Hyperparameters

| Parameter      | Description                                           | Default       |
| -------------- | ----------------------------------------------------- | ------------- |
| `n_clusters`   | Number of clusters                                    | `8`           |
| `init`         | Initialization method: 'random', 'k-means++'          | `'k-means++'` |
| `max_iter`     | Maximum number of iterations                          | `300`         |
| `tol`          | Tolerance for declaring convergence                   | `1e-4`        |
| `n_init`       | Number of times to run with different initializations | `10`          |
| `random_state` | Random seed for reproducibility                       | `None`        |

## Finding Optimal K

To determine the optimal number of clusters, use the Elbow Method:

```python
inertias = []
for k in range(1, 11):
    kmeans = KMeans(n_clusters=k)
    kmeans.fit(X)
    inertias.append(kmeans.inertia_)

# Plot elbow curve
plt.plot(range(1, 11), inertias, marker='o')
plt.xlabel('Number of clusters')
plt.ylabel('Inertia')
plt.title('Elbow Method for Optimal K')
```

## Use Cases

- Customer segmentation
- Image compression (color quantization)
- Document clustering
- Anomaly detection
- Feature engineering through cluster assignments
- Data preprocessing for supervised learning

## References

- MacQueen, J. (1967). Some methods for classification and analysis of multivariate observations. Proceedings of the Fifth Berkeley Symposium on Mathematical Statistics and Probability.
- Arthur, D., & Vassilvitskii, S. (2007). k-means++: The advantages of careful seeding. Proceedings of the Eighteenth Annual ACM-SIAM Symposium on Discrete Algorithms.
