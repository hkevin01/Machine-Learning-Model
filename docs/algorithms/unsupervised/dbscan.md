# DBSCAN (Density-Based Spatial Clustering of Applications with Noise)

## Overview

DBSCAN is a density-based clustering algorithm that groups together points that are closely packed, while marking points in low-density regions as outliers. It can discover clusters of arbitrary shape without requiring the number of clusters to be specified beforehand.

## Principles

DBSCAN operates based on these key concepts:

1. **Core Points**: Points with at least `min_samples` points within distance `eps`
2. **Border Points**: Points within distance `eps` of a core point but not core points themselves
3. **Noise Points**: Points that are neither core nor border points
4. **Directly Density-Reachable**: A point q is directly density-reachable from p if p is a core point and q is within distance `eps` of p
5. **Density-Connected**: Two points are density-connected if they are both density-reachable from a common point

## Advantages

- Does not require specifying the number of clusters
- Can find arbitrarily shaped clusters
- Robust to outliers
- Only needs two parameters: `eps` and `min_samples`
- Does not assume or impose cluster shapes

## Limitations

- Struggles with varying densities across clusters
- Sensitive to parameter choices
- Struggles with high-dimensional data due to the "curse of dimensionality"
- Less effective when clusters are close to each other
- Computationally more intensive than K-means (O(n²) in the worst case)

## Implementation

Our implementation supports various distance metrics and provides methods for finding optimal parameters:

```python
from machine_learning_model.unsupervised.dbscan import DBSCAN

# Basic usage
dbscan = DBSCAN(eps=0.5, min_samples=5)
clusters = dbscan.fit_predict(X)

# Using different distance metric
dbscan = DBSCAN(eps=0.5, min_samples=5, metric='cosine')
clusters = dbscan.fit_predict(X)

# Access core points and labels
core_mask = dbscan.core_sample_indices_
labels = dbscan.labels_  # -1 indicates noise points
```

## Hyperparameters

| Parameter     | Description                                                                     | Default       |
| ------------- | ------------------------------------------------------------------------------- | ------------- |
| `eps`         | Maximum distance between two samples to be considered neighbors                 | Required      |
| `min_samples` | Minimum samples in a neighborhood for a point to be a core point                | `5`           |
| `metric`      | Distance metric: 'euclidean', 'manhattan', 'cosine', etc.                       | `'euclidean'` |
| `algorithm`   | Algorithm to compute nearest neighbors: 'auto', 'ball_tree', 'kd_tree', 'brute' | `'auto'`      |
| `leaf_size`   | Leaf size for BallTree or KDTree                                                | `30`          |
| `p`           | Power parameter for Minkowski metric                                            | `2`           |

## Finding Optimal Parameters

To find optimal `eps` and `min_samples` values:

```python
from sklearn.neighbors import NearestNeighbors
import numpy as np
import matplotlib.pyplot as plt

# K-distance graph for finding eps
k = 5  # min_samples - 1
nn = NearestNeighbors(n_neighbors=k)
nn.fit(X)
distances, _ = nn.kneighbors(X)
distances = np.sort(distances[:, k-1])

# Plot k-distance graph
plt.plot(distances)
plt.xlabel('Points')
plt.ylabel(f'{k}-th nearest neighbor distance')
plt.title('K-distance Graph')
```

## Use Cases

- Spatial data analysis
- Anomaly detection
- Image segmentation
- Network analysis
- Traffic pattern analysis
- Customer behavior clustering
- Noise detection in datasets

## References

- Ester, M., Kriegel, H. P., Sander, J., & Xu, X. (1996). A density-based algorithm for discovering clusters in large spatial databases with noise. KDD, 96(34), 226-231.
- Schubert, E., Sander, J., Ester, M., Kriegel, H. P., & Xu, X. (2017). DBSCAN revisited, revisited: why and how you should (still) use DBSCAN. ACM Transactions on Database Systems (TODS), 42(3), 1-21.
