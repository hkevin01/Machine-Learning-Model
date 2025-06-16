# Unsupervised Learning Algorithms

This directory contains documentation for unsupervised learning algorithms - those that find patterns in data without explicit labels.

## Available Algorithms

- [**K-means Clustering**](kmeans.md) - Partitions data into k clusters based on feature similarity
  - Centroid-based clustering
  - Simple and efficient implementation
  - Works best with spherical clusters

- [**DBSCAN**](dbscan.md) - Density-Based Spatial Clustering of Applications with Noise
  - Discovers clusters of arbitrary shape
  - Robust to outliers
  - No need to specify number of clusters

- [**Principal Component Analysis (PCA)**](pca.md) - Dimensionality reduction technique
  - Reduces feature space while preserving variance
  - Useful for visualization and preprocessing
  - Handles correlated features

## Common Interface

All unsupervised learning algorithms implement this common interface:

```python
# For clustering algorithms
model.fit(X)
clusters = model.predict(X)
# or
clusters = model.fit_predict(X)

# For dimensionality reduction
model.fit(X)
X_transformed = model.transform(X)
# or
X_transformed = model.fit_transform(X)
```

## When to Use

- **K-means**: When you know the number of clusters and expect them to be roughly spherical
- **DBSCAN**: When you don't know the number of clusters and clusters may have irregular shapes
- **PCA**: When you need to reduce dimensionality, visualize high-dimensional data, or remove correlations

## References

See individual algorithm pages for specific references and academic papers.
