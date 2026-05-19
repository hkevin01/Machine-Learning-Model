# Clustering Datasets

This folder contains datasets for unsupervised learning and clustering tasks.

## Available Datasets

### 1. Mall Customers (`customers/`)
- **File**: `mall_customers.csv`
- **Samples**: 200
- **Features**: 4 (Gender, Age, Annual_Income, Spending_Score)
- **Use Case**: Customer segmentation, market analysis
- **Clustering Goal**: Identify customer groups for targeted marketing

### 2. Wholesale Customers (`wholesale/`)
- **File**: `wholesale_customers.csv`
- **Samples**: 440
- **Features**: 8 (Fresh, Milk, Grocery, Frozen, Detergents_Paper, Delicassen, Channel, Region)
- **Use Case**: Product category analysis, supply chain optimization
- **Source**: UCI Machine Learning Repository

### 3. Synthetic Datasets (`synthetic/`)
- **Files**: `blobs.csv`, `circles.csv`, `moons.csv`
- **Samples**: 300 each
- **Features**: 2 (for visualization)
- **Use Case**: Algorithm testing, visualization, educational purposes

## Usage Examples

```python
import pandas as pd
import matplotlib.pyplot as plt

# Load Mall Customers
customers = pd.read_csv('data/raw/clustering/customers/mall_customers.csv')
X = customers[['Annual_Income', 'Spending_Score']]

# Load Synthetic Blobs
blobs = pd.read_csv('data/raw/clustering/synthetic/blobs.csv')
plt.scatter(blobs['x'], blobs['y'])
```

## Clustering Algorithms to Try

| <sub>Algorithm</sub> | <sub>Best For</sub> | <sub>Parameters</sub> |
|-----------|----------|------------|
| <sub>K-Means</sub> | <sub>Spherical clusters</sub> | <sub>n_clusters</sub> |
| <sub>DBSCAN</sub> | <sub>Arbitrary shapes</sub> | <sub>eps, min_samples</sub> |
| <sub>Agglomerative</sub> | <sub>Hierarchical</sub> | <sub>n_clusters, linkage</sub> |
| <sub>Gaussian Mixture</sub> | <sub>Overlapping clusters</sub> | <sub>n_components</sub> |