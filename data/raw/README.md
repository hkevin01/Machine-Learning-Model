# Raw Data Directory

This directory contains original, unmodified datasets organized by machine learning task type.

## Directory Structure

```
raw/
├── classification/          # Datasets for supervised classification
│   ├── iris/               # Iris flower dataset
│   ├── wine/               # Wine quality dataset
│   ├── breast_cancer/      # Medical diagnosis dataset
│   └── digits/             # Handwritten digit recognition
├── regression/             # Datasets for supervised regression
│   ├── housing/            # Housing price prediction
│   ├── boston/             # Boston housing prices
│   └── diabetes/           # Medical progression prediction
├── clustering/             # Datasets for unsupervised learning
│   ├── customers/          # Customer segmentation
│   ├── wholesale/          # Product category analysis
│   └── synthetic/          # Generated clustering datasets
└── text/                   # Text datasets for NLP
    ├── newsgroups/         # Text classification
    └── semi_supervised/    # Labeled/unlabeled splits
```

## Data Principles

### Raw Data Rules
1. **Immutable**: Never modify files in this directory
2. **Original Format**: Keep data in its source format
3. **Documented**: Each dataset has accompanying README
4. **Versioned**: Track data sources and download dates

### Quality Standards
- **Complete**: No missing critical information
- **Consistent**: Standardized column names and formats
- **Clean**: Remove obviously corrupted entries
- **Documented**: Clear feature descriptions and metadata

## Usage Guidelines

```python
# Always copy to processed/ before manipulation
import pandas as pd
import shutil

# Load raw data (read-only)
df = pd.read_csv('data/raw/classification/iris/iris.csv')

# Process and save to processed/
df_clean = df.dropna()
df_clean.to_csv('data/processed/iris_clean.csv', index=False)
```

## Dataset Summary

| Task | Datasets | Total Samples | Best For |
|------|----------|---------------|----------|
| Classification | 4 | 2,694 | Supervised learning |
| Regression | 3 | 21,588 | Prediction tasks |
| Clustering | 4 | 1,140 | Unsupervised learning |
| Text/NLP | 2 | 1,000 | Semi-supervised learning |

## Next Steps

After reviewing raw data:
1. Move to `../processed/` for cleaning
2. Create features in `../features/`
3. Split data in `../interim/` for training
4. Document transformations applied

---

**Last Updated**: December 15, 2024
**Total Size**: ~5MB
**Datasets**: 13 complete datasets ready for ML experiments
