# Classification Datasets

This folder contains datasets for classification tasks, organized by dataset name.

## Available Datasets

### 1. Iris Dataset (`iris/`)
- **File**: `iris.csv`
- **Samples**: 150
- **Features**: 4 (sepal_length, sepal_width, petal_length, petal_width)
- **Classes**: 3 (setosa, versicolor, virginica)
- **Use Case**: Multi-class classification, beginner-friendly
- **Source**: UCI Machine Learning Repository

### 2. Wine Dataset (`wine/`)
- **File**: `wine.csv`
- **Samples**: 178
- **Features**: 13 (chemical analysis features)
- **Classes**: 3 (wine cultivars)
- **Use Case**: Multi-class classification with chemical features
- **Source**: UCI Machine Learning Repository

### 3. Breast Cancer Wisconsin (`breast_cancer/`)
- **File**: `breast_cancer.csv`
- **Samples**: 569
- **Features**: 30 (cell nucleus measurements)
- **Classes**: 2 (malignant, benign)
- **Use Case**: Binary classification, medical diagnosis
- **Source**: UCI Machine Learning Repository

### 4. Digits Dataset (`digits/`)
- **File**: `digits.csv`
- **Samples**: 1797
- **Features**: 64 (8x8 pixel values)
- **Classes**: 10 (digits 0-9)
- **Use Case**: Multi-class classification, image recognition
- **Source**: Scikit-learn built-in dataset

## Usage Examples

```python
import pandas as pd

# Load Iris dataset
iris = pd.read_csv('data/raw/classification/iris/iris.csv')
X = iris.drop('species', axis=1)
y = iris['species']

# Load Wine dataset
wine = pd.read_csv('data/raw/classification/wine/wine.csv')
X = wine.drop('target', axis=1)
y = wine['target']
```

## Dataset Characteristics

| Dataset | Type | Difficulty | Features | Best For |
|---------|------|------------|----------|----------|
| Iris | Multi-class | Beginner | Continuous | Learning basics |
| Wine | Multi-class | Intermediate | Continuous | Feature analysis |
| Breast Cancer | Binary | Intermediate | Continuous | Medical ML |
| Digits | Multi-class | Advanced | Discrete | Image recognition |
