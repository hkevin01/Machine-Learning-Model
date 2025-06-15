# Regression Datasets

This folder contains datasets for regression tasks, organized by dataset name.

## Available Datasets

### 1. California Housing (`housing/`)
- **File**: `california_housing.csv`
- **Samples**: 20,640
- **Features**: 8 (longitude, latitude, housing_median_age, etc.)
- **Target**: median_house_value (continuous)
- **Use Case**: Price prediction, real estate analysis
- **Source**: California census data (1990)

### 2. Boston Housing (`boston/`)
- **File**: `boston_housing.csv`
- **Samples**: 506
- **Features**: 13 (crime rate, tax rate, etc.)
- **Target**: median home value (continuous)
- **Use Case**: Price prediction, urban analytics
- **Source**: Harrison & Rubinfeld (1978)

### 3. Diabetes Dataset (`diabetes/`)
- **File**: `diabetes.csv`
- **Samples**: 442
- **Features**: 10 (age, sex, BMI, blood pressure, etc.)
- **Target**: diabetes progression (continuous)
- **Use Case**: Medical prediction, health analytics
- **Source**: Scikit-learn built-in dataset

## Usage Examples

```python
import pandas as pd

# Load California Housing
housing = pd.read_csv('data/raw/regression/housing/california_housing.csv')
X = housing.drop('median_house_value', axis=1)
y = housing['median_house_value']

# Load Diabetes dataset
diabetes = pd.read_csv('data/raw/regression/diabetes/diabetes.csv')
X = diabetes.drop('target', axis=1)
y = diabetes['target']
```

## Dataset Characteristics

| Dataset | Samples | Features | Target Range | Difficulty |
|---------|---------|----------|--------------|------------|
| California Housing | 20,640 | 8 | $15k-$500k | Intermediate |
| Boston Housing | 506 | 13 | $5k-$50k | Beginner |
| Diabetes | 442 | 10 | 25-346 | Intermediate |
