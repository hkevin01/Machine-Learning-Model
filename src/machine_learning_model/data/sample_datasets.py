"""
Sample dataset utilities for demonstrations and examples.
"""

import os
from typing import Any, Dict, Tuple

import numpy as np
import pandas as pd
from sklearn.datasets import (
    load_breast_cancer,
    load_diabetes,
    load_iris,
    load_wine,
    make_blobs,
    make_classification,
    make_regression,
)


class SampleDatasets:
    """Manage sample datasets for algorithm demonstrations."""
    
    @staticmethod
    def get_iris_data() -> Tuple[np.ndarray, np.ndarray, Dict[str, Any]]:
        """Load Iris dataset for classification."""
        iris = load_iris()
        return iris.data, iris.target, {
            'feature_names': iris.feature_names,
            'target_names': iris.target_names,
            'description': iris.DESCR,
            'task_type': 'classification'
        }
    
    @staticmethod
    def get_diabetes_data() -> Tuple[np.ndarray, np.ndarray, Dict[str, Any]]:
        """Load Diabetes dataset for regression."""
        diabetes = load_diabetes()
        return diabetes.data, diabetes.target, {
            'feature_names': diabetes.feature_names,
            'description': diabetes.DESCR,
            'task_type': 'regression'
        }
    
    @staticmethod
    def get_breast_cancer_data() -> Tuple[np.ndarray, np.ndarray, Dict[str, Any]]:
        """Load Breast Cancer dataset for classification."""
        cancer = load_breast_cancer()
        return cancer.data, cancer.target, {
            'feature_names': cancer.feature_names,
            'target_names': cancer.target_names,
            'description': cancer.DESCR,
            'task_type': 'classification'
        }
    
    @staticmethod
    def get_wine_data() -> Tuple[np.ndarray, np.ndarray, Dict[str, Any]]:
        """Load Wine dataset for classification."""
        wine = load_wine()
        return wine.data, wine.target, {
            'feature_names': wine.feature_names,
            'target_names': wine.target_names,
            'description': wine.DESCR,
            'task_type': 'classification'
        }
    
    @staticmethod
    def create_synthetic_classification(n_samples: int = 1000, n_features: int = 20, 
                                      n_classes: int = 3) -> Tuple[np.ndarray, np.ndarray, Dict[str, Any]]:
        """Create synthetic classification dataset."""
        X, y = make_classification(
            n_samples=n_samples,
            n_features=n_features,
            n_classes=n_classes,
            n_informative=n_features//2,
            n_redundant=n_features//4,
            random_state=42
        )
        return X, y, {
            'feature_names': [f'feature_{i}' for i in range(n_features)],
            'target_names': [f'class_{i}' for i in range(n_classes)],
            'description': f'Synthetic classification dataset with {n_samples} samples and {n_features} features',
            'task_type': 'classification'
        }
    
    @staticmethod
    def create_synthetic_regression(n_samples: int = 1000, n_features: int = 10, 
                                  noise: float = 0.1) -> Tuple[np.ndarray, np.ndarray, Dict[str, Any]]:
        """Create synthetic regression dataset."""
        X, y = make_regression(
            n_samples=n_samples,
            n_features=n_features,
            noise=noise,
            random_state=42
        )
        return X, y, {
            'feature_names': [f'feature_{i}' for i in range(n_features)],
            'description': f'Synthetic regression dataset with {n_samples} samples and {n_features} features',
            'task_type': 'regression'
        }
    
    @staticmethod
    def create_clustering_data(n_samples: int = 300, centers: int = 4) -> Tuple[np.ndarray, np.ndarray, Dict[str, Any]]:
        """Create synthetic clustering dataset."""
        X, y = make_blobs(
            n_samples=n_samples,
            centers=centers,
            n_features=2,
            random_state=42,
            cluster_std=1.5
        )
        return X, y, {
            'feature_names': ['feature_0', 'feature_1'],
            'target_names': [f'cluster_{i}' for i in range(centers)],
            'description': f'Synthetic clustering dataset with {n_samples} samples and {centers} clusters',
            'task_type': 'clustering'
        }
    
    @staticmethod
    def get_dataset_for_algorithm(algorithm_name: str) -> Tuple[np.ndarray, np.ndarray, Dict[str, Any]]:
        """Get appropriate dataset for specific algorithm."""
        if algorithm_name in ["Decision Trees", "Random Forest", "Logistic Regression"]:
            return SampleDatasets.get_iris_data()
        elif algorithm_name == "Linear Regression":
            return SampleDatasets.get_diabetes_data()
        elif algorithm_name in ["K-Means Clustering", "DBSCAN"]:
            return SampleDatasets.create_clustering_data()
        elif algorithm_name == "Principal Component Analysis":
            return SampleDatasets.get_breast_cancer_data()
        else:
            # Default to iris for classification algorithms
            return SampleDatasets.get_iris_data()
    
    @staticmethod
    def save_dataset(X: np.ndarray, y: np.ndarray, name: str, 
                    feature_names: list = None, target_names: list = None):
        """Save dataset to CSV files."""
        os.makedirs('data/processed', exist_ok=True)
        
        # Save features
        if feature_names:
            df_X = pd.DataFrame(X, columns=feature_names)
        else:
            df_X = pd.DataFrame(X, columns=[f'feature_{i}' for i in range(X.shape[1])])
        
        df_X.to_csv(f'data/processed/{name}_features.csv', index=False)
        
        # Save targets
        df_y = pd.DataFrame(y, columns=['target'])
        df_y.to_csv(f'data/processed/{name}_targets.csv', index=False)
        
        print(f"Dataset '{name}' saved to data/processed/")
    
    @staticmethod
    def load_dataset(name: str) -> Tuple[np.ndarray, np.ndarray]:
        """Load dataset from CSV files."""
        try:
            df_X = pd.read_csv(f'data/processed/{name}_features.csv')
            df_y = pd.read_csv(f'data/processed/{name}_targets.csv')
            return df_X.values, df_y.values.ravel()
        except FileNotFoundError:
            raise FileNotFoundError(f"Dataset '{name}' not found in data/processed/")

if __name__ == "__main__":
    # Create and save sample datasets
    datasets = SampleDatasets()
    
    print("Creating sample datasets...")
    
    # Save all standard datasets
    X, y, info = datasets.get_iris_data()
    datasets.save_dataset(X, y, 'iris', info['feature_names'], info['target_names'])
    
    X, y, info = datasets.get_diabetes_data()
    datasets.save_dataset(X, y, 'diabetes', info['feature_names'])
    
    X, y, info = datasets.create_synthetic_classification()
    datasets.save_dataset(X, y, 'synthetic_classification', info['feature_names'], info['target_names'])
    
    X, y, info = datasets.create_synthetic_regression()
    datasets.save_dataset(X, y, 'synthetic_regression', info['feature_names'])
    
    X, y, info = datasets.create_clustering_data()
    datasets.save_dataset(X, y, 'clustering_demo', info['feature_names'], info['target_names'])
    
    print("✅ Sample datasets created and saved to data/processed/")
