"""
Tests for Decision Tree implementation.
"""

import numpy as np
import pandas as pd
import pytest
from sklearn.datasets import make_classification, make_regression
from sklearn.metrics import accuracy_score, mean_squared_error
from sklearn.model_selection import train_test_split

from machine_learning_model.supervised.decision_tree import (
    DecisionTreeClassifier,
    DecisionTreeRegressor,
)


class TestDecisionTreeClassifier:
    """Test suite for Decision Tree Classifier."""
    
    def test_basic_classification(self):
        """Test basic classification functionality."""
        # Create simple dataset
        X, y = make_classification(n_samples=100, n_features=4, n_classes=2, random_state=42)
        
        # Train classifier
        clf = DecisionTreeClassifier(random_state=42)
        clf.fit(X, y)
        
        # Make predictions
        predictions = clf.predict(X)
        
        # Check basic properties
        assert len(predictions) == len(y)
        assert set(predictions).issubset(set(y))
        assert accuracy_score(y, predictions) > 0.8  # Should achieve reasonable accuracy
    
    def test_feature_importances(self):
        """Test feature importance calculation."""
        X, y = make_classification(n_samples=100, n_features=4, random_state=42)
        
        clf = DecisionTreeClassifier(random_state=42)
        clf.fit(X, y)
        
        # Check feature importances
        assert clf.feature_importances_ is not None
        assert len(clf.feature_importances_) == X.shape[1]
        assert np.isclose(np.sum(clf.feature_importances_), 1.0, atol=1e-7)
    
    def test_pandas_input(self):
        """Test with pandas DataFrame input."""
        X, y = make_classification(n_samples=50, n_features=3, random_state=42)
        X_df = pd.DataFrame(X, columns=['feature1', 'feature2', 'feature3'])
        y_series = pd.Series(y)
        
        clf = DecisionTreeClassifier(random_state=42)
        clf.fit(X_df, y_series)
        
        predictions = clf.predict(X_df)
        assert len(predictions) == len(y)


class TestDecisionTreeRegressor:
    """Test suite for Decision Tree Regressor."""
    
    def test_basic_regression(self):
        """Test basic regression functionality."""
        # Create simple dataset
        X, y = make_regression(n_samples=100, n_features=4, noise=0.1, random_state=42)
        
        # Train regressor
        reg = DecisionTreeRegressor(random_state=42)
        reg.fit(X, y)
        
        # Make predictions
        predictions = reg.predict(X)
        
        # Check basic properties
        assert len(predictions) == len(y)
        assert mean_squared_error(y, predictions) < 100  # Should achieve reasonable MSE
    
    def test_tree_structure(self):
        """Test tree structure extraction."""
        X, y = make_regression(n_samples=50, n_features=2, random_state=42)
        
        reg = DecisionTreeRegressor(max_depth=3, random_state=42)
        reg.fit(X, y)
        
        tree_structure = reg.get_tree_structure()
        assert isinstance(tree_structure, dict)
        assert 'type' in tree_structure
    
    def test_hyperparameters(self):
        """Test different hyperparameter settings."""
        X, y = make_regression(n_samples=100, n_features=3, random_state=42)
        
        # Test different max_depth values
        for max_depth in [1, 3, 5]:
            reg = DecisionTreeRegressor(max_depth=max_depth, random_state=42)
            reg.fit(X, y)
            predictions = reg.predict(X)
            assert len(predictions) == len(y)


def test_node_class():
    """Test Node class functionality."""
    from machine_learning_model.supervised.decision_tree import Node

    # Test leaf node
    leaf = Node(value=1.0)
    assert leaf.is_leaf()
    assert leaf.value == 1.0
    
    # Test split node
    split_node = Node(feature=0, threshold=0.5, left=leaf, right=leaf)
    assert not split_node.is_leaf()
    assert split_node.feature == 0
    assert split_node.threshold == 0.5
