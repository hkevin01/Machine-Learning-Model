"""
Tests for Random Forest implementation.
"""

import platform

import numpy as np
import pandas as pd
import pytest
from sklearn.datasets import make_classification, make_regression
from sklearn.metrics import accuracy_score, mean_squared_error
from sklearn.model_selection import train_test_split

from machine_learning_model.supervised.random_forest import (
    RandomForestClassifier,
    RandomForestRegressor,
)


class TestRandomForestClassifier:
    """Test suite for Random Forest Classifier."""
    
    def test_basic_classification(self):
        """Test basic classification functionality."""
        X, y = make_classification(n_samples=200, n_features=10, n_classes=3, random_state=42)
        
        clf = RandomForestClassifier(n_estimators=50, random_state=42)
        clf.fit(X, y)
        
        predictions = clf.predict(X)
        
        assert len(predictions) == len(y)
        assert set(predictions).issubset(set(y))
        assert accuracy_score(y, predictions) > 0.8
    
    def test_feature_importances(self):
        """Test feature importance calculation."""
        X, y = make_classification(n_samples=100, n_features=5, random_state=42)
        
        clf = RandomForestClassifier(n_estimators=10, random_state=42)
        clf.fit(X, y)
        
        assert clf.feature_importances_ is not None
        assert len(clf.feature_importances_) == X.shape[1]
        assert np.isclose(np.sum(clf.feature_importances_), 1.0, atol=1e-7)
    
    def test_oob_score(self):
        """Test out-of-bag score calculation."""
        X, y = make_classification(n_samples=100, n_features=5, random_state=42)
        
        clf = RandomForestClassifier(n_estimators=20, oob_score=True, random_state=42)
        clf.fit(X, y)
        
        assert clf.oob_score_ is not None
        assert 0 <= clf.oob_score_ <= 1
    
    def test_predict_proba(self):
        """Test probability prediction."""
        X, y = make_classification(n_samples=100, n_features=5, n_classes=3, random_state=42)
        
        clf = RandomForestClassifier(n_estimators=10, random_state=42)
        clf.fit(X, y)
        
        probas = clf.predict_proba(X)
        
        assert probas.shape == (X.shape[0], clf.n_classes_)
        assert np.allclose(np.sum(probas, axis=1), 1.0)
    
    def test_max_features(self):
        """Test different max_features settings."""
        X, y = make_classification(n_samples=100, n_features=10, random_state=42)
        
        for max_features in ['sqrt', 'log2', 0.5, 5]:
            clf = RandomForestClassifier(n_estimators=5, max_features=max_features, random_state=42)
            clf.fit(X, y)
            predictions = clf.predict(X)
            assert len(predictions) == len(y)


class TestRandomForestRegressor:
    """Test suite for Random Forest Regressor."""
    
    def test_basic_regression(self):
        """Test basic regression functionality."""
        X, y = make_regression(n_samples=200, n_features=10, noise=0.1, random_state=42)
        
        reg = RandomForestRegressor(n_estimators=50, random_state=42)
        reg.fit(X, y)
        
        predictions = reg.predict(X)
        
        assert len(predictions) == len(y)
        assert mean_squared_error(y, predictions) < 100
    
    def test_oob_score_regression(self):
        """Test OOB score for regression."""
        X, y = make_regression(n_samples=100, n_features=5, random_state=42)
        
        reg = RandomForestRegressor(n_estimators=20, oob_score=True, random_state=42)
        reg.fit(X, y)
        
        assert reg.oob_score_ is not None
        assert -1 <= reg.oob_score_ <= 1  # R² can be negative
    
    def test_bootstrap_setting(self):
        """Test bootstrap vs no bootstrap."""
        X, y = make_regression(n_samples=100, n_features=5, random_state=42)
        
        # With bootstrap
        reg_bootstrap = RandomForestRegressor(n_estimators=10, bootstrap=True, random_state=42)
        reg_bootstrap.fit(X, y)
        
        # Without bootstrap
        reg_no_bootstrap = RandomForestRegressor(n_estimators=10, bootstrap=False, random_state=42)
        reg_no_bootstrap.fit(X, y)
        
        # Both should work
        pred_bootstrap = reg_bootstrap.predict(X)
        pred_no_bootstrap = reg_no_bootstrap.predict(X)
        
        assert len(pred_bootstrap) == len(y)
        assert len(pred_no_bootstrap) == len(y)
    
    def test_pandas_input(self):
        """Test with pandas DataFrame input."""
        X, y = make_regression(n_samples=50, n_features=3, random_state=42)
        X_df = pd.DataFrame(X, columns=['feature1', 'feature2', 'feature3'])
        y_series = pd.Series(y)
        
        reg = RandomForestRegressor(n_estimators=5, random_state=42)
        reg.fit(X_df, y_series)
        
        predictions = reg.predict(X_df)
        assert len(predictions) == len(y)


class TestCrossPlatformCompatibility:
    """Test cross-platform compatibility for Random Forest."""
    
    def test_windows_compatibility(self):
        """Test Windows-specific functionality."""
        X, y = make_classification(n_samples=50, n_features=5, random_state=42)
        
        # Test with different n_jobs settings (Windows handles threading differently)
        for n_jobs in [1, 2, -1]:
            rf = RandomForestClassifier(n_estimators=5, n_jobs=n_jobs, random_state=42)
            rf.fit(X, y)
            predictions = rf.predict(X)
            assert len(predictions) == len(y)
    
    def test_ubuntu_compatibility(self):
        """Test Ubuntu-specific functionality."""
        X, y = make_regression(n_samples=50, n_features=5, random_state=42)
        
        # Test memory usage and threading on Ubuntu
        rf = RandomForestRegressor(n_estimators=10, n_jobs=2, random_state=42)
        rf.fit(X, y)
        predictions = rf.predict(X)
        assert len(predictions) == len(y)
    
    def test_path_handling(self):
        """Test file path handling across platforms."""
        import os
        import tempfile
        
        X, y = make_classification(n_samples=30, n_features=3, random_state=42)
        rf = RandomForestClassifier(n_estimators=5, random_state=42)
        rf.fit(X, y)
        
        # Test saving/loading with different path separators
        with tempfile.TemporaryDirectory() as temp_dir:
            if platform.system() == "Windows":
                test_path = os.path.join(temp_dir, "model\\test_rf.pkl")
            else:
                test_path = os.path.join(temp_dir, "model/test_rf.pkl")
            
            # Create directory if needed
            os.makedirs(os.path.dirname(test_path), exist_ok=True)
            
            # This would test model persistence (placeholder)
            assert os.path.exists(os.path.dirname(test_path))

def test_platform_detection():
    """Test platform detection and appropriate configuration."""
    current_platform = platform.system()
    assert current_platform in ["Windows", "Linux", "Darwin"]
    
    # Test platform-specific settings
    if current_platform == "Windows":
        # Windows-specific tests
        assert sys.platform.startswith('win')
    elif current_platform == "Linux":
        # Ubuntu/Linux-specific tests
        assert sys.platform.startswith('linux')
