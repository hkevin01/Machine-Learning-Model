"""Random Forest Classifier implementation with comprehensive features."""

import os
from typing import Any, Dict, Optional

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier as SklearnRandomForest
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.preprocessing import LabelEncoder


class RandomForestClassifier:
    """Enhanced Random Forest Classifier with comprehensive features."""
    
    def __init__(self, 
                 n_estimators: int = 100,
                 max_depth: Optional[int] = None,
                 min_samples_split: int = 2,
                 min_samples_leaf: int = 1,
                 criterion: str = "gini",
                 random_state: int = 42):
        """
        Initialize Random Forest Classifier.
        
        Args:
            n_estimators: Number of trees in the forest
            max_depth: Maximum depth of each tree
            min_samples_split: Minimum samples required to split
            min_samples_leaf: Minimum samples required at leaf node
            criterion: Split criterion ("gini" or "entropy")
            random_state: Random state for reproducibility
        """
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.criterion = criterion
        self.random_state = random_state
        
        # Initialize the model
        self.model = SklearnRandomForest(
            n_estimators=n_estimators,
            max_depth=max_depth,
            min_samples_split=min_samples_split,
            min_samples_leaf=min_samples_leaf,
            criterion=criterion,
            random_state=random_state
        )
        
        # Training state
        self.is_trained = False
        self.feature_names = None
        self.class_names = None
        self.label_encoder = LabelEncoder()
        
        # Performance metrics
        self.training_accuracy = None
        self.validation_accuracy = None
        self.cross_val_scores = None
        
    def fit(self, X: pd.DataFrame, y: pd.Series, validation_split: float = 0.2) -> 'RandomForestClassifier':
        """
        Fit the Random Forest model.
        
        Args:
            X: Feature matrix
            y: Target variable
            validation_split: Fraction of data to use for validation
            
        Returns:
            Self for method chaining
        """
        # Store feature names
        self.feature_names = X.columns.tolist() if hasattr(X, 'columns') else None
        
        # Encode target variable if needed
        if y.dtype == 'object':
            y_encoded = self.label_encoder.fit_transform(y)
            self.class_names = self.label_encoder.classes_
        else:
            y_encoded = y
            self.class_names = np.unique(y)
        
        # Split data for validation
        X_train, X_val, y_train, y_val = train_test_split(
            X, y_encoded, test_size=validation_split, random_state=self.random_state, stratify=y_encoded
        )
        
        # Fit the model
        self.model.fit(X_train, y_train)
        self.is_trained = True
        
        # Calculate performance metrics
        self.training_accuracy = accuracy_score(y_train, self.model.predict(X_train))
        self.validation_accuracy = accuracy_score(y_val, self.model.predict(X_val))
        
        # Cross-validation
        self.cross_val_scores = cross_val_score(
            self.model, X, y_encoded, cv=5, scoring='accuracy'
        )
        
        return self
    
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """
        Make predictions.
        
        Args:
            X: Feature matrix
            
        Returns:
            Predicted classes
        """
        if not self.is_trained:
            raise ValueError("Model must be trained before making predictions")
        
        predictions = self.model.predict(X)
        
        # Decode predictions if label encoder was used
        if hasattr(self.label_encoder, 'classes_'):
            predictions = self.label_encoder.inverse_transform(predictions)
        
        return predictions
    
    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """
        Predict class probabilities.
        
        Args:
            X: Feature matrix
            
        Returns:
            Class probabilities
        """
        if not self.is_trained:
            raise ValueError("Model must be trained before making predictions")
        
        return self.model.predict_proba(X)
    
    def get_feature_importance(self) -> Dict[str, float]:
        """
        Get feature importance scores.
        
        Returns:
            Dictionary mapping feature names to importance scores
        """
        if not self.is_trained:
            raise ValueError("Model must be trained before getting feature importance")
        
        if self.feature_names is None:
            return {f"feature_{i}": importance for i, importance in enumerate(self.model.feature_importances_)}
        
        return dict(zip(self.feature_names, self.model.feature_importances_))
    
    def evaluate(self, X: pd.DataFrame, y: pd.Series) -> Dict[str, Any]:
        """
        Evaluate model performance.
        
        Args:
            X: Feature matrix
            y: True labels
            
        Returns:
            Dictionary containing evaluation metrics
        """
        if not self.is_trained:
            raise ValueError("Model must be trained before evaluation")
        
        # Encode target if needed
        if y.dtype == 'object' and hasattr(self.label_encoder, 'classes_'):
            y_encoded = self.label_encoder.transform(y)
        else:
            y_encoded = y
        
        predictions = self.model.predict(X)
        
        # Calculate metrics
        accuracy = accuracy_score(y_encoded, predictions)
        report = classification_report(y_encoded, predictions, output_dict=True)
        conf_matrix = confusion_matrix(y_encoded, predictions)
        
        return {
            'accuracy': accuracy,
            'classification_report': report,
            'confusion_matrix': conf_matrix,
            'feature_importance': self.get_feature_importance()
        }
    
    def get_model_info(self) -> Dict[str, Any]:
        """
        Get comprehensive model information.
        
        Returns:
            Dictionary containing model information
        """
        if not self.is_trained:
            return {"status": "Model not trained"}
        
        return {
            "model_type": "Random Forest Classifier",
            "parameters": {
                "n_estimators": self.n_estimators,
                "max_depth": self.max_depth,
                "min_samples_split": self.min_samples_split,
                "min_samples_leaf": self.min_samples_leaf,
                "criterion": self.criterion,
                "random_state": self.random_state
            },
            "training_metrics": {
                "training_accuracy": self.training_accuracy,
                "validation_accuracy": self.validation_accuracy,
                "cross_val_mean": self.cross_val_scores.mean(),
                "cross_val_std": self.cross_val_scores.std()
            },
            "data_info": {
                "n_features": len(self.feature_names) if self.feature_names else None,
                "feature_names": self.feature_names,
                "n_classes": len(self.class_names),
                "class_names": self.class_names.tolist() if hasattr(self.class_names, 'tolist') else list(self.class_names)
            }
        }
    
    def save_model(self, filepath: str) -> None:
        """
        Save the trained model to disk.
        
        Args:
            filepath: Path to save the model
        """
        if not self.is_trained:
            raise ValueError("Model must be trained before saving")
        
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        
        # Save the model
        joblib.dump(self, filepath)
        print(f"Model saved to {filepath}")
    
    @classmethod
    def load_model(cls, filepath: str) -> 'RandomForestClassifier':
        """
        Load a trained model from disk.
        
        Args:
            filepath: Path to the saved model
            
        Returns:
            Loaded RandomForestClassifier instance
        """
        return joblib.load(filepath)


def create_random_forest_example() -> tuple[RandomForestClassifier, Dict[str, Any]]:
    """
    Create a complete example using the Iris dataset.
    
    Returns:
        Tuple of (trained_model, evaluation_results)
    """
    from ..data.loaders import load_iris_dataset

    # Load data
    iris_data = load_iris_dataset()
    
    # Prepare features and target
    X = iris_data.drop('species', axis=1)
    y = iris_data['species']
    
    # Create and train model
    rf_model = RandomForestClassifier(n_estimators=50, max_depth=5, random_state=42)
    rf_model.fit(X, y)
    
    # Evaluate model
    evaluation_results = rf_model.evaluate(X, y)
    
    return rf_model, evaluation_results 