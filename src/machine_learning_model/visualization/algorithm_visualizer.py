"""
Algorithm visualization utilities for interactive demonstrations.
"""

import warnings
from typing import Any, Dict, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.datasets import (
    load_diabetes,
    load_iris,
    make_classification,
    make_regression,
)
from sklearn.metrics import accuracy_score, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings('ignore')

class AlgorithmVisualizer:
    """Visualize algorithm performance and decision boundaries."""
    
    def __init__(self):
        self.fig = None
        self.axes = None
        
    def visualize_decision_tree(self, save_path: str = None) -> Dict[str, Any]:
        """Visualize Decision Tree with demo data."""
        from machine_learning_model.supervised.decision_tree import (
            DecisionTreeClassifier,
        )

        # Load demo data
        iris = load_iris()
        X, y = iris.data[:, :2], iris.target  # Use first 2 features for visualization
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
        
        # Train model
        clf = DecisionTreeClassifier(max_depth=5, random_state=42)
        clf.fit(X_train, y_train)
        
        # Make predictions
        y_pred = clf.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        
        # Create visualization
        self.fig, self.axes = plt.subplots(2, 2, figsize=(15, 12))
        self.fig.suptitle('Decision Tree Visualization - Iris Dataset', fontsize=16)
        
        # Decision boundary
        self._plot_decision_boundary(clf, X, y, self.axes[0, 0], 
                                   title='Decision Boundary', 
                                   feature_names=iris.feature_names[:2])
        
        # Feature importance
        feature_importance = clf.feature_importances_
        self.axes[0, 1].bar(range(len(feature_importance)), feature_importance)
        self.axes[0, 1].set_title('Feature Importance')
        self.axes[0, 1].set_xlabel('Features')
        self.axes[0, 1].set_ylabel('Importance')
        self.axes[0, 1].set_xticks(range(len(iris.feature_names[:2])))
        self.axes[0, 1].set_xticklabels(iris.feature_names[:2], rotation=45)
        
        # Predictions vs Actual
        self.axes[1, 0].scatter(y_test, y_pred, alpha=0.7)
        self.axes[1, 0].plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--')
        self.axes[1, 0].set_xlabel('Actual')
        self.axes[1, 0].set_ylabel('Predicted')
        self.axes[1, 0].set_title(f'Predictions vs Actual (Accuracy: {accuracy:.3f})')
        
        # Tree structure info
        tree_info = f"""
Tree Statistics:
• Max Depth: {clf.max_depth}
• Training Samples: {len(X_train)}
• Test Samples: {len(X_test)}
• Accuracy: {accuracy:.3f}
• Classes: {len(iris.target_names)}
        """
        self.axes[1, 1].text(0.1, 0.5, tree_info, fontsize=12, verticalalignment='center')
        self.axes[1, 1].axis('off')
        self.axes[1, 1].set_title('Model Information')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        plt.show()
        
        return {
            'accuracy': accuracy,
            'feature_importance': feature_importance,
            'model': clf,
            'test_data': (X_test, y_test, y_pred)
        }
    
    def visualize_random_forest(self, save_path: str = None) -> Dict[str, Any]:
        """Visualize Random Forest with demo data."""
        from machine_learning_model.supervised.random_forest import (
            RandomForestClassifier,
        )

        # Load demo data
        iris = load_iris()
        X, y = iris.data, iris.target
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
        
        # Train model
        rf = RandomForestClassifier(n_estimators=50, max_depth=5, random_state=42, oob_score=True)
        rf.fit(X_train, y_train)
        
        # Make predictions
        y_pred = rf.predict(X_test)
        y_prob = rf.predict_proba(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        
        # Create visualization
        self.fig, self.axes = plt.subplots(2, 2, figsize=(15, 12))
        self.fig.suptitle('Random Forest Visualization - Iris Dataset', fontsize=16)
        
        # Decision boundary (using first 2 features)
        self._plot_decision_boundary(rf, X[:, :2], y, self.axes[0, 0],
                                   title='Decision Boundary (2D projection)',
                                   feature_names=iris.feature_names[:2])
        
        # Feature importance
        feature_importance = rf.feature_importances_
        self.axes[0, 1].bar(range(len(feature_importance)), feature_importance)
        self.axes[0, 1].set_title('Feature Importance')
        self.axes[0, 1].set_xlabel('Features')
        self.axes[0, 1].set_ylabel('Importance')
        self.axes[0, 1].set_xticks(range(len(iris.feature_names)))
        self.axes[0, 1].set_xticklabels(iris.feature_names, rotation=45)
        
        # Prediction probabilities
        prob_df = pd.DataFrame(y_prob, columns=iris.target_names)
        prob_df.plot(kind='box', ax=self.axes[1, 0])
        self.axes[1, 0].set_title('Prediction Probability Distribution')
        self.axes[1, 0].set_ylabel('Probability')
        
        # Model statistics
        model_info = f"""
Random Forest Statistics:
• Number of Trees: {rf.n_estimators}
• Max Depth: {rf.max_depth}
• Training Samples: {len(X_train)}
• Test Samples: {len(X_test)}
• Accuracy: {accuracy:.3f}
• OOB Score: {rf.oob_score_:.3f}
• Classes: {len(iris.target_names)}
        """
        self.axes[1, 1].text(0.1, 0.5, model_info, fontsize=12, verticalalignment='center')
        self.axes[1, 1].axis('off')
        self.axes[1, 1].set_title('Model Information')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        plt.show()
        
        return {
            'accuracy': accuracy,
            'oob_score': rf.oob_score_,
            'feature_importance': feature_importance,
            'model': rf,
            'test_data': (X_test, y_test, y_pred, y_prob)
        }
    
    def visualize_linear_regression(self, save_path: str = None) -> Dict[str, Any]:
        """Visualize Linear Regression with demo data."""
        from sklearn.linear_model import LinearRegression

        # Load demo data
        diabetes = load_diabetes()
        X, y = diabetes.data, diabetes.target
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
        
        # Train model
        lr = LinearRegression()
        lr.fit(X_train, y_train)
        
        # Make predictions
        y_pred = lr.predict(X_test)
        mse = mean_squared_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)
        
        # Create visualization
        self.fig, self.axes = plt.subplots(2, 2, figsize=(15, 12))
        self.fig.suptitle('Linear Regression Visualization - Diabetes Dataset', fontsize=16)
        
        # Predictions vs Actual
        self.axes[0, 0].scatter(y_test, y_pred, alpha=0.6)
        self.axes[0, 0].plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
        self.axes[0, 0].set_xlabel('Actual Values')
        self.axes[0, 0].set_ylabel('Predicted Values')
        self.axes[0, 0].set_title(f'Predictions vs Actual (R² = {r2:.3f})')
        self.axes[0, 0].grid(True, alpha=0.3)
        
        # Residuals plot
        residuals = y_test - y_pred
        self.axes[0, 1].scatter(y_pred, residuals, alpha=0.6)
        self.axes[0, 1].axhline(y=0, color='r', linestyle='--')
        self.axes[0, 1].set_xlabel('Predicted Values')
        self.axes[0, 1].set_ylabel('Residuals')
        self.axes[0, 1].set_title('Residuals Plot')
        self.axes[0, 1].grid(True, alpha=0.3)
        
        # Feature coefficients
        coefficients = lr.coef_
        feature_names = diabetes.feature_names
        self.axes[1, 0].barh(range(len(coefficients)), coefficients)
        self.axes[1, 0].set_yticks(range(len(feature_names)))
        self.axes[1, 0].set_yticklabels(feature_names)
        self.axes[1, 0].set_xlabel('Coefficient Value')
        self.axes[1, 0].set_title('Feature Coefficients')
        self.axes[1, 0].grid(True, alpha=0.3)
        
        # Model statistics
        model_info = f"""
Linear Regression Statistics:
• Training Samples: {len(X_train)}
• Test Samples: {len(X_test)}
• Features: {X.shape[1]}
• MSE: {mse:.2f}
• R² Score: {r2:.3f}
• Intercept: {lr.intercept_:.2f}
        """
        self.axes[1, 1].text(0.1, 0.5, model_info, fontsize=12, verticalalignment='center')
        self.axes[1, 1].axis('off')
        self.axes[1, 1].set_title('Model Information')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        plt.show()
        
        return {
            'mse': mse,
            'r2_score': r2,
            'coefficients': coefficients,
            'model': lr,
            'test_data': (X_test, y_test, y_pred)
        }
    
    def visualize_logistic_regression(self, save_path: str = None) -> Dict[str, Any]:
        """Visualize Logistic Regression with demo data."""
        from sklearn.linear_model import LogisticRegression

        # Load demo data
        iris = load_iris()
        X, y = iris.data[:, :2], iris.target  # Use first 2 features for visualization
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
        
        # Scale features
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        # Train model
        lr = LogisticRegression(max_iter=1000, random_state=42)
        lr.fit(X_train_scaled, y_train)
        
        # Make predictions
        y_pred = lr.predict(X_test_scaled)
        y_prob = lr.predict_proba(X_test_scaled)
        accuracy = accuracy_score(y_test, y_pred)
        
        # Create visualization
        self.fig, self.axes = plt.subplots(2, 2, figsize=(15, 12))
        self.fig.suptitle('Logistic Regression Visualization - Iris Dataset', fontsize=16)
        
        # Decision boundary
        self._plot_decision_boundary(lr, X_train_scaled, y_train, self.axes[0, 0],
                                   title='Decision Boundary',
                                   feature_names=iris.feature_names[:2])
        
        # Confusion matrix
        from sklearn.metrics import confusion_matrix
        cm = confusion_matrix(y_test, y_pred)
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=self.axes[0, 1],
                   xticklabels=iris.target_names, yticklabels=iris.target_names)
        self.axes[0, 1].set_title('Confusion Matrix')
        self.axes[0, 1].set_ylabel('Actual')
        self.axes[0, 1].set_xlabel('Predicted')
        
        # Prediction probabilities
        prob_df = pd.DataFrame(y_prob, columns=iris.target_names)
        prob_df.plot(kind='box', ax=self.axes[1, 0])
        self.axes[1, 0].set_title('Prediction Probability Distribution')
        self.axes[1, 0].set_ylabel('Probability')
        
        # Model statistics
        model_info = f"""
Logistic Regression Statistics:
• Training Samples: {len(X_train)}
• Test Samples: {len(X_test)}
• Features: {X.shape[1]}
• Accuracy: {accuracy:.3f}
• Classes: {len(iris.target_names)}
• Solver: {lr.solver}
        """
        self.axes[1, 1].text(0.1, 0.5, model_info, fontsize=12, verticalalignment='center')
        self.axes[1, 1].axis('off')
        self.axes[1, 1].set_title('Model Information')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        plt.show()
        
        return {
            'accuracy': accuracy,
            'confusion_matrix': cm,
            'probabilities': y_prob,
            'model': lr,
            'test_data': (X_test_scaled, y_test, y_pred)
        }
    
    def _plot_decision_boundary(self, model, X, y, ax, title="Decision Boundary", feature_names=None):
        """Plot decision boundary for 2D data."""
        h = 0.02
        x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
        y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
        xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
        
        # Make predictions on mesh
        try:
            Z = model.predict(np.c_[xx.ravel(), yy.ravel()])
        except:
            # If model doesn't support 2D prediction, skip boundary
            ax.scatter(X[:, 0], X[:, 1], c=y, cmap=plt.cm.RdYlBu, edgecolors='black')
            ax.set_title(title)
            if feature_names:
                ax.set_xlabel(feature_names[0])
                ax.set_ylabel(feature_names[1])
            return
        
        Z = Z.reshape(xx.shape)
        
        # Plot decision boundary and data points
        ax.contourf(xx, yy, Z, alpha=0.8, cmap=plt.cm.RdYlBu)
        scatter = ax.scatter(X[:, 0], X[:, 1], c=y, cmap=plt.cm.RdYlBu, edgecolors='black')
        ax.set_title(title)
        
        if feature_names:
            ax.set_xlabel(feature_names[0])
            ax.set_ylabel(feature_names[1])

def run_algorithm_demo(algorithm_name: str) -> Dict[str, Any]:
    """Run a demo of the specified algorithm."""
    visualizer = AlgorithmVisualizer()
    
    import os
    os.makedirs('test-outputs/artifacts', exist_ok=True)
    save_path = f'test-outputs/artifacts/{algorithm_name.lower().replace(" ", "_")}_demo.png'
    
    if algorithm_name == "Decision Trees":
        return visualizer.visualize_decision_tree(save_path)
    elif algorithm_name == "Random Forest":
        return visualizer.visualize_random_forest(save_path)
    elif algorithm_name == "Linear Regression":
        return visualizer.visualize_linear_regression(save_path)
    elif algorithm_name == "Logistic Regression":
        return visualizer.visualize_logistic_regression(save_path)
    else:
        raise ValueError(f"Demo not available for {algorithm_name}")

if __name__ == "__main__":
    # Run all demos
    algorithms = ["Decision Trees", "Random Forest", "Linear Regression", "Logistic Regression"]
    
    for algorithm in algorithms:
        try:
            print(f"\n🔄 Running {algorithm} demo...")
            results = run_algorithm_demo(algorithm)
            print(f"✅ {algorithm} demo completed successfully!")
        except Exception as e:
            print(f"❌ {algorithm} demo failed: {e}")
    
    print(f"\n📁 Check 'test-outputs/artifacts/' for visualizations")
