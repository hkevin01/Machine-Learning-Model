"""
Decision Tree Example: Classification and Regression
"""

import matplotlib.pyplot as plt
import numpy as np
from sklearn.datasets import load_iris, make_classification, make_regression
from sklearn.metrics import accuracy_score, classification_report, mean_squared_error
from sklearn.model_selection import train_test_split

from machine_learning_model.supervised.decision_tree import (
    DecisionTreeClassifier,
    DecisionTreeRegressor,
)


def classification_example():
    """Demonstrate decision tree classification."""
    print("=" * 50)
    print("Decision Tree Classification Example")
    print("=" * 50)
    
    # Load iris dataset
    iris = load_iris()
    X, y = iris.data, iris.target
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    
    # Train decision tree
    clf = DecisionTreeClassifier(max_depth=5, random_state=42)
    clf.fit(X_train, y_train)
    
    # Make predictions
    y_pred = clf.predict(X_test)
    
    # Evaluate
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Accuracy: {accuracy:.3f}")
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, target_names=iris.target_names))
    
    # Feature importances
    print("\nFeature Importances:")
    for i, importance in enumerate(clf.feature_importances_):
        print(f"{iris.feature_names[i]}: {importance:.3f}")
    
    return clf


def regression_example():
    """Demonstrate decision tree regression."""
    print("\n" + "=" * 50)
    print("Decision Tree Regression Example")
    print("=" * 50)
    
    # Generate regression dataset
    X, y = make_regression(n_samples=200, n_features=1, noise=20, random_state=42)
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    
    # Train decision tree
    reg = DecisionTreeRegressor(max_depth=5, random_state=42)
    reg.fit(X_train, y_train)
    
    # Make predictions
    y_pred = reg.predict(X_test)
    
    # Evaluate
    mse = mean_squared_error(y_test, y_pred)
    print(f"Mean Squared Error: {mse:.3f}")
    
    # Visualize results
    plt.figure(figsize=(10, 6))
    
    # Sort for plotting
    sort_idx = np.argsort(X_test.flatten())
    X_test_sorted = X_test[sort_idx]
    y_test_sorted = y_test[sort_idx]
    y_pred_sorted = y_pred[sort_idx]
    
    plt.scatter(X_test, y_test, alpha=0.6, label='True values')
    plt.plot(X_test_sorted, y_pred_sorted, color='red', linewidth=2, label='Predictions')
    plt.xlabel('Feature')
    plt.ylabel('Target')
    plt.title('Decision Tree Regression Results')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('test-outputs/artifacts/decision_tree_regression.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    return reg


def hyperparameter_comparison():
    """Compare different hyperparameter settings."""
    print("\n" + "=" * 50)
    print("Hyperparameter Comparison")
    print("=" * 50)
    
    # Generate dataset
    X, y = make_classification(n_samples=300, n_features=10, n_classes=3, random_state=42)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    
    # Test different max_depth values
    depths = [3, 5, 10, None]
    results = []
    
    for depth in depths:
        clf = DecisionTreeClassifier(max_depth=depth, random_state=42)
        clf.fit(X_train, y_train)
        
        train_acc = accuracy_score(y_train, clf.predict(X_train))
        test_acc = accuracy_score(y_test, clf.predict(X_test))
        
        results.append({
            'max_depth': depth,
            'train_accuracy': train_acc,
            'test_accuracy': test_acc
        })
        
        print(f"Max Depth: {depth}")
        print(f"  Train Accuracy: {train_acc:.3f}")
        print(f"  Test Accuracy: {test_acc:.3f}")
        print()
    
    return results


if __name__ == "__main__":
    # Create output directory
    import os
    os.makedirs('test-outputs/artifacts', exist_ok=True)
    
    # Run examples
    classifier = classification_example()
    regressor = regression_example()
    comparison_results = hyperparameter_comparison()
    
    print("\n🎉 Decision Tree examples completed successfully!")
    print("Check 'test-outputs/artifacts/' for visualizations.")
