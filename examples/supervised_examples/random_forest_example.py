"""
Random Forest Example: Classification and Regression with Feature Importance Analysis
"""

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.datasets import load_diabetes, load_wine
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    mean_squared_error,
    r2_score,
)
from sklearn.model_selection import train_test_split

from machine_learning_model.supervised.decision_tree import (
    DecisionTreeClassifier,
    DecisionTreeRegressor,
)
from machine_learning_model.supervised.random_forest import (
    RandomForestClassifier,
    RandomForestRegressor,
)


def classification_example():
    """Demonstrate Random Forest classification with feature importance."""
    print("=" * 60)
    print("Random Forest Classification Example")
    print("=" * 60)
    
    # Load wine dataset
    wine = load_wine()
    X, y = wine.data, wine.target
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    
    # Train Random Forest
    rf_clf = RandomForestClassifier(
        n_estimators=100, 
        max_depth=10, 
        oob_score=True, 
        random_state=42
    )
    rf_clf.fit(X_train, y_train)
    
    # Train single Decision Tree for comparison
    dt_clf = DecisionTreeClassifier(max_depth=10, random_state=42)
    dt_clf.fit(X_train, y_train)
    
    # Make predictions
    rf_pred = rf_clf.predict(X_test)
    dt_pred = dt_clf.predict(X_test)
    
    # Evaluate
    rf_accuracy = accuracy_score(y_test, rf_pred)
    dt_accuracy = accuracy_score(y_test, dt_pred)
    
    print(f"Random Forest Accuracy: {rf_accuracy:.3f}")
    print(f"Decision Tree Accuracy: {dt_accuracy:.3f}")
    print(f"Random Forest OOB Score: {rf_clf.oob_score_:.3f}")
    
    print("\nRandom Forest Classification Report:")
    print(classification_report(y_test, rf_pred, target_names=wine.target_names))
    
    # Feature importances comparison
    plt.figure(figsize=(15, 5))
    
    # Random Forest importances
    plt.subplot(1, 2, 1)
    indices = np.argsort(rf_clf.feature_importances_)[::-1]
    plt.title('Random Forest Feature Importances')
    plt.bar(range(len(rf_clf.feature_importances_)), rf_clf.feature_importances_[indices])
    plt.xticks(range(len(rf_clf.feature_importances_)), 
               [wine.feature_names[i] for i in indices], rotation=45)
    
    # Decision Tree importances
    plt.subplot(1, 2, 2)
    indices_dt = np.argsort(dt_clf.feature_importances_)[::-1]
    plt.title('Decision Tree Feature Importances')
    plt.bar(range(len(dt_clf.feature_importances_)), dt_clf.feature_importances_[indices_dt])
    plt.xticks(range(len(dt_clf.feature_importances_)), 
               [wine.feature_names[i] for i in indices_dt], rotation=45)
    
    plt.tight_layout()
    plt.savefig('test-outputs/artifacts/random_forest_feature_importance.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    return rf_clf, dt_clf


def regression_example():
    """Demonstrate Random Forest regression."""
    print("\n" + "=" * 60)
    print("Random Forest Regression Example")
    print("=" * 60)
    
    # Load diabetes dataset
    diabetes = load_diabetes()
    X, y = diabetes.data, diabetes.target
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    
    # Train Random Forest
    rf_reg = RandomForestRegressor(
        n_estimators=100, 
        max_depth=10, 
        oob_score=True, 
        random_state=42
    )
    rf_reg.fit(X_train, y_train)
    
    # Train single Decision Tree for comparison
    dt_reg = DecisionTreeRegressor(max_depth=10, random_state=42)
    dt_reg.fit(X_train, y_train)
    
    # Make predictions
    rf_pred = rf_reg.predict(X_test)
    dt_pred = dt_reg.predict(X_test)
    
    # Evaluate
    rf_mse = mean_squared_error(y_test, rf_pred)
    dt_mse = mean_squared_error(y_test, dt_pred)
    rf_r2 = r2_score(y_test, rf_pred)
    dt_r2 = r2_score(y_test, dt_pred)
    
    print(f"Random Forest MSE: {rf_mse:.3f}")
    print(f"Decision Tree MSE: {dt_mse:.3f}")
    print(f"Random Forest R²: {rf_r2:.3f}")
    print(f"Decision Tree R²: {dt_r2:.3f}")
    print(f"Random Forest OOB Score: {rf_reg.oob_score_:.3f}")
    
    # Visualize predictions
    plt.figure(figsize=(15, 5))
    
    # Random Forest predictions
    plt.subplot(1, 2, 1)
    plt.scatter(y_test, rf_pred, alpha=0.6)
    plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
    plt.xlabel('Actual')
    plt.ylabel('Predicted')
    plt.title(f'Random Forest Predictions (R² = {rf_r2:.3f})')
    plt.grid(True, alpha=0.3)
    
    # Decision Tree predictions
    plt.subplot(1, 2, 2)
    plt.scatter(y_test, dt_pred, alpha=0.6)
    plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
    plt.xlabel('Actual')
    plt.ylabel('Predicted')
    plt.title(f'Decision Tree Predictions (R² = {dt_r2:.3f})')
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('test-outputs/artifacts/random_forest_regression.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    return rf_reg, dt_reg


def hyperparameter_analysis():
    """Analyze effect of different hyperparameters."""
    print("\n" + "=" * 60)
    print("Random Forest Hyperparameter Analysis")
    print("=" * 60)
    
    from sklearn.datasets import make_classification
    
    X, y = make_classification(n_samples=500, n_features=20, n_classes=3, random_state=42)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    
    # Test different n_estimators
    n_estimators_range = [10, 25, 50, 100, 200]
    results = []
    
    for n_est in n_estimators_range:
        rf = RandomForestClassifier(n_estimators=n_est, oob_score=True, random_state=42)
        rf.fit(X_train, y_train)
        
        train_acc = accuracy_score(y_train, rf.predict(X_train))
        test_acc = accuracy_score(y_test, rf.predict(X_test))
        oob_acc = rf.oob_score_
        
        results.append({
            'n_estimators': n_est,
            'train_accuracy': train_acc,
            'test_accuracy': test_acc,
            'oob_accuracy': oob_acc
        })
        
        print(f"n_estimators={n_est:3d}: Train={train_acc:.3f}, Test={test_acc:.3f}, OOB={oob_acc:.3f}")
    
    # Visualize results
    results_data = np.array([[r['train_accuracy'], r['test_accuracy'], r['oob_accuracy']] for r in results])
    
    plt.figure(figsize=(10, 6))
    plt.plot(n_estimators_range, results_data[:, 0], 'o-', label='Training Accuracy')
    plt.plot(n_estimators_range, results_data[:, 1], 's-', label='Test Accuracy')
    plt.plot(n_estimators_range, results_data[:, 2], '^-', label='OOB Accuracy')
    
    plt.xlabel('Number of Estimators')
    plt.ylabel('Accuracy')
    plt.title('Random Forest Performance vs Number of Estimators')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('test-outputs/artifacts/random_forest_hyperparameters.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    return results


if __name__ == "__main__":
    # Create output directory
    import os
    os.makedirs('test-outputs/artifacts', exist_ok=True)
    
    # Run examples
    rf_classifier, dt_classifier = classification_example()
    rf_regressor, dt_regressor = regression_example()
    hyperparameter_results = hyperparameter_analysis()
    
    print("\n🎉 Random Forest examples completed successfully!")
    print("📊 Feature importance and performance comparisons generated")
    print("📁 Check 'test-outputs/artifacts/' for visualizations")
