"""
Logistic Regression Example with Multiple Datasets
"""

import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.datasets import load_breast_cancer, load_iris, make_classification
from sklearn.linear_model import LogisticRegression as SklearnLogisticRegression
from sklearn.metrics import (
    accuracy_score,
    auc,
    classification_report,
    confusion_matrix,
    roc_curve,
)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


def iris_classification_example():
    """Logistic regression on iris dataset."""
    print("=" * 60)
    print("Logistic Regression Example - Iris Dataset")
    print("=" * 60)
    
    # Load iris dataset
    iris = load_iris()
    X, y = iris.data, iris.target
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    
    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Train Logistic Regression
    model = SklearnLogisticRegression(max_iter=1000, random_state=42)
    model.fit(X_train_scaled, y_train)
    
    # Make predictions
    y_pred = model.predict(X_test_scaled)
    y_pred_proba = model.predict_proba(X_test_scaled)
    
    # Calculate metrics
    accuracy = accuracy_score(y_test, y_pred)
    
    print(f"Accuracy: {accuracy:.3f}")
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, target_names=iris.target_names))
    
    # Visualizations
    plt.figure(figsize=(15, 10))
    
    # Confusion Matrix
    plt.subplot(2, 3, 1)
    cm = confusion_matrix(y_test, y_pred)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=iris.target_names, yticklabels=iris.target_names)
    plt.title('Confusion Matrix')
    plt.ylabel('Actual')
    plt.xlabel('Predicted')
    
    # Feature Coefficients
    plt.subplot(2, 3, 2)
    coefficients = model.coef_
    feature_names = iris.feature_names
    
    for i, class_name in enumerate(iris.target_names):
        plt.bar([f + i*0.25 for f in range(len(feature_names))], 
                coefficients[i], width=0.25, label=class_name, alpha=0.8)
    
    plt.xlabel('Features')
    plt.ylabel('Coefficient Value')
    plt.title('Feature Coefficients by Class')
    plt.xticks(range(len(feature_names)), feature_names, rotation=45)
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Decision Boundary (using first 2 features)
    plt.subplot(2, 3, 3)
    X_2d = X_test_scaled[:, :2]  # Use first 2 features for visualization
    
    # Create mesh
    h = 0.02
    x_min, x_max = X_2d[:, 0].min() - 1, X_2d[:, 0].max() + 1
    y_min, y_max = X_2d[:, 1].min() - 1, X_2d[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
    
    # Train model on 2D data for visualization
    model_2d = SklearnLogisticRegression(max_iter=1000, random_state=42)
    model_2d.fit(X_train_scaled[:, :2], y_train)
    
    # Predict on mesh
    Z = model_2d.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    
    plt.contourf(xx, yy, Z, alpha=0.8, cmap=plt.cm.RdYlBu)
    scatter = plt.scatter(X_2d[:, 0], X_2d[:, 1], c=y_test, cmap=plt.cm.RdYlBu, edgecolors='black')
    plt.xlabel(feature_names[0])
    plt.ylabel(feature_names[1])
    plt.title('Decision Boundary (2D)')
    plt.colorbar(scatter)
    
    # Prediction Probabilities
    plt.subplot(2, 3, 4)
    prob_df = pd.DataFrame(y_pred_proba, columns=iris.target_names)
    prob_df['Actual'] = y_test
    prob_df['Predicted'] = y_pred
    
    # Show probability distribution for each class
    for i, class_name in enumerate(iris.target_names):
        class_mask = y_test == i
        if np.any(class_mask):
            plt.hist(y_pred_proba[class_mask, i], bins=10, alpha=0.7, 
                    label=f'{class_name} (actual)', density=True)
    
    plt.xlabel('Prediction Probability')
    plt.ylabel('Density')
    plt.title('Prediction Probability Distribution')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # ROC Curves (One vs Rest)
    plt.subplot(2, 3, 5)
    for i, class_name in enumerate(iris.target_names):
        # Create binary labels (one vs rest)
        y_binary = (y_test == i).astype(int)
        y_score = y_pred_proba[:, i]
        
        fpr, tpr, _ = roc_curve(y_binary, y_score)
        roc_auc = auc(fpr, tpr)
        
        plt.plot(fpr, tpr, label=f'{class_name} (AUC = {roc_auc:.2f})')
    
    plt.plot([0, 1], [0, 1], 'k--', alpha=0.6)
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curves (One vs Rest)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Feature Importance Summary
    plt.subplot(2, 3, 6)
    feature_importance = np.abs(coefficients).mean(axis=0)
    plt.bar(range(len(feature_names)), feature_importance)
    plt.xlabel('Features')
    plt.ylabel('Average |Coefficient|')
    plt.title('Feature Importance')
    plt.xticks(range(len(feature_names)), feature_names, rotation=45)
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    os.makedirs('test-outputs/artifacts', exist_ok=True)
    plt.savefig('test-outputs/artifacts/logistic_regression_iris.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    return model, accuracy

def breast_cancer_classification_example():
    """Logistic regression on breast cancer dataset."""
    print("\n" + "=" * 60)
    print("Logistic Regression Example - Breast Cancer Dataset")
    print("=" * 60)
    
    # Load breast cancer dataset
    cancer = load_breast_cancer()
    X, y = cancer.data, cancer.target
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    
    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Train with different regularization strengths
    C_values = [0.001, 0.01, 0.1, 1.0, 10.0, 100.0]
    results = []
    
    for C in C_values:
        model = SklearnLogisticRegression(C=C, max_iter=1000, random_state=42)
        model.fit(X_train_scaled, y_train)
        
        train_score = model.score(X_train_scaled, y_train)
        test_score = model.score(X_test_scaled, y_test)
        
        results.append({
            'C': C,
            'train_accuracy': train_score,
            'test_accuracy': test_score,
            'difference': train_score - test_score
        })
    
    results_df = pd.DataFrame(results)
    print("Regularization Impact:")
    print(results_df)
    
    # Best model
    best_C = results_df.loc[results_df['test_accuracy'].idxmax(), 'C']
    best_model = SklearnLogisticRegression(C=best_C, max_iter=1000, random_state=42)
    best_model.fit(X_train_scaled, y_train)
    
    y_pred = best_model.predict(X_test_scaled)
    y_pred_proba = best_model.predict_proba(X_test_scaled)
    
    accuracy = accuracy_score(y_test, y_pred)
    print(f"\nBest Model (C={best_C}):")
    print(f"Test Accuracy: {accuracy:.3f}")
    
    # Visualizations
    plt.figure(figsize=(15, 5))
    
    # Regularization curve
    plt.subplot(1, 3, 1)
    plt.semilogx(results_df['C'], results_df['train_accuracy'], 'o-', label='Training')
    plt.semilogx(results_df['C'], results_df['test_accuracy'], 's-', label='Validation')
    plt.xlabel('Regularization Parameter (C)')
    plt.ylabel('Accuracy')
    plt.title('Regularization Impact')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Feature coefficients (top 20)
    plt.subplot(1, 3, 2)
    coefficients = best_model.coef_[0]
    feature_importance = pd.DataFrame({
        'Feature': cancer.feature_names,
        'Coefficient': coefficients,
        'Abs_Coefficient': np.abs(coefficients)
    }).sort_values('Abs_Coefficient', ascending=False).head(20)
    
    plt.barh(range(len(feature_importance)), feature_importance['Coefficient'])
    plt.yticks(range(len(feature_importance)), feature_importance['Feature'])
    plt.xlabel('Coefficient Value')
    plt.title('Top 20 Feature Coefficients')
    plt.grid(True, alpha=0.3)
    
    # ROC Curve
    plt.subplot(1, 3, 3)
    fpr, tpr, _ = roc_curve(y_test, y_pred_proba[:, 1])
    roc_auc = auc(fpr, tpr)
    
    plt.plot(fpr, tpr, label=f'ROC Curve (AUC = {roc_auc:.3f})')
    plt.plot([0, 1], [0, 1], 'k--', alpha=0.6)
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('test-outputs/artifacts/logistic_regression_breast_cancer.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    return best_model, results_df

if __name__ == "__main__":
    # Run examples
    iris_model, iris_accuracy = iris_classification_example()
    cancer_model, regularization_results = breast_cancer_classification_example()
    
    print("\n🎉 Logistic Regression examples completed successfully!")
    print("📁 Check 'test-outputs/artifacts/' for visualizations")
