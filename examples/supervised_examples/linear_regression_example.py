"""
Linear Regression Example with Scikit-learn Integration
"""

import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.datasets import load_diabetes, make_regression
from sklearn.linear_model import LinearRegression as SklearnLinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split


def diabetes_regression_example():
    """Linear regression on diabetes dataset."""
    print("=" * 60)
    print("Linear Regression Example - Diabetes Dataset")
    print("=" * 60)
    
    # Load diabetes dataset
    diabetes = load_diabetes()
    X, y = diabetes.data, diabetes.target
    feature_names = diabetes.feature_names
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    
    # Train Linear Regression
    model = SklearnLinearRegression()
    model.fit(X_train, y_train)
    
    # Make predictions
    y_pred = model.predict(X_test)
    
    # Calculate metrics
    mse = mean_squared_error(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    
    print(f"Mean Squared Error: {mse:.2f}")
    print(f"Mean Absolute Error: {mae:.2f}")
    print(f"R² Score: {r2:.3f}")
    
    # Feature importance (coefficients)
    coefficients = pd.DataFrame({
        'Feature': feature_names,
        'Coefficient': model.coef_,
        'Abs_Coefficient': np.abs(model.coef_)
    }).sort_values('Abs_Coefficient', ascending=False)
    
    print("\nFeature Importance (Coefficients):")
    print(coefficients)
    
    # Visualizations
    plt.figure(figsize=(15, 5))
    
    # Predictions vs Actual
    plt.subplot(1, 3, 1)
    plt.scatter(y_test, y_pred, alpha=0.6)
    plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
    plt.xlabel('Actual Values')
    plt.ylabel('Predicted Values')
    plt.title(f'Predictions vs Actual (R² = {r2:.3f})')
    plt.grid(True, alpha=0.3)
    
    # Residuals plot
    plt.subplot(1, 3, 2)
    residuals = y_test - y_pred
    plt.scatter(y_pred, residuals, alpha=0.6)
    plt.axhline(y=0, color='r', linestyle='--')
    plt.xlabel('Predicted Values')
    plt.ylabel('Residuals')
    plt.title('Residuals Plot')
    plt.grid(True, alpha=0.3)
    
    # Feature coefficients
    plt.subplot(1, 3, 3)
    plt.barh(range(len(coefficients)), coefficients['Coefficient'])
    plt.yticks(range(len(coefficients)), coefficients['Feature'])
    plt.xlabel('Coefficient Value')
    plt.title('Feature Coefficients')
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    os.makedirs('test-outputs/artifacts', exist_ok=True)
    plt.savefig('test-outputs/artifacts/linear_regression_diabetes.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    return model, coefficients

def synthetic_regression_example():
    """Linear regression on synthetic dataset with noise analysis."""
    print("\n" + "=" * 60)
    print("Linear Regression Example - Synthetic Data with Noise Analysis")
    print("=" * 60)
    
    # Generate synthetic data with different noise levels
    noise_levels = [0.1, 0.5, 1.0, 2.0]
    results = []
    
    plt.figure(figsize=(20, 5))
    
    for i, noise in enumerate(noise_levels):
        X, y = make_regression(n_samples=200, n_features=1, noise=noise*10, random_state=42)
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
        
        # Train model
        model = SklearnLinearRegression()
        model.fit(X_train, y_train)
        
        # Predict
        y_pred = model.predict(X_test)
        
        # Calculate metrics
        r2 = r2_score(y_test, y_pred)
        mse = mean_squared_error(y_test, y_pred)
        
        results.append({
            'noise_level': noise,
            'r2_score': r2,
            'mse': mse
        })
        
        # Plot
        plt.subplot(1, 4, i+1)
        
        # Sort for line plotting
        sort_idx = np.argsort(X_test.flatten())
        X_test_sorted = X_test[sort_idx]
        y_pred_sorted = y_pred[sort_idx]
        
        plt.scatter(X_test, y_test, alpha=0.6, label='Actual')
        plt.plot(X_test_sorted, y_pred_sorted, color='red', linewidth=2, label='Predicted')
        plt.xlabel('Feature')
        plt.ylabel('Target')
        plt.title(f'Noise Level: {noise}\nR² = {r2:.3f}')
        plt.legend()
        plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('test-outputs/artifacts/linear_regression_noise_analysis.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # Results summary
    results_df = pd.DataFrame(results)
    print("\nNoise Impact Analysis:")
    print(results_df)
    
    return results_df

if __name__ == "__main__":
    # Run examples
    diabetes_model, diabetes_coefficients = diabetes_regression_example()
    noise_results = synthetic_regression_example()
    
    print("\n🎉 Linear Regression examples completed successfully!")
    print("📁 Check 'test-outputs/artifacts/' for visualizations")
