"""
Performance comparison utilities for ML algorithms.
"""

import time
import warnings
from typing import Any, Dict, List, Tuple

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
from sklearn.model_selection import cross_val_score, train_test_split

warnings.filterwarnings('ignore')


class AlgorithmComparison:
    """Compare performance of multiple ML algorithms."""
    
    def __init__(self):
        self.results = {}
        self.datasets = {}
    
    def add_dataset(self, name: str, X: np.ndarray, y: np.ndarray, task_type: str = 'classification'):
        """Add a dataset for comparison."""
        self.datasets[name] = {
            'X': X,
            'y': y,
            'task_type': task_type
        }
    
    def compare_algorithms(self, algorithms: Dict[str, Any], cv_folds: int = 5) -> pd.DataFrame:
        """Compare multiple algorithms across all datasets."""
        all_results = []
        
        for dataset_name, dataset in self.datasets.items():
            X, y = dataset['X'], dataset['y']
            task_type = dataset['task_type']
            
            print(f"\nEvaluating on {dataset_name} dataset...")
            
            for algo_name, algorithm in algorithms.items():
                try:
                    # Time the training
                    start_time = time.time()
                    
                    # Cross-validation
                    if task_type == 'classification':
                        scores = cross_val_score(algorithm, X, y, cv=cv_folds, scoring='accuracy')
                        metric_name = 'accuracy'
                    else:
                        scores = cross_val_score(algorithm, X, y, cv=cv_folds, scoring='r2')
                        metric_name = 'r2_score'
                    
                    training_time = time.time() - start_time
                    
                    # Single train-test split for additional metrics
                    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
                    
                    start_time = time.time()
                    algorithm.fit(X_train, y_train)
                    fit_time = time.time() - start_time
                    
                    start_time = time.time()
                    y_pred = algorithm.predict(X_test)
                    predict_time = time.time() - start_time
                    
                    if task_type == 'classification':
                        test_score = accuracy_score(y_test, y_pred)
                    else:
                        test_score = r2_score(y_test, y_pred)
                    
                    result = {
                        'dataset': dataset_name,
                        'algorithm': algo_name,
                        'task_type': task_type,
                        f'cv_{metric_name}_mean': scores.mean(),
                        f'cv_{metric_name}_std': scores.std(),
                        f'test_{metric_name}': test_score,
                        'fit_time': fit_time,
                        'predict_time': predict_time,
                        'total_cv_time': training_time
                    }
                    
                    all_results.append(result)
                    print(f"  {algo_name}: {metric_name}={scores.mean():.3f}±{scores.std():.3f}")
                    
                except Exception as e:
                    print(f"  {algo_name}: Failed - {str(e)}")
                    continue
        
        self.results = pd.DataFrame(all_results)
        return self.results
    
    def plot_performance_comparison(self, metric: str = 'cv_accuracy_mean', save_path: str = None):
        """Plot performance comparison across algorithms and datasets."""
        if self.results.empty:
            print("No results to plot. Run compare_algorithms first.")
            return
        
        # Create pivot table for heatmap
        pivot_data = self.results.pivot(index='algorithm', columns='dataset', values=metric)
        
        plt.figure(figsize=(12, 8))
        
        # Performance heatmap
        plt.subplot(2, 2, 1)
        sns.heatmap(pivot_data, annot=True, fmt='.3f', cmap='viridis', 
                    cbar_kws={'label': metric})
        plt.title(f'Performance Comparison - {metric}')
        plt.xticks(rotation=45)
        plt.yticks(rotation=0)
        
        # Performance by algorithm (bar plot)
        plt.subplot(2, 2, 2)
        avg_performance = self.results.groupby('algorithm')[metric].mean().sort_values(ascending=False)
        plt.bar(range(len(avg_performance)), avg_performance.values)
        plt.xticks(range(len(avg_performance)), avg_performance.index, rotation=45)
        plt.ylabel(metric)
        plt.title('Average Performance by Algorithm')
        plt.grid(True, alpha=0.3)
        
        # Training time comparison
        plt.subplot(2, 2, 3)
        if 'fit_time' in self.results.columns:
            avg_time = self.results.groupby('algorithm')['fit_time'].mean().sort_values(ascending=True)
            plt.barh(range(len(avg_time)), avg_time.values)
            plt.yticks(range(len(avg_time)), avg_time.index)
            plt.xlabel('Average Fit Time (seconds)')
            plt.title('Training Time Comparison')
            plt.grid(True, alpha=0.3)
        
        # Performance vs Time scatter
        plt.subplot(2, 2, 4)
        if 'fit_time' in self.results.columns:
            for dataset in self.results['dataset'].unique():
                data = self.results[self.results['dataset'] == dataset]
                plt.scatter(data['fit_time'], data[metric], label=dataset, alpha=0.7)
            
            plt.xlabel('Fit Time (seconds)')
            plt.ylabel(metric)
            plt.title('Performance vs Training Time')
            plt.legend()
            plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
    
    def generate_report(self) -> str:
        """Generate a comprehensive comparison report."""
        if self.results.empty:
            return "No results available. Run compare_algorithms first."
        
        report = []
        report.append("=" * 60)
        report.append("ALGORITHM PERFORMANCE COMPARISON REPORT")
        report.append("=" * 60)
        
        # Summary by algorithm
        report.append("\n📊 PERFORMANCE SUMMARY BY ALGORITHM")
        report.append("-" * 40)
        
        for task_type in self.results['task_type'].unique():
            task_data = self.results[self.results['task_type'] == task_type]
            metric_col = f'cv_accuracy_mean' if task_type == 'classification' else 'cv_r2_score_mean'
            
            if metric_col in task_data.columns:
                report.append(f"\n{task_type.upper()} TASKS:")
                summary = task_data.groupby('algorithm')[metric_col].agg(['mean', 'std', 'count'])
                summary = summary.sort_values('mean', ascending=False)
                
                for algo, row in summary.iterrows():
                    report.append(f"  {algo:20} | Avg: {row['mean']:.3f} ± {row['std']:.3f} | Datasets: {row['count']}")
        
        # Performance by dataset
        report.append(f"\n📈 PERFORMANCE BY DATASET")
        report.append("-" * 40)
        
        for dataset in self.results['dataset'].unique():
            dataset_data = self.results[self.results['dataset'] == dataset]
            task_type = dataset_data['task_type'].iloc[0]
            metric_col = f'cv_accuracy_mean' if task_type == 'classification' else 'cv_r2_score_mean'
            
            if metric_col in dataset_data.columns:
                report.append(f"\n{dataset} ({task_type}):")
                best_performance = dataset_data.loc[dataset_data[metric_col].idxmax()]
                worst_performance = dataset_data.loc[dataset_data[metric_col].idxmin()]
                
                report.append(f"  Best:  {best_performance['algorithm']} ({best_performance[metric_col]:.3f})")
                report.append(f"  Worst: {worst_performance['algorithm']} ({worst_performance[metric_col]:.3f})")
        
        # Speed analysis
        if 'fit_time' in self.results.columns:
            report.append(f"\n⚡ SPEED ANALYSIS")
            report.append("-" * 40)
            
            speed_summary = self.results.groupby('algorithm')['fit_time'].agg(['mean', 'std'])
            speed_summary = speed_summary.sort_values('mean', ascending=True)
            
            report.append("\nAverage Training Time:")
            for algo, row in speed_summary.iterrows():
                report.append(f"  {algo:20} | {row['mean']:.4f} ± {row['std']:.4f} seconds")
        
        # Recommendations
        report.append(f"\n💡 RECOMMENDATIONS")
        report.append("-" * 40)
        
        # Best overall performer
        classification_data = self.results[self.results['task_type'] == 'classification']
        regression_data = self.results[self.results['task_type'] == 'regression']
        
        if not classification_data.empty and 'cv_accuracy_mean' in classification_data.columns:
            best_classifier = classification_data.loc[classification_data['cv_accuracy_mean'].idxmax()]
            report.append(f"\nBest Classifier: {best_classifier['algorithm']}")
            report.append(f"  Average Accuracy: {classification_data.groupby('algorithm')['cv_accuracy_mean'].mean().max():.3f}")
        
        if not regression_data.empty and 'cv_r2_score_mean' in regression_data.columns:
            best_regressor = regression_data.loc[regression_data['cv_r2_score_mean'].idxmax()]
            report.append(f"\nBest Regressor: {best_regressor['algorithm']}")
            report.append(f"  Average R² Score: {regression_data.groupby('algorithm')['cv_r2_score_mean'].mean().max():.3f}")
        
        # Speed vs accuracy trade-off
        if 'fit_time' in self.results.columns:
            report.append(f"\nSpeed vs Performance Trade-offs:")
            for task_type in self.results['task_type'].unique():
                task_data = self.results[self.results['task_type'] == task_type]
                metric_col = f'cv_accuracy_mean' if task_type == 'classification' else 'cv_r2_score_mean'
                
                if metric_col in task_data.columns:
                    # Find algorithms that are both fast and accurate
                    task_summary = task_data.groupby('algorithm').agg({
                        metric_col: 'mean',
                        'fit_time': 'mean'
                    })
                    
                    # Normalize scores for comparison
                    task_summary['performance_norm'] = (task_summary[metric_col] - task_summary[metric_col].min()) / (task_summary[metric_col].max() - task_summary[metric_col].min())
                    task_summary['speed_norm'] = 1 - ((task_summary['fit_time'] - task_summary['fit_time'].min()) / (task_summary['fit_time'].max() - task_summary['fit_time'].min()))
                    task_summary['balance_score'] = (task_summary['performance_norm'] + task_summary['speed_norm']) / 2
                    
                    best_balance = task_summary.loc[task_summary['balance_score'].idxmax()]
                    report.append(f"  {task_type}: {best_balance.name} (best speed/performance balance)")
        
        report.append("\n" + "=" * 60)
        
        return "\n".join(report)


def run_comprehensive_comparison():
    """Run a comprehensive algorithm comparison."""
    print("🔄 Starting Comprehensive Algorithm Comparison...")
    
    # Initialize comparison
    comparison = AlgorithmComparison()
    
    # Add datasets
    print("📊 Loading datasets...")
    
    # Classification datasets
    iris = load_iris()
    comparison.add_dataset('Iris', iris.data, iris.target, 'classification')
    
    # Synthetic classification
    X_class, y_class = make_classification(n_samples=1000, n_features=20, n_classes=3, random_state=42)
    comparison.add_dataset('Synthetic_Classification', X_class, y_class, 'classification')
    
    # Regression datasets
    diabetes = load_diabetes()
    comparison.add_dataset('Diabetes', diabetes.data, diabetes.target, 'regression')
    
    # Synthetic regression
    X_reg, y_reg = make_regression(n_samples=1000, n_features=10, noise=0.1, random_state=42)
    comparison.add_dataset('Synthetic_Regression', X_reg, y_reg, 'regression')
    
    # Define algorithms to compare
    print("🤖 Initializing algorithms...")
    from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
    from sklearn.linear_model import LinearRegression, LogisticRegression
    from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
    from sklearn.svm import SVC, SVR
    from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor

    # We'll need to handle different algorithms for different tasks
    classification_algorithms = {
        'Logistic_Regression': LogisticRegression(max_iter=1000, random_state=42),
        'Decision_Tree': DecisionTreeClassifier(random_state=42),
        'Random_Forest': RandomForestClassifier(n_estimators=100, random_state=42),
        'SVM': SVC(random_state=42),
        'KNN': KNeighborsClassifier()
    }
    
    regression_algorithms = {
        'Linear_Regression': LinearRegression(),
        'Decision_Tree': DecisionTreeRegressor(random_state=42),
        'Random_Forest': RandomForestRegressor(n_estimators=100, random_state=42),
        'SVM': SVR(),
        'KNN': KNeighborsRegressor()
    }
    
    # Run comparisons separately for classification and regression
    all_results = []
    
    # Classification comparison
    print("🎯 Running classification comparisons...")
    class_datasets = {k: v for k, v in comparison.datasets.items() if v['task_type'] == 'classification'}
    if class_datasets:
        class_comparison = AlgorithmComparison()
        class_comparison.datasets = class_datasets
        class_results = class_comparison.compare_algorithms(classification_algorithms)
        all_results.append(class_results)
    
    # Regression comparison  
    print("📈 Running regression comparisons...")
    reg_datasets = {k: v for k, v in comparison.datasets.items() if v['task_type'] == 'regression'}
    if reg_datasets:
        reg_comparison = AlgorithmComparison()
        reg_comparison.datasets = reg_datasets
        reg_results = reg_comparison.compare_algorithms(regression_algorithms)
        all_results.append(reg_results)
    
    # Combine results
    if all_results:
        combined_results = pd.concat(all_results, ignore_index=True)
        comparison.results = combined_results
        
        # Generate visualizations
        print("📊 Generating visualizations...")
        os.makedirs('test-outputs/artifacts', exist_ok=True)
        
        # Plot classification results
        if not class_results.empty:
            class_comparison.plot_performance_comparison(
                metric='cv_accuracy_mean',
                save_path='test-outputs/artifacts/classification_comparison.png'
            )
        
        # Plot regression results
        if not reg_results.empty:
            reg_comparison.plot_performance_comparison(
                metric='cv_r2_score_mean', 
                save_path='test-outputs/artifacts/regression_comparison.png'
            )
        
        # Generate comprehensive report
        print("📝 Generating report...")
        report = comparison.generate_report()
        
        # Save report
        with open('test-outputs/artifacts/algorithm_comparison_report.txt', 'w') as f:
            f.write(report)
        
        print("\n" + report)
        
        return comparison
    
    else:
        print("❌ No results generated")
        return None


if __name__ == "__main__":
    import os
    os.makedirs('test-outputs/artifacts', exist_ok=True)
    
    comparison = run_comprehensive_comparison()
    
    if comparison:
        print("\n🎉 Comprehensive algorithm comparison completed!")
        print("📁 Check 'test-outputs/artifacts/' for detailed results")
        print("   - classification_comparison.png")
        print("   - regression_comparison.png") 
        print("   - algorithm_comparison_report.txt")
