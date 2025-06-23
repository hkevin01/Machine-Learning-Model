"""
ML Workflow - Complete Machine Learning Pipeline Implementation

This module provides a comprehensive, step-by-step workflow for machine learning projects,
integrating data collection through deployment with intelligent guidance and automation.
"""

import logging
import os
from typing import Any, Dict, List, Optional, Tuple
from dataclasses import dataclass
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
import matplotlib.pyplot as plt
import seaborn as sns

from .ml_agent import MLAgent, WorkflowStep, WorkflowStepStatus, WorkflowStepType

logger = logging.getLogger(__name__)


class MLWorkflow:
    """
    Complete Machine Learning Workflow Implementation.
    
    This class provides a comprehensive ML pipeline that integrates with the MLAgent
    to provide guided, step-by-step machine learning project execution.
    """
    
    def __init__(self, agent: MLAgent):
        """
        Initialize the ML Workflow.
        
        Args:
            agent: The ML Agent that manages workflow state and guidance
        """
        self.agent = agent
        self.data: Optional[pd.DataFrame] = None
        self.target_column: Optional[str] = None
        self.X_train: Optional[pd.DataFrame] = None
        self.X_val: Optional[pd.DataFrame] = None
        self.X_test: Optional[pd.DataFrame] = None
        self.y_train: Optional[pd.Series] = None
        self.y_val: Optional[pd.Series] = None
        self.y_test: Optional[pd.Series] = None
        self.models: Dict[str, Any] = {}
        self.scalers: Dict[str, Any] = {}
        self.encoders: Dict[str, Any] = {}
        self.results: Dict[str, Any] = {}
        
        # Create workspace directories
        self._setup_workspace()
    
    def _setup_workspace(self):
        """Set up the workspace directory structure."""
        base_dir = self.agent.workspace_dir
        directories = [
            "data/raw",
            "data/processed", 
            "data/interim",
            "data/features",
            "models/trained",
            "models/experiments",
            "results/plots",
            "results/reports",
            "logs"
        ]
        
        for directory in directories:
            full_path = os.path.join(base_dir, directory)
            os.makedirs(full_path, exist_ok=True)
    
    def execute_current_step(self, **kwargs) -> bool:
        """
        Execute the current workflow step.
        
        Args:
            **kwargs: Step-specific parameters
            
        Returns:
            True if step executed successfully, False otherwise
        """
        current_step = self.agent.get_current_step()
        if not current_step:
            logger.info("Workflow completed!")
            return True
        
        try:
            self.agent.start_current_step()
            
            if current_step.step_type == WorkflowStepType.DATA_COLLECTION:
                success = self._execute_data_collection(**kwargs)
            elif current_step.step_type == WorkflowStepType.DATA_PREPROCESSING:
                success = self._execute_data_preprocessing(**kwargs)
            elif current_step.step_type == WorkflowStepType.EXPLORATORY_DATA_ANALYSIS:
                success = self._execute_eda(**kwargs)
            elif current_step.step_type == WorkflowStepType.FEATURE_ENGINEERING:
                success = self._execute_feature_engineering(**kwargs)
            elif current_step.step_type == WorkflowStepType.DATA_SPLITTING:
                success = self._execute_data_splitting(**kwargs)
            elif current_step.step_type == WorkflowStepType.ALGORITHM_SELECTION:
                success = self._execute_algorithm_selection(**kwargs)
            elif current_step.step_type == WorkflowStepType.MODEL_TRAINING:
                success = self._execute_model_training(**kwargs)
            elif current_step.step_type == WorkflowStepType.MODEL_EVALUATION:
                success = self._execute_model_evaluation(**kwargs)
            elif current_step.step_type == WorkflowStepType.HYPERPARAMETER_TUNING:
                success = self._execute_hyperparameter_tuning(**kwargs)
            elif current_step.step_type == WorkflowStepType.MODEL_DEPLOYMENT:
                success = self._execute_model_deployment(**kwargs)
            elif current_step.step_type == WorkflowStepType.MONITORING:
                success = self._execute_monitoring(**kwargs)
            else:
                logger.error(f"Unknown step type: {current_step.step_type}")
                success = False
            
            if success:
                self.agent.complete_current_step()
                logger.info(f"Successfully completed step: {current_step.name}")
            else:
                self.agent.fail_current_step("Step execution failed")
                logger.error(f"Failed to complete step: {current_step.name}")
            
            return success
            
        except Exception as e:
            error_msg = f"Error executing step {current_step.name}: {str(e)}"
            logger.error(error_msg)
            self.agent.fail_current_step(error_msg)
            return False
    
    def _execute_data_collection(self, **kwargs) -> bool:
        """Execute data collection step."""
        logger.info("Executing data collection step...")
        
        # Check for provided data
        if 'data' in kwargs:
            self.data = kwargs['data']
            self.target_column = kwargs.get('target_column')
        elif 'file_path' in kwargs:
            # Load data from file
            file_path = kwargs['file_path']
            if file_path.endswith('.csv'):
                self.data = pd.read_csv(file_path)
            elif file_path.endswith('.json'):
                self.data = pd.read_json(file_path)
            else:
                logger.error(f"Unsupported file format: {file_path}")
                return False
            self.target_column = kwargs.get('target_column')
        else:
            # Load sample dataset for demonstration
            from ..data.sample_datasets import SampleDatasetManager
            dataset_manager = SampleDatasetManager()
            
            # Default to Iris dataset for demo
            dataset_name = kwargs.get('dataset', 'iris')
            dataset = dataset_manager.get_dataset(dataset_name)
            if dataset:
                self.data = dataset['data']
                self.target_column = dataset['target_column']
            else:
                logger.error(f"Failed to load dataset: {dataset_name}")
                return False
        
        if self.data is None:
            logger.error("No data loaded")
            return False
        
        # Save raw data
        raw_data_path = os.path.join(self.agent.workspace_dir, "data/raw/dataset.csv")
        self.data.to_csv(raw_data_path, index=False)
        
        # Update metadata
        metadata = {
            "data_shape": self.data.shape,
            "target_column": self.target_column,
            "features": list(self.data.columns),
            "data_types": self.data.dtypes.to_dict(),
            "missing_values": self.data.isnull().sum().to_dict()
        }
        
        self.agent.get_current_step().metadata.update(metadata)
        logger.info(f"Loaded dataset with shape: {self.data.shape}")
        return True
    
    def _execute_data_preprocessing(self, **kwargs) -> bool:
        """Execute data preprocessing step."""
        logger.info("Executing data preprocessing step...")
        
        if self.data is None:
            logger.error("No data available for preprocessing")
            return False
        
        # Create a copy for processing
        processed_data = self.data.copy()
        
        # Handle missing values
        missing_strategy = kwargs.get('missing_strategy', 'auto')
        if missing_strategy == 'auto':
            # Intelligent missing value handling
            for column in processed_data.columns:
                if processed_data[column].isnull().any():
                    if processed_data[column].dtype in ['int64', 'float64']:
                        # Fill numerical columns with median
                        processed_data[column].fillna(processed_data[column].median(), inplace=True)
                    else:
                        # Fill categorical columns with mode
                        processed_data[column].fillna(processed_data[column].mode()[0], inplace=True)
        
        # Handle outliers (simple IQR method for numerical columns)
        remove_outliers = kwargs.get('remove_outliers', False)
        if remove_outliers:
            numerical_columns = processed_data.select_dtypes(include=[np.number]).columns
            for column in numerical_columns:
                if column != self.target_column:
                    Q1 = processed_data[column].quantile(0.25)
                    Q3 = processed_data[column].quantile(0.75)
                    IQR = Q3 - Q1
                    lower_bound = Q1 - 1.5 * IQR
                    upper_bound = Q3 + 1.5 * IQR
                    processed_data = processed_data[
                        (processed_data[column] >= lower_bound) & 
                        (processed_data[column] <= upper_bound)
                    ]
        
        # Encode categorical variables
        categorical_columns = processed_data.select_dtypes(include=['object']).columns
        for column in categorical_columns:
            if column != self.target_column:
                le = LabelEncoder()
                processed_data[column] = le.fit_transform(processed_data[column])
                self.encoders[column] = le
        
        self.data = processed_data
        
        # Save processed data
        processed_data_path = os.path.join(self.agent.workspace_dir, "data/processed/dataset_processed.csv")
        self.data.to_csv(processed_data_path, index=False)
        
        # Update metadata
        metadata = {
            "original_shape": kwargs.get('original_shape', self.data.shape),
            "processed_shape": self.data.shape,
            "outliers_removed": remove_outliers,
            "encoded_columns": list(self.encoders.keys()),
            "missing_values_after": self.data.isnull().sum().to_dict()
        }
        
        self.agent.get_current_step().metadata.update(metadata)
        logger.info(f"Preprocessing completed. Data shape: {self.data.shape}")
        return True
    
    def _execute_eda(self, **kwargs) -> bool:
        """Execute exploratory data analysis step."""
        logger.info("Executing exploratory data analysis step...")
        
        if self.data is None:
            logger.error("No data available for EDA")
            return False
        
        # Create plots directory
        plots_dir = os.path.join(self.agent.workspace_dir, "results/plots")
        os.makedirs(plots_dir, exist_ok=True)
        
        # Basic statistics
        basic_stats = self.data.describe()
        basic_stats.to_csv(os.path.join(plots_dir, "basic_statistics.csv"))
        
        # Correlation matrix for numerical features
        numerical_data = self.data.select_dtypes(include=[np.number])
        if len(numerical_data.columns) > 1:
            plt.figure(figsize=(10, 8))
            correlation_matrix = numerical_data.corr()
            sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', center=0)
            plt.title('Feature Correlation Matrix')
            plt.tight_layout()
            plt.savefig(os.path.join(plots_dir, "correlation_matrix.png"), dpi=300, bbox_inches='tight')
            plt.close()
        
        # Distribution plots for numerical features
        if len(numerical_data.columns) > 0:
            fig, axes = plt.subplots(nrows=(len(numerical_data.columns) + 2) // 3, ncols=3, 
                                   figsize=(15, 5 * ((len(numerical_data.columns) + 2) // 3)))
            axes = axes.flatten() if len(numerical_data.columns) > 3 else [axes]
            
            for i, column in enumerate(numerical_data.columns):
                if i < len(axes):
                    numerical_data[column].hist(bins=30, ax=axes[i], alpha=0.7)
                    axes[i].set_title(f'Distribution of {column}')
                    axes[i].set_xlabel(column)
                    axes[i].set_ylabel('Frequency')
            
            # Hide empty subplots
            for i in range(len(numerical_data.columns), len(axes)):
                axes[i].set_visible(False)
            
            plt.tight_layout()
            plt.savefig(os.path.join(plots_dir, "feature_distributions.png"), dpi=300, bbox_inches='tight')
            plt.close()
        
        # Target variable analysis
        if self.target_column and self.target_column in self.data.columns:
            plt.figure(figsize=(10, 6))
            if self.data[self.target_column].dtype in ['int64', 'float64']:
                # Continuous target
                self.data[self.target_column].hist(bins=30, alpha=0.7)
                plt.title(f'Distribution of Target Variable: {self.target_column}')
                plt.xlabel(self.target_column)
                plt.ylabel('Frequency')
            else:
                # Categorical target
                self.data[self.target_column].value_counts().plot(kind='bar')
                plt.title(f'Class Distribution: {self.target_column}')
                plt.xlabel('Classes')
                plt.ylabel('Count')
                plt.xticks(rotation=45)
            
            plt.tight_layout()
            plt.savefig(os.path.join(plots_dir, "target_distribution.png"), dpi=300, bbox_inches='tight')
            plt.close()
        
        # Update metadata
        metadata = {
            "basic_statistics": basic_stats.to_dict(),
            "data_types": self.data.dtypes.to_dict(),
            "unique_values": {col: self.data[col].nunique() for col in self.data.columns},
            "plots_generated": ["correlation_matrix.png", "feature_distributions.png", "target_distribution.png"]
        }
        
        self.agent.get_current_step().metadata.update(metadata)
        logger.info("EDA completed successfully")
        return True
    
    def _execute_feature_engineering(self, **kwargs) -> bool:
        """Execute feature engineering step."""
        logger.info("Executing feature engineering step...")
        
        if self.data is None:
            logger.error("No data available for feature engineering")
            return False
        
        # Create a copy for feature engineering
        engineered_data = self.data.copy()
        
        # Feature scaling for numerical features
        numerical_columns = engineered_data.select_dtypes(include=[np.number]).columns
        if self.target_column in numerical_columns:
            numerical_columns = numerical_columns.drop(self.target_column)
        
        scale_features = kwargs.get('scale_features', True)
        if scale_features and len(numerical_columns) > 0:
            scaler = StandardScaler()
            engineered_data[numerical_columns] = scaler.fit_transform(engineered_data[numerical_columns])
            self.scalers['feature_scaler'] = scaler
        
        # Feature selection based on correlation (optional)
        remove_high_correlation = kwargs.get('remove_high_correlation', False)
        correlation_threshold = kwargs.get('correlation_threshold', 0.95)
        
        if remove_high_correlation and len(numerical_columns) > 1:
            correlation_matrix = engineered_data[numerical_columns].corr().abs()
            upper_triangle = correlation_matrix.where(
                np.triu(np.ones(correlation_matrix.shape), k=1).astype(bool)
            )
            
            # Find features with high correlation
            high_corr_features = [column for column in upper_triangle.columns 
                                if any(upper_triangle[column] > correlation_threshold)]
            
            if high_corr_features:
                engineered_data = engineered_data.drop(columns=high_corr_features)
                logger.info(f"Removed highly correlated features: {high_corr_features}")
        
        self.data = engineered_data
        
        # Save feature-engineered data
        features_data_path = os.path.join(self.agent.workspace_dir, "data/features/dataset_features.csv")
        self.data.to_csv(features_data_path, index=False)
        
        # Update metadata
        metadata = {
            "original_features": kwargs.get('original_features', list(self.data.columns)),
            "final_features": list(self.data.columns),
            "scaling_applied": scale_features,
            "correlation_filtering": remove_high_correlation,
            "removed_features": kwargs.get('removed_features', [])
        }
        
        self.agent.get_current_step().metadata.update(metadata)
        logger.info("Feature engineering completed successfully")
        return True
    
    def _execute_data_splitting(self, **kwargs) -> bool:
        """Execute data splitting step."""
        logger.info("Executing data splitting step...")
        
        if self.data is None or self.target_column is None:
            logger.error("No data or target column available for splitting")
            return False
        
        # Prepare features and target
        X = self.data.drop(columns=[self.target_column])
        y = self.data[self.target_column]
        
        # Split parameters
        test_size = kwargs.get('test_size', 0.2)
        val_size = kwargs.get('val_size', 0.2)
        random_state = kwargs.get('random_state', 42)
        stratify = kwargs.get('stratify', True)
        
        # Determine if we should stratify (for classification)
        stratify_param = y if stratify and y.nunique() < 20 else None
        
        # First split: separate test set
        X_temp, self.X_test, y_temp, self.y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state, stratify=stratify_param
        )
        
        # Second split: separate train and validation
        val_size_adjusted = val_size / (1 - test_size)  # Adjust for remaining data
        stratify_temp = y_temp if stratify and y_temp.nunique() < 20 else None
        
        self.X_train, self.X_val, self.y_train, self.y_val = train_test_split(
            X_temp, y_temp, test_size=val_size_adjusted, random_state=random_state, 
            stratify=stratify_temp
        )
        
        # Save splits
        splits_dir = os.path.join(self.agent.workspace_dir, "data/processed")
        self.X_train.to_csv(os.path.join(splits_dir, "X_train.csv"), index=False)
        self.X_val.to_csv(os.path.join(splits_dir, "X_val.csv"), index=False)
        self.X_test.to_csv(os.path.join(splits_dir, "X_test.csv"), index=False)
        self.y_train.to_csv(os.path.join(splits_dir, "y_train.csv"), index=False)
        self.y_val.to_csv(os.path.join(splits_dir, "y_val.csv"), index=False)
        self.y_test.to_csv(os.path.join(splits_dir, "y_test.csv"), index=False)
        
        # Update metadata
        metadata = {
            "train_size": len(self.X_train),
            "val_size": len(self.X_val),
            "test_size": len(self.X_test),
            "train_ratio": len(self.X_train) / len(X),
            "val_ratio": len(self.X_val) / len(X),
            "test_ratio": len(self.X_test) / len(X),
            "stratified": stratify,
            "random_state": random_state,
            "target_distribution_train": self.y_train.value_counts().to_dict() if self.y_train.nunique() < 20 else "continuous"
        }
        
        self.agent.get_current_step().metadata.update(metadata)
        logger.info(f"Data splitting completed: Train={len(self.X_train)}, Val={len(self.X_val)}, Test={len(self.X_test)}")
        return True
    
    def _execute_algorithm_selection(self, **kwargs) -> bool:
        """Execute algorithm selection step."""
        logger.info("Executing algorithm selection step...")
        
        if self.X_train is None or self.y_train is None:
            logger.error("Training data not available for algorithm selection")
            return False
        
        # Determine problem type
        is_classification = self.y_train.nunique() < 20 and self.y_train.dtype == 'object' or self.y_train.nunique() <= 10
        problem_type = "classification" if is_classification else "regression"
        
        # Import available algorithms
        from ..supervised.decision_tree import DecisionTreeClassifier as CustomDecisionTreeClassifier
        from ..supervised.decision_tree import DecisionTreeRegressor as CustomDecisionTreeRegressor
        from ..supervised.random_forest import RandomForestClassifier as CustomRandomForestClassifier
        from ..supervised.random_forest import RandomForestRegressor as CustomRandomForestRegressor
        
        # Also import sklearn as baseline
        from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
        from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
        from sklearn.linear_model import LogisticRegression, LinearRegression
        
        # Select algorithms based on problem type and preferences
        selected_algorithms = kwargs.get('algorithms', 'auto')
        
        if selected_algorithms == 'auto':
            if is_classification:
                self.models = {
                    'custom_decision_tree': CustomDecisionTreeClassifier(),
                    'custom_random_forest': CustomRandomForestClassifier(n_estimators=10),
                    'sklearn_decision_tree': DecisionTreeClassifier(random_state=42),
                    'sklearn_random_forest': RandomForestClassifier(n_estimators=10, random_state=42),
                    'logistic_regression': LogisticRegression(random_state=42, max_iter=1000)
                }
            else:
                self.models = {
                    'custom_decision_tree': CustomDecisionTreeRegressor(),
                    'custom_random_forest': CustomRandomForestRegressor(n_estimators=10),
                    'sklearn_decision_tree': DecisionTreeRegressor(random_state=42),
                    'sklearn_random_forest': RandomForestRegressor(n_estimators=10, random_state=42),
                    'linear_regression': LinearRegression()
                }
        else:
            # Use user-provided algorithms
            self.models = selected_algorithms
        
        # Update metadata
        metadata = {
            "problem_type": problem_type,
            "selected_algorithms": list(self.models.keys()),
            "n_features": self.X_train.shape[1],
            "n_samples": self.X_train.shape[0],
            "target_classes": self.y_train.nunique() if is_classification else "continuous"
        }
        
        self.agent.get_current_step().metadata.update(metadata)
        logger.info(f"Algorithm selection completed for {problem_type} problem with {len(self.models)} algorithms")
        return True
    
    def _execute_model_training(self, **kwargs) -> bool:
        """Execute model training step."""
        logger.info("Executing model training step...")
        
        if not self.models or self.X_train is None or self.y_train is None:
            logger.error("Models or training data not available")
            return False
        
        trained_models = {}
        training_results = {}
        
        for model_name, model in self.models.items():
            try:
                logger.info(f"Training {model_name}...")
                
                # Train the model
                model.fit(self.X_train, self.y_train)
                trained_models[model_name] = model
                
                # Basic training metrics (if available)
                if hasattr(model, 'score'):
                    train_score = model.score(self.X_train, self.y_train)
                    training_results[model_name] = {'train_score': train_score}
                else:
                    training_results[model_name] = {'train_score': 'N/A'}
                
                logger.info(f"Successfully trained {model_name}")
                
            except Exception as e:
                logger.error(f"Failed to train {model_name}: {str(e)}")
                training_results[model_name] = {'error': str(e)}
        
        self.models = trained_models
        
        # Update metadata
        metadata = {
            "trained_models": list(trained_models.keys()),
            "training_results": training_results,
            "failed_models": [name for name, result in training_results.items() if 'error' in result]
        }
        
        self.agent.get_current_step().metadata.update(metadata)
        logger.info(f"Model training completed. Successfully trained {len(trained_models)} models")
        return True
    
    def _execute_model_evaluation(self, **kwargs) -> bool:
        """Execute model evaluation step."""
        logger.info("Executing model evaluation step...")
        
        if not self.models or self.X_val is None or self.y_val is None:
            logger.error("Models or validation data not available")
            return False
        
        from sklearn.metrics import accuracy_score, mean_squared_error, r2_score, classification_report
        
        evaluation_results = {}
        is_classification = self.y_val.nunique() < 20 and self.y_val.dtype == 'object' or self.y_val.nunique() <= 10
        
        for model_name, model in self.models.items():
            try:
                # Make predictions
                y_pred = model.predict(self.X_val)
                
                if is_classification:
                    # Classification metrics
                    accuracy = accuracy_score(self.y_val, y_pred)
                    evaluation_results[model_name] = {
                        'accuracy': accuracy,
                        'type': 'classification'
                    }
                    
                    # Classification report
                    report = classification_report(self.y_val, y_pred, output_dict=True)
                    evaluation_results[model_name]['classification_report'] = report
                    
                else:
                    # Regression metrics
                    mse = mean_squared_error(self.y_val, y_pred)
                    r2 = r2_score(self.y_val, y_pred)
                    evaluation_results[model_name] = {
                        'mse': mse,
                        'rmse': np.sqrt(mse),
                        'r2_score': r2,
                        'type': 'regression'
                    }
                
                logger.info(f"Evaluated {model_name}")
                
            except Exception as e:
                logger.error(f"Failed to evaluate {model_name}: {str(e)}")
                evaluation_results[model_name] = {'error': str(e)}
        
        self.results = evaluation_results
        
        # Save evaluation results
        results_dir = os.path.join(self.agent.workspace_dir, "results/reports")
        import json
        with open(os.path.join(results_dir, "evaluation_results.json"), 'w') as f:
            json.dump(evaluation_results, f, indent=2, default=str)
        
        # Find best model
        best_model = None
        best_score = float('-inf')
        metric_name = 'accuracy' if is_classification else 'r2_score'
        
        for model_name, results in evaluation_results.items():
            if metric_name in results:
                score = results[metric_name]
                if score > best_score:
                    best_score = score
                    best_model = model_name
        
        # Update metadata
        metadata = {
            "evaluation_results": evaluation_results,
            "best_model": best_model,
            "best_score": best_score,
            "metric_used": metric_name,
            "problem_type": "classification" if is_classification else "regression"
        }
        
        self.agent.get_current_step().metadata.update(metadata)
        logger.info(f"Model evaluation completed. Best model: {best_model} ({metric_name}={best_score:.4f})")
        return True
    
    def _execute_hyperparameter_tuning(self, **kwargs) -> bool:
        """Execute hyperparameter tuning step."""
        logger.info("Executing hyperparameter tuning step...")
        
        # For now, implement a simple grid search for the best model
        # This is a placeholder for more sophisticated tuning
        
        metadata = {
            "tuning_method": "placeholder",
            "status": "implemented_basic_version",
            "note": "Advanced hyperparameter tuning to be implemented"
        }
        
        self.agent.get_current_step().metadata.update(metadata)
        logger.info("Hyperparameter tuning step completed (basic implementation)")
        return True
    
    def _execute_model_deployment(self, **kwargs) -> bool:
        """Execute model deployment step."""
        logger.info("Executing model deployment step...")
        
        if not self.models:
            logger.error("No trained models available for deployment")
            return False
        
        # Save models to disk
        import joblib
        models_dir = os.path.join(self.agent.workspace_dir, "models/trained")
        
        deployed_models = []
        for model_name, model in self.models.items():
            try:
                model_path = os.path.join(models_dir, f"{model_name}.pkl")
                joblib.dump(model, model_path)
                deployed_models.append(model_name)
                logger.info(f"Deployed {model_name} to {model_path}")
            except Exception as e:
                logger.error(f"Failed to deploy {model_name}: {str(e)}")
        
        # Save scalers and encoders if they exist
        if self.scalers:
            scalers_path = os.path.join(models_dir, "scalers.pkl")
            joblib.dump(self.scalers, scalers_path)
        
        if self.encoders:
            encoders_path = os.path.join(models_dir, "encoders.pkl")
            joblib.dump(self.encoders, encoders_path)
        
        # Update metadata
        metadata = {
            "deployed_models": deployed_models,
            "models_directory": models_dir,
            "scalers_saved": bool(self.scalers),
            "encoders_saved": bool(self.encoders)
        }
        
        self.agent.get_current_step().metadata.update(metadata)
        logger.info(f"Model deployment completed. Deployed {len(deployed_models)} models")
        return True
    
    def _execute_monitoring(self, **kwargs) -> bool:
        """Execute monitoring step."""
        logger.info("Executing monitoring step...")
        
        # This is a placeholder for production monitoring
        # In a real implementation, this would set up monitoring infrastructure
        
        metadata = {
            "monitoring_type": "placeholder",
            "status": "setup_complete",
            "note": "Production monitoring infrastructure to be implemented"
        }
        
        self.agent.get_current_step().metadata.update(metadata)
        logger.info("Monitoring step completed (placeholder implementation)")
        return True
    
    def run_complete_workflow(self, **kwargs) -> bool:
        """
        Run the complete workflow from start to finish.
        
        Args:
            **kwargs: Parameters for workflow execution
            
        Returns:
            True if workflow completed successfully, False otherwise
        """
        logger.info("Starting complete ML workflow execution...")
        
        while self.agent.get_current_step() is not None:
            current_step = self.agent.get_current_step()
            logger.info(f"Executing step: {current_step.name}")
            
            success = self.execute_current_step(**kwargs)
            if not success:
                logger.error(f"Workflow failed at step: {current_step.name}")
                return False
            
            # Advance to next step
            if not self.agent.advance_to_next_step():
                break
        
        logger.info("Complete ML workflow execution finished successfully!")
        return True
    
    def get_workflow_report(self) -> Dict[str, Any]:
        """Generate a comprehensive workflow report."""
        completed, total, progress = self.agent.get_progress()
        
        report = {
            "project_name": self.agent.project_name,
            "workflow_progress": {
                "completed_steps": completed,
                "total_steps": total,
                "progress_percentage": progress
            },
            "data_summary": {},
            "model_performance": {},
            "recommendations": self.agent.get_recommendations()
        }
        
        # Add data summary if available
        if self.data is not None:
            report["data_summary"] = {
                "shape": self.data.shape,
                "features": list(self.data.columns),
                "target_column": self.target_column,
                "missing_values": self.data.isnull().sum().to_dict()
            }
        
        # Add model performance if available
        if self.results:
            report["model_performance"] = self.results
        
        # Add step details
        report["step_details"] = []
        for step in self.agent.steps:
            step_info = {
                "name": step.name,
                "status": step.status.value,
                "progress": step.progress,
                "metadata": step.metadata
            }
            report["step_details"].append(step_info)
        
        return report
