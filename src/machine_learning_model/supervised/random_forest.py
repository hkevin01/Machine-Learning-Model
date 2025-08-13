"""Random Forest implementation for classification and regression."""

from concurrent.futures import ThreadPoolExecutor
from typing import Optional, Union

import numpy as np
import pandas as pd

from .decision_tree import DecisionTreeClassifier, DecisionTreeRegressor

try:  # optional tracking
    from ..tracking import mlflow_tracking as _tracking  # type: ignore
except Exception:  # pragma: no cover
    _tracking = None

try:  # structured logging
    from ..logging_utils import get_logger  # type: ignore
    _logger = get_logger(__name__)
except Exception:  # pragma: no cover
    import logging as _stdlib_logging
    _logger = _stdlib_logging.getLogger(__name__)


class BaseRandomForest:
    """Base class for Random Forest algorithms."""

    def __init__(self, n_estimators: int = 100, max_depth: Optional[int] = None,
                 min_samples_split: int = 2, min_samples_leaf: int = 1,
                 max_features: Union[str, int, float] = 'sqrt', bootstrap: bool = True,
                 oob_score: bool = False, random_state: Optional[int] = None,
                 n_jobs: Optional[int] = None, verbose: int = 0):

        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.max_features = max_features
        self.bootstrap = bootstrap
        self.oob_score = oob_score
        self.random_state = random_state
        self.n_jobs = n_jobs if n_jobs is not None else 1
        self.verbose = verbose

        # Initialize attributes
        self.estimators_ = []
        self.feature_importances_ = None
        self.oob_score_ = None
        self.oob_prediction_ = None
        self.n_features_ = None
        self.n_outputs_ = None

        # Set random state
        if random_state is not None:
            np.random.seed(random_state)

    def _get_max_features(self, n_features: int) -> int:
        """Calculate the number of features to consider for each split."""
        if isinstance(self.max_features, str):
            if self.max_features == 'sqrt':
                return int(np.sqrt(n_features))
            elif self.max_features == 'log2':
                return int(np.log2(n_features))
            elif self.max_features == 'auto':
                return int(np.sqrt(n_features))
            else:
                raise ValueError(f"Unknown max_features: {self.max_features}")
        elif isinstance(self.max_features, int):
            return min(self.max_features, n_features)
        elif isinstance(self.max_features, float):
            return int(self.max_features * n_features)
        else:
            raise ValueError(f"Invalid max_features type: {type(self.max_features)}")

    def _bootstrap_sample(self, X: np.ndarray, y: np.ndarray) -> tuple:
        """Create a bootstrap sample of the data."""
        n_samples = X.shape[0]

        if self.bootstrap:
            # Sample with replacement
            indices = np.random.choice(n_samples, size=n_samples, replace=True)
        else:
            # Use all samples without replacement
            indices = np.arange(n_samples)
            np.random.shuffle(indices)

        return X[indices], y[indices], indices

    def _calculate_oob_score(self, X: np.ndarray, y: np.ndarray):
        """Calculate out-of-bag score."""
        n_samples = X.shape[0]
        oob_prediction = np.zeros((n_samples,) + y.shape[1:] if y.ndim > 1 else (n_samples,))
        oob_count = np.zeros(n_samples)

        for estimator, bootstrap_indices in zip(self.estimators_, self.bootstrap_indices_):
            # Find out-of-bag samples
            oob_indices = np.setdiff1d(np.arange(n_samples), bootstrap_indices)

            if len(oob_indices) > 0:
                oob_pred = estimator.predict(X[oob_indices])
                oob_prediction[oob_indices] += oob_pred
                oob_count[oob_indices] += 1

        # Avoid division by zero
        valid_oob = oob_count > 0
        oob_prediction[valid_oob] /= oob_count[valid_oob][:, np.newaxis] if y.ndim > 1 else oob_count[valid_oob]

        self.oob_prediction_ = oob_prediction

        # Calculate OOB score
        if np.sum(valid_oob) > 0:
            self.oob_score_ = self._calculate_oob_metric(y[valid_oob], oob_prediction[valid_oob])
        else:
            self.oob_score_ = 0.0

    def _calculate_feature_importances(self):
        """Calculate feature importances by averaging across all trees."""
        if not self.estimators_:
            return

        importances = np.zeros(self.n_features_)

        for estimator in self.estimators_:
            if hasattr(estimator, 'feature_importances_') and estimator.feature_importances_ is not None:
                importances += estimator.feature_importances_

        # Average and normalize
        importances /= len(self.estimators_)

        # Ensure importances sum to 1
        if np.sum(importances) > 0:
            importances /= np.sum(importances)

        self.feature_importances_ = importances

    def _fit_single_tree(self, args):
        """Fit a single tree (for parallel processing)."""
        X, y, tree_idx, random_state = args

        # Set random state for this tree
        if random_state is not None:
            np.random.seed(random_state + tree_idx)

        # Create bootstrap sample
        X_bootstrap, y_bootstrap, bootstrap_indices = self._bootstrap_sample(X, y)

        # Create and fit tree
        tree = self._create_tree()
        tree.fit(X_bootstrap, y_bootstrap)

        return tree, bootstrap_indices

    def fit(self, X: Union[np.ndarray, pd.DataFrame], y: Union[np.ndarray, pd.Series]):
        """Train the Random Forest."""
        # Convert inputs
        if isinstance(X, pd.DataFrame):
            X = X.values
        if isinstance(y, pd.Series):
            y = y.values

        self.n_features_ = X.shape[1]
        self.n_outputs_ = 1 if y.ndim == 1 else y.shape[1]

        # Prepare arguments for parallel processing
        if self.random_state is not None:
            random_states = [self.random_state + i for i in range(self.n_estimators)]
        else:
            random_states = [None] * self.n_estimators

        args_list = [(X, y, i, random_states[i]) for i in range(self.n_estimators)]

        run_ctx = None
        if _tracking and _tracking.tracking_enabled():
            run_ctx = _tracking.start_run()

        if run_ctx:
            with run_ctx:
                _tracking.log_params({
                    "n_estimators": self.n_estimators,
                    "max_depth": self.max_depth,
                    "bootstrap": self.bootstrap,
                })
                _logger.info(f"training_start model=RandomForest n_estimators={self.n_estimators}")
                if self.n_jobs == 1:
                    results = [self._fit_single_tree(a) for a in args_list]
                else:
                    with ThreadPoolExecutor(max_workers=self.n_jobs) as executor:
                        results = list(executor.map(self._fit_single_tree, args_list))
        else:
            if self.n_jobs == 1:
                results = [self._fit_single_tree(a) for a in args_list]
            else:
                with ThreadPoolExecutor(max_workers=self.n_jobs) as executor:
                    results = list(executor.map(self._fit_single_tree, args_list))

        # Extract trees and bootstrap indices
        self.estimators_ = [result[0] for result in results]
        self.bootstrap_indices_ = [result[1] for result in results]

        # Calculate feature importances
        self._calculate_feature_importances()
        if _tracking and _tracking.tracking_enabled() and self.feature_importances_ is not None:
            _tracking.log_feature_importances(self.feature_importances_)

        # Calculate OOB score if requested
        if self.oob_score:
            self._calculate_oob_score(X, y)

        if self.verbose > 0:
            _logger.info(f"training_complete model=RandomForest trees={len(self.estimators_)}")
        # Log metrics if tracking enabled
        if _tracking and _tracking.tracking_enabled():
            try:
                preds = self.predict(X)
                if isinstance(self, RandomForestClassifier):
                    acc = float(np.mean(preds == y))
                    _tracking.log_metrics({"train_accuracy": acc})
                    _logger.info(f"training_metrics accuracy={acc:.4f}")
                else:  # Regressor
                    ss_res = float(np.sum((y - preds) ** 2))
                    ss_tot = float(np.sum((y - np.mean(y)) ** 2))
                    r2 = 1 - (ss_res / ss_tot) if ss_tot != 0 else 0.0
                    _tracking.log_metrics({"train_r2": r2})
                    _logger.info(f"training_metrics r2={r2:.4f}")
            except Exception as e:  # pragma: no cover
                _logger.warning(f"metric_logging_failed error={e}")

        return self

    def predict(self, X: Union[np.ndarray, pd.DataFrame]) -> np.ndarray:
        """Make predictions using the ensemble."""
        if isinstance(X, pd.DataFrame):
            X = X.values

        if not self.estimators_:
            raise ValueError("Forest not fitted yet")

        # Get predictions from all trees
        predictions = np.array([tree.predict(X) for tree in self.estimators_])

        # Aggregate predictions
        return self._aggregate_predictions(predictions)


class RandomForestClassifier(BaseRandomForest):
    """Random Forest Classifier implementation."""

    def __init__(self, criterion: str = 'gini', **kwargs):
        super().__init__(**kwargs)
        self.criterion = criterion
        self.classes_ = None
        self.n_classes_ = None

    def _create_tree(self):
        """Create a single decision tree."""
        max_features = self._get_max_features(self.n_features_)

        return DecisionTreeClassifier(
            criterion=self.criterion,
            max_depth=self.max_depth,
            min_samples_split=self.min_samples_split,
            min_samples_leaf=self.min_samples_leaf,
            max_features=max_features
        )

    def _aggregate_predictions(self, predictions: np.ndarray) -> np.ndarray:
        """Aggregate predictions using majority voting."""
        # predictions shape: (n_estimators, n_samples)
        n_samples = predictions.shape[1]
        # Use class dtype (supports string/object labels)
        if hasattr(self, 'classes_') and self.classes_ is not None:
            final_predictions = np.empty(n_samples, dtype=self.classes_.dtype)
        else:
            final_predictions = np.empty(n_samples, dtype=predictions.dtype)

        for i in range(n_samples):
            # Count votes for each class
            sample_predictions = predictions[:, i]
            unique_classes, counts = np.unique(sample_predictions, return_counts=True)

            # Return class with most votes
            final_predictions[i] = unique_classes[np.argmax(counts)]

        return final_predictions

    def _calculate_oob_metric(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """Calculate accuracy for OOB score."""
        return np.mean(y_true == y_pred)

    def fit(self, X: Union[np.ndarray, pd.DataFrame], y: Union[np.ndarray, pd.Series]):
        """Train the Random Forest Classifier."""
        if isinstance(y, pd.Series):
            y = y.values

        self.classes_ = np.unique(y)
        self.n_classes_ = len(self.classes_)

        return super().fit(X, y)

    def predict_proba(self, X: Union[np.ndarray, pd.DataFrame]) -> np.ndarray:
        """Predict class probabilities."""
        if isinstance(X, pd.DataFrame):
            X = X.values

        if not self.estimators_:
            raise ValueError("Forest not fitted yet")

        n_samples = X.shape[0]
        probabilities = np.zeros((n_samples, self.n_classes_))

        # Get predictions from all trees
        for tree in self.estimators_:
            tree_predictions = tree.predict(X)

            # Convert to probabilities (simple voting)
            for i, pred in enumerate(tree_predictions):
                class_idx = np.where(self.classes_ == pred)[0][0]
                probabilities[i, class_idx] += 1

        # Normalize to get probabilities
        probabilities /= len(self.estimators_)

        return probabilities


class RandomForestRegressor(BaseRandomForest):
    """Random Forest Regressor implementation."""

    def __init__(self, criterion: str = 'mse', **kwargs):
        super().__init__(**kwargs)
        self.criterion = criterion

    def _create_tree(self):
        """Create a single decision tree."""
        max_features = self._get_max_features(self.n_features_)

        return DecisionTreeRegressor(
            criterion=self.criterion,
            max_depth=self.max_depth,
            min_samples_split=self.min_samples_split,
            min_samples_leaf=self.min_samples_leaf,
            max_features=max_features
        )

    def _aggregate_predictions(self, predictions: np.ndarray) -> np.ndarray:
        """Aggregate predictions using averaging."""
        return np.mean(predictions, axis=0)

    def _calculate_oob_metric(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """Calculate R² score for OOB evaluation."""
        ss_res = np.sum((y_true - y_pred) ** 2)
        ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    return 1 - (ss_res / ss_tot) if ss_tot != 0 else 0.0

