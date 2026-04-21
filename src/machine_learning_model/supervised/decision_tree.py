"""
Module: supervised.decision_tree
Purpose: Pure-Python implementation of the CART decision tree algorithm
         supporting both classification (Gini/Entropy impurity) and
         regression (MSE/MAE impurity). Provides a scikit-learn–compatible
         fit/predict interface and integrates with MLflow tracking.
Rationale: A first-principles implementation makes the learning mechanics
           transparent for educational use while remaining functional for
           real datasets.
Constraints: Not optimised for very large datasets (>100 k rows) — use
             scikit-learn's Cython-accelerated tree for production at scale.
Failure Modes: RecursionError for pathologically deep trees on datasets
               with zero variance; guarded by max_depth stopping criterion.
"""

from abc import ABC, abstractmethod
from typing import Any, Dict, Optional, Union

import numpy as np
import pandas as pd

try:  # Optional tracking
    from ..tracking import mlflow_tracking as _tracking  # type: ignore
except Exception:  # pragma: no cover
    _tracking = None

try:  # Structured logging
    from ..logging_utils import get_logger  # type: ignore
    _logger = get_logger(__name__)
except Exception:  # pragma: no cover
    import logging as _stdlib_logging
    _logger = _stdlib_logging.getLogger(__name__)


class Node:
    """
    Purpose:  Single node in the binary decision tree.
              Internal nodes store a split rule (feature + threshold);
              leaf nodes store a scalar prediction value.
    """

    def __init__(self, feature: Optional[int] = None, threshold: Optional[float] = None,
                 left=None, right=None, value: Optional[float] = None):
        self.feature = feature      # Feature index to split on
        self.threshold = threshold  # Threshold value for split
        self.left = left           # Left child node
        self.right = right         # Right child node
        self.value = value         # Prediction value (for leaf nodes)

    def is_leaf(self) -> bool:
        """Check if node is a leaf node."""
        return self.value is not None


class BaseDecisionTree(ABC):
    """
    Purpose:    Abstract base implementing the shared CART tree-building
                algorithm. Subclasses provide task-specific impurity
                measures and leaf-value calculations.
    Inputs:     max_depth          — maximum tree depth; guards against
                                     overfitting and stack overflow.
                min_samples_split  — minimum samples required to split.
                min_samples_leaf   — minimum samples in a leaf node.
                max_features       — number of features to consider at each
                                     split; None means consider all features.
                                     Used by Random Forest to inject feature
                                     subsampling at the tree level.
                random_state       — seed for reproducibility.
    """

    def __init__(self, max_depth: int = 10, min_samples_split: int = 2,
                 min_samples_leaf: int = 1, max_features: Optional[int] = None,
                 random_state: Optional[int] = None):
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.max_features = max_features  # None = use all features
        self.random_state = random_state
        self.root = None
        self.feature_importances_ = None

        if random_state is not None:
            np.random.seed(random_state)

    @abstractmethod
    def _calculate_impurity(self, y: np.ndarray) -> float:
        """Calculate impurity of a node."""
        pass

    @abstractmethod
    def _calculate_leaf_value(self, y: np.ndarray) -> float:
        """Calculate the prediction value for a leaf node."""
        pass

    def _information_gain(self, y: np.ndarray, left_indices: np.ndarray, right_indices: np.ndarray) -> float:
        """
        Purpose:  Compute the reduction in weighted impurity achieved by a
                  candidate binary split — the core CART optimisation signal.
        Inputs:   y             — full label array for the current node.
                  left_indices  — row indices going to the left child.
                  right_indices — row indices going to the right child.
        Returns:  Non-negative float; 0.0 when either child is empty.
        """
        n = len(y)
        n_left, n_right = len(left_indices), len(right_indices)

        if n_left == 0 or n_right == 0:
            return 0

        # Calculate weighted impurity after split
        left_impurity = self._calculate_impurity(y[left_indices])
        right_impurity = self._calculate_impurity(y[right_indices])

        weighted_impurity = (n_left / n) * left_impurity + (n_right / n) * right_impurity

        # Information gain is reduction in impurity
        parent_impurity = self._calculate_impurity(y)
        return parent_impurity - weighted_impurity

    def _best_split(self, X: np.ndarray, y: np.ndarray) -> tuple:
        """
        Purpose:  Exhaustive greedy search over all features and threshold
                  candidates for the split that maximises information gain.
        Inputs:   X — feature matrix (n_samples × n_features).
                  y — label array (n_samples,).
        Returns:  (best_feature_idx, best_threshold, best_gain) tuple.
                  Returns (None, None, -1) when no valid split exists.
        Complexity: O(n_features × n_unique_values × n_samples).
        """
        best_gain = -1
        best_feature = None
        best_threshold = None

        n_features = X.shape[1]

        # When max_features is set (e.g. injected by Random Forest), randomly
        # select a subset of feature indices to consider at this split.
        if self.max_features is not None and self.max_features < n_features:
            feature_indices = np.random.choice(n_features, size=self.max_features, replace=False)
        else:
            feature_indices = range(n_features)

        for feature_idx in feature_indices:
            feature_values = X[:, feature_idx]
            thresholds = np.unique(feature_values)

            for threshold in thresholds:
                left_indices = np.where(feature_values <= threshold)[0]
                right_indices = np.where(feature_values > threshold)[0]

                if len(left_indices) < self.min_samples_leaf or len(right_indices) < self.min_samples_leaf:
                    continue

                gain = self._information_gain(y, left_indices, right_indices)

                if gain > best_gain:
                    best_gain = gain
                    best_feature = feature_idx
                    best_threshold = threshold

        return best_feature, best_threshold, best_gain

    def _build_tree(self, X: np.ndarray, y: np.ndarray, depth: int = 0) -> Node:
        """
        Purpose:   Recursively partition X/y into a binary tree via the
                   CART algorithm until a stopping criterion is met.
        Stopping:  depth ≥ max_depth OR n_samples < min_samples_split
                   OR all labels are identical OR no positive-gain split
                   exists.
        Returns:   Root Node of the subtree; always a leaf at base case.
        """
        n_samples, n_features = X.shape

        # Check stopping criteria (max_depth=None means grow fully)
        if (
            (self.max_depth is not None and depth >= self.max_depth)
            or n_samples < self.min_samples_split
            or len(np.unique(y)) == 1
        ):
            leaf_value = self._calculate_leaf_value(y)
            return Node(value=leaf_value)

        # Find best split
        best_feature, best_threshold, best_gain = self._best_split(X, y)

        if best_feature is None or best_gain <= 0:
            leaf_value = self._calculate_leaf_value(y)
            return Node(value=leaf_value)

        # Split data
        left_indices = X[:, best_feature] <= best_threshold
        right_indices = ~left_indices

        # Recursively build left and right subtrees
        left_child = self._build_tree(X[left_indices], y[left_indices], depth + 1)
        right_child = self._build_tree(X[right_indices], y[right_indices], depth + 1)

        return Node(
            feature=best_feature,
            threshold=best_threshold,
            left=left_child,
            right=right_child,
        )

    def fit(self, X: Union[np.ndarray, pd.DataFrame], y: Union[np.ndarray, pd.Series]):
        """Train the decision tree.

        Logs parameters, feature importances, and basic training metrics (accuracy / R2)
        when MLflow tracking is enabled. Always emits structured log messages.
        """
        if isinstance(X, pd.DataFrame):
            X = X.values
        if isinstance(y, pd.Series):
            y = y.values
        run_ctx = None
        if _tracking and _tracking.tracking_enabled():
            run_ctx = _tracking.start_run()
        if run_ctx:
            with run_ctx:
                _tracking.log_params({
                    "max_depth": self.max_depth,
                    "min_samples_split": self.min_samples_split,
                    "min_samples_leaf": self.min_samples_leaf,
                })
                _logger.info(f"training_start model=DecisionTree max_depth={self.max_depth}")
                self.root = self._build_tree(X, y)
                self._calculate_feature_importances(X, y)
                if self.feature_importances_ is not None:
                    _tracking.log_feature_importances(self.feature_importances_)
                # Metrics: compute on training data
                try:
                    preds = self.predict(X)
                    # Determine task type heuristically
                    if y.dtype.kind in {"i", "u", "b"} or len(np.unique(y)) < max(20, len(y) * 0.5):
                        acc = float(np.mean(preds == y))
                        _tracking.log_metrics({"train_accuracy": acc})
                        _logger.info(f"training_metrics accuracy={acc:.4f}")
                    else:
                        # Regression R2
                        ss_res = float(np.sum((y - preds) ** 2))
                        ss_tot = float(np.sum((y - np.mean(y)) ** 2))
                        r2 = 1 - (ss_res / ss_tot) if ss_tot != 0 else 0.0
                        _tracking.log_metrics({"train_r2": r2})
                        _logger.info(f"training_metrics r2={r2:.4f}")
                except Exception as e:  # pragma: no cover - metrics best effort
                    _logger.warning(f"metric_logging_failed error={e}")
        else:
            self.root = self._build_tree(X, y)
            self._calculate_feature_importances(X, y)
            _logger.info(f"training_complete model=DecisionTree max_depth={self.max_depth}")
        return self

    def _predict_sample(self, x: np.ndarray) -> float:
        """Predict a single sample."""
        node = self.root

        while not node.is_leaf():
            if x[node.feature] <= node.threshold:
                node = node.left
            else:
                node = node.right

        return node.value

    def predict(self, X: Union[np.ndarray, pd.DataFrame]) -> np.ndarray:
        """
        Purpose:  Traverse the fitted tree for every sample and return
                  the leaf prediction value.
        Inputs:   X — feature matrix (n_samples × n_features); accepts
                      numpy array or pandas DataFrame.
        Returns:  1-D numpy array of predictions (n_samples,).
        Precond:  fit() must have been called first (self.root is set).
        """
        if isinstance(X, pd.DataFrame):
            X = X.values

        predictions = np.array([self._predict_sample(x) for x in X])
        return predictions

    def _calculate_feature_importances(self, X: np.ndarray, y: np.ndarray) -> None:
        """
        Purpose:    Compute a normalised importance score for each feature
                    by accumulating weighted information-gain contributions
                    across every split in the fitted tree.
        Side-effect: Sets self.feature_importances_ (ndarray, shape n_features).
                     Values sum to 1.0 after normalisation.
        Precond:    fit() / _build_tree() must have been called first.
        """
        n_features = X.shape[1]
        importances = np.zeros(n_features)

        def traverse_tree(node: Node, n_samples: int):
            if node.is_leaf():
                return

            # Calculate weighted information gain for this split
            left_samples = np.sum(X[:, node.feature] <= node.threshold)
            right_samples = n_samples - left_samples

            if left_samples > 0 and right_samples > 0:
                left_indices = X[:, node.feature] <= node.threshold
                right_indices = ~left_indices
                gain = self._information_gain(y, np.where(left_indices)[0], np.where(right_indices)[0])
                importances[node.feature] += gain * (n_samples / len(X))

            # Recursively traverse children
            if node.left:
                traverse_tree(node.left, left_samples)
            if node.right:
                traverse_tree(node.right, right_samples)

        if self.root:
            traverse_tree(self.root, len(X))
            # Normalize importances
            if np.sum(importances) > 0:
                importances = importances / np.sum(importances)

        self.feature_importances_ = importances

    def get_tree_structure(self) -> Dict[str, Any]:
        """Get tree structure for visualization."""
        def node_to_dict(node: Node) -> Dict[str, Any]:
            if node.is_leaf():
                return {"type": "leaf", "value": node.value}
            else:
                return {
                    "type": "split",
                    "feature": node.feature,
                    "threshold": node.threshold,
                    "left": node_to_dict(node.left),
                    "right": node_to_dict(node.right)
                }

        return node_to_dict(self.root) if self.root else {}


class DecisionTreeClassifier(BaseDecisionTree):
    """
    Purpose:  CART classifier using Gini impurity or information entropy.
    Inputs:   criterion — 'gini' (default) or 'entropy'.
              **kwargs  — forwarded to BaseDecisionTree.
    Usage:    clf = DecisionTreeClassifier(criterion='gini', max_depth=5)
              clf.fit(X_train, y_train)
              y_pred = clf.predict(X_test)
    """

    def __init__(self, criterion: str = 'gini', **kwargs):
        super().__init__(**kwargs)
        self.criterion = criterion
        self.classes_ = None
        self.n_classes_ = None

    def _calculate_impurity(self, y: np.ndarray) -> float:
        """Calculate impurity using Gini or Entropy."""
        if len(y) == 0:
            return 0

        _, counts = np.unique(y, return_counts=True)
        probabilities = counts / len(y)

        if self.criterion == 'gini':
            return 1 - np.sum(probabilities ** 2)
        elif self.criterion == 'entropy':
            # Avoid log(0) by adding small epsilon
            return -np.sum(probabilities * np.log2(probabilities + 1e-10))
        else:
            raise ValueError(f"Unknown criterion: {self.criterion}")

    def _calculate_leaf_value(self, y: np.ndarray) -> float:
        """Return the most common class in the leaf."""
        values, counts = np.unique(y, return_counts=True)
        return values[np.argmax(counts)]

    def fit(self, X: Union[np.ndarray, pd.DataFrame], y: Union[np.ndarray, pd.Series]):
        """Train the decision tree classifier (adds class metadata)."""
        if isinstance(y, pd.Series):
            y = y.values

        self.classes_ = np.unique(y)
        self.n_classes_ = len(self.classes_)

        return super().fit(X, y)

    def predict_proba(self, X: Union[np.ndarray, pd.DataFrame]) -> np.ndarray:
        """Predict class probabilities."""
        # For simplicity, return one-hot probabilities
        # In a full implementation, we'd store class distributions in leaf nodes
        predictions = self.predict(X)
        probabilities = np.zeros((len(predictions), self.n_classes_))

        for i, pred in enumerate(predictions):
            class_idx = np.where(self.classes_ == pred)[0][0]
            probabilities[i, class_idx] = 1.0

        return probabilities


class DecisionTreeRegressor(BaseDecisionTree):
    """
    Purpose:  CART regressor minimising MSE or MAE node impurity.
    Inputs:   criterion — 'mse' | 'squared_error' (alias) | 'mae'.
              **kwargs  — forwarded to BaseDecisionTree.
    Usage:    reg = DecisionTreeRegressor(criterion='mse', max_depth=8)
              reg.fit(X_train, y_train)
              y_pred = reg.predict(X_test)
    """

    def __init__(self, criterion: str = 'mse', **kwargs):
        super().__init__(**kwargs)
        # Normalize criterion aliases to internal names
        if criterion == 'squared_error':  # sklearn style alias
            criterion = 'mse'
        self.criterion = criterion

    def _calculate_impurity(self, y: np.ndarray) -> float:
        """Calculate impurity using MSE or MAE."""
        if len(y) == 0:
            return 0
        if self.criterion == 'mse':  # mean squared error
            mean = np.mean(y)
            return np.mean((y - mean) ** 2)
        elif self.criterion == 'mae':
            median = np.median(y)
            return np.mean(np.abs(y - median))
        else:
            raise ValueError(f"Unknown criterion: {self.criterion}")

    def _calculate_leaf_value(self, y: np.ndarray) -> float:
        """Return the mean value for regression."""
        return np.mean(y)
