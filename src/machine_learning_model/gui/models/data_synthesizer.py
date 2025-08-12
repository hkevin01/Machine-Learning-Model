"""Synthetic data generation utilities for Quick Run panel.

Separated as part of MVC refactor (Step 2). Keeps GUI code slim and testable.
"""
from __future__ import annotations

from dataclasses import dataclass
import numpy as np


@dataclass(slots=True)
class SyntheticDataSpec:
    task: str  # classification | regression | clustering | dimensionality_reduction
    n_samples: int = 100
    n_features: int = 5
    seed: int = 42


def generate_synthetic_data(spec: SyntheticDataSpec) -> tuple[np.ndarray, np.ndarray | None]:
    """Generate synthetic dataset for a given task.

    Returns X, y where y may be None for tasks not requiring labels.
    """
    rng = np.random.default_rng(spec.seed)
    X = rng.normal(size=(spec.n_samples, spec.n_features))

    task = spec.task.lower()
    if task == "classification":
        linear_combo = X[:, 0] + 0.5 * X[:, 1] - 0.3 * X[:, 2]
        y = (linear_combo > 0).astype(int)
        return X, y
    if task == "regression":
        noise = rng.normal(scale=0.1, size=spec.n_samples)
        y = 0.7 * X[:, 0] + 0.3 * X[:, 1] - 0.2 * X[:, 2] + noise
        return X, y
    return X, None


__all__ = ["SyntheticDataSpec", "generate_synthetic_data"]
