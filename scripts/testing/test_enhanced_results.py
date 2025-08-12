#!/usr/bin/env python3
"""Quick test of enhanced algorithm results."""

import os
import sys

# Add the src directory to the path - adjust for moved location (scripts/testing -> root)
root_dir = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
sys.path.insert(0, os.path.join(root_dir, 'src'))

from machine_learning_model.gui.models.algorithm_runner import run_algorithm
from machine_learning_model.gui.models.data_synthesizer import SyntheticDataSpec


def test_enhanced_results():
    """Test the enhanced algorithm results."""
    print("Testing enhanced algorithm results...")

    # Test linear regression
    spec = SyntheticDataSpec(task="regression", n_samples=100, n_features=3)
    result = run_algorithm("Linear Regression", "regression", spec)

    print(f"\n=== {result.algorithm} Results ===")
    print(f"Task: {result.task}")
    print(f"Success: {result.success}")
    print(f"Execution Time: {result.execution_time:.4f}s")
    print(f"Details:\n{result.details}")
    print(f"Metrics: {result.metrics}")
    print(f"Model Info: {result.model_info}")
    print(f"Performance Summary: {result.performance_summary}")
    print("Recommendations:")
    for i, rec in enumerate(result.recommendations, 1):
        print(f"  {i}. {rec}")
    print()


if __name__ == "__main__":
    test_enhanced_results()
