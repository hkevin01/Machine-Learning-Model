#!/usr/bin/env python3
"""Test the enhanced algorithm results functionality."""

import os
import sys

# Add src to path for testing
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'src'))

try:
    from machine_learning_model.gui.models.algorithm_runner import run_algorithm
    from machine_learning_model.gui.models.data_synthesizer import SyntheticDataSpec
    print("✅ Successfully imported enhanced algorithm modules")
except Exception as e:
    print(f"❌ Import failed: {e}")
    sys.exit(1)

def test_enhanced_linear_regression():
    """Test enhanced linear regression results."""
    print("\n=== Testing Enhanced Linear Regression ===")

    try:
        spec = SyntheticDataSpec(task="regression", n_samples=50, n_features=2)
        result = run_algorithm("Linear Regression", "regression", spec)

        print(f"Algorithm: {result.algorithm}")
        print(f"Task: {result.task}")
        print(f"Success: {result.success}")
        print(f"Execution Time: {result.execution_time:.4f}s")
        print(f"Performance Summary: {result.performance_summary}")
        print(f"Model Parameters: {result.model_info.get('parameters', {})}")
        print(f"Number of Recommendations: {len(result.recommendations)}")

        # Check that new fields exist and are populated
        assert hasattr(result, 'execution_time'), "Missing execution_time field"
        assert hasattr(result, 'model_info'), "Missing model_info field"
        assert hasattr(result, 'performance_summary'), "Missing performance_summary field"
        assert hasattr(result, 'recommendations'), "Missing recommendations field"

        assert result.execution_time > 0, "Execution time should be positive"
        assert len(result.model_info) > 0, "Model info should be populated"
        assert len(result.performance_summary) > 0, "Performance summary should be populated"
        assert len(result.recommendations) > 0, "Should have recommendations"

        print("✅ Enhanced Linear Regression test passed!")
        return True

    except Exception as e:
        print(f"❌ Enhanced Linear Regression test failed: {e}")
        return False

def main():
    """Run all enhanced algorithm tests."""
    print("🚀 Testing Enhanced Algorithm Results")

    success = test_enhanced_linear_regression()

    if success:
        print("\n🎉 All enhanced algorithm tests passed!")
        print("Algorithm output enhancements are working correctly!")
    else:
        print("\n💥 Some tests failed!")
        sys.exit(1)

if __name__ == "__main__":
    main()
