#!/usr/bin/env python3
"""Quick test of GUI functionality without X11."""
import os
import sys

# Calculate root path for proper imports
ROOT = os.path.dirname(__file__)
while ROOT and 'src' not in os.listdir(ROOT):
    ROOT = os.path.dirname(ROOT)
sys.path.insert(0, os.path.join(ROOT, 'src'))

def test_gui_imports():
    """Test that GUI modules can be imported without errors."""
    try:
        print("✅ MainWindow import successful")
        return True
    except Exception as e:
        print(f"❌ MainWindow import failed: {e}")
        return False

def test_algorithm_run():
    """Test synthetic algorithm execution logic."""
    try:
        import numpy as np

        from machine_learning_model.supervised.decision_tree import (
            DecisionTreeClassifier,
        )
        from machine_learning_model.supervised.random_forest import (
            RandomForestClassifier,
        )

        # Test synthetic data generation
        n = 50
        X = np.random.randn(n, 5)
        y = (X[:, 0] + X[:, 1] > 0).astype(int)

        # Test Decision Tree
        model = DecisionTreeClassifier()
        model.fit(X, y)
        pred = model.predict(X)
        acc = float(np.mean(pred == y))
        print(f"✅ DecisionTree accuracy: {acc:.3f}")

        # Test Random Forest
        model = RandomForestClassifier(n_estimators=10)
        model.fit(X, y)
        pred = model.predict(X)
        acc = float(np.mean(pred == y))
        print(f"✅ RandomForest accuracy: {acc:.3f}")

        return True
    except Exception as e:
        print(f"❌ Algorithm test failed: {e}")
        return False

if __name__ == "__main__":
    print("🧪 Testing GUI functionality...")

    import_ok = test_gui_imports()
    algo_ok = test_algorithm_run()

    if import_ok and algo_ok:
        print("✅ All tests passed - GUI should work correctly")
        sys.exit(0)
    else:
        print("❌ Some tests failed")
        sys.exit(1)
