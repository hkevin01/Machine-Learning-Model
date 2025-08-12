#!/usr/bin/env python3
"""Test script to verify GUI imports work correctly."""

import sys
import os

# Add the src directory to the Python path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

def test_gui_imports():
    """Test that GUI components can be imported successfully."""
    try:
        print("Testing clean implementation import...")
        from machine_learning_model.gui.main_window_fixed import MainWindow, main
        print("✅ Clean implementation imports successfully")
        
        print("Testing __init__.py import...")
        from machine_learning_model.gui import MainWindow as MainWindowFromInit
        print("✅ __init__.py imports successfully")
        
        print("Testing synthetic algorithm execution...")
        import numpy as np
        from machine_learning_model.supervised.decision_tree import DecisionTreeClassifier
        
        # Quick synthetic test
        X = np.random.randn(50, 3)
        y = (X[:, 0] + X[:, 1] > 0).astype(int)
        model = DecisionTreeClassifier()
        model.fit(X, y)
        pred = model.predict(X)
        acc = float(np.mean(pred == y))
        print(f"✅ Algorithm execution test passed - Accuracy: {acc:.3f}")
        
        return True
        
    except Exception as e:
        print(f"❌ Import failed: {e}")
        return False

if __name__ == "__main__":
    print("🧪 Testing GUI system...")
    success = test_gui_imports()
    if success:
        print("🎉 All tests passed! GUI should launch successfully.")
        sys.exit(0)
    else:
        print("💥 Tests failed!")
        sys.exit(1)
