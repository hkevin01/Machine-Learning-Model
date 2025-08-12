#!/usr/bin/env python3
"""Test script to verify PyQt6 GUI works correctly without full Docker setup."""

import sys
import os

# Add the src directory to the Python path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

def test_pyqt6_imports():
    """Test that PyQt6 components can be imported successfully."""
    try:
        print("Testing PyQt6 imports...")
        from PyQt6.QtWidgets import QApplication, QMainWindow
        from PyQt6.QtCore import Qt
        print("✅ PyQt6 core imports successful")
        
        print("Testing PyQt6 GUI implementation import...")
        from machine_learning_model.gui.main_window_pyqt6 import MLExplorerMainWindow
        print("✅ PyQt6 GUI implementation imports successfully")
        
        print("Testing algorithm execution...")
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
        print(f"❌ Import/execution failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_gui_creation():
    """Test GUI creation without showing the window."""
    try:
        print("Testing GUI window creation...")
        from PyQt6.QtWidgets import QApplication
        from machine_learning_model.gui.main_window_pyqt6 import MLExplorerMainWindow
        
        # Create QApplication (required for any Qt GUI)
        app = QApplication(sys.argv if 'app' not in locals() else [])
        
        # Create main window (but don't show it)
        window = MLExplorerMainWindow()
        print("✅ GUI window created successfully")
        
        # Clean up
        window.close()
        app.quit()
        
        return True
        
    except Exception as e:
        print(f"❌ GUI creation failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    print("🧪 Testing PyQt6 GUI system...")
    
    # Test imports first
    imports_ok = test_pyqt6_imports()
    if not imports_ok:
        print("💥 Import tests failed!")
        sys.exit(1)
    
    # Test GUI creation
    gui_ok = test_gui_creation()
    if not gui_ok:
        print("💥 GUI creation tests failed!")
        sys.exit(1)
    
    print("🎉 All tests passed! PyQt6 GUI should work correctly.")
    print("💡 To run the full GUI: python run_gui.py")
    sys.exit(0)
