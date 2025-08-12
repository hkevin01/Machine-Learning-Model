#!/usr/bin/env python3
"""
PyQt6-only GUI launcher for Machine Learning Framework Explorer.
This script requires PyQt6 and will not fall back to tkinter.
"""

import os
import sys

# Add the src directory to the Python path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))


def main():
    """Launch the PyQt6 GUI application."""
    print("🚀 Starting Machine Learning Framework Explorer (PyQt6)...")
    
    try:
        # Import PyQt6 components
        print("🔄 Loading PyQt6 GUI components...")
        from machine_learning_model.gui.main_window_pyqt6 import main as gui_main
        
        print("✅ PyQt6 successfully loaded")
        print("🎯 Launching PyQt6 GUI...")
        
        return gui_main()
        
    except ImportError as e:
        print(f"❌ PyQt6 not available: {e}")
        print("📦 Please install PyQt6:")
        print("   pip install PyQt6")
        print("   Or: pip install -r requirements.txt")
        sys.exit(1)
    except Exception as e:
        print(f"❌ Error launching PyQt6 GUI: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    sys.exit(main())
