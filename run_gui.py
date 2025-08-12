#!/usr/bin/env python3
"""
Standalone GUI launcher for Machine Learning Framework Explorer.
This script runs the GUI with PyQt6 by default, falling back to tkinter if needed.
"""

import os
import sys

# Add the src directory to the Python path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))


def main():
    """Launch the GUI application."""
    print("🚀 Starting Machine Learning Framework Explorer...")
    
    try:
        # Try PyQt6 implementation first (modern interface)
        print("🔄 Loading PyQt6 GUI components...")
        from machine_learning_model.gui.main_window_pyqt6 import main as gui_main
        gui_main()
    except ImportError as e:
        print(f"❌ PyQt6 not available: {e}")
        try:
            # Fallback to tkinter implementation
            print("🔄 Falling back to tkinter implementation...")
            from machine_learning_model.gui.main_window_fixed import main as gui_main
            gui_main()
        except ImportError as e2:
            print(f"❌ Tkinter fallback failed: {e2}")
            print("📦 Please ensure GUI dependencies are installed:")
            print("   pip install PyQt6")
            print("   Or: pip install -r requirements.txt")
            sys.exit(1)
    except Exception as e:
        print(f"❌ Error launching GUI: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()