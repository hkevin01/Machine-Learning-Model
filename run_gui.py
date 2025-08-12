#!/usr/bin/env python3
"""
Standalone GUI launcher for Machine Learning Framework Explorer.
This script runs the GUI without import issues.
"""

import os
import sys

# Add the src directory to the Python path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

def main():
    """Launch the GUI application."""
    print("🚀 Starting Machine Learning Framework Explorer...")
    
    try:
        # Use clean implementation
        print("🔄 Loading GUI components...")
        from machine_learning_model.gui.main_window_fixed import main as gui_main
        gui_main()
    except ImportError as e:
        print(f"❌ Import error: {e}")
        print(" Please ensure all dependencies are installed:")
        print("   pip install -r requirements.txt")
        sys.exit(1)
    except Exception as e:
        print(f"❌ Error launching GUI: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()