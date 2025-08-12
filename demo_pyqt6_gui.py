#!/usr/bin/env python3
"""
Final demonstration script for the new PyQt6 GUI.
This script launches the GUI with informational output.
"""

import sys
import os

# Add the src directory to the Python path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))


def main():
    """Launch the PyQt6 GUI with demonstration info."""
    print("🚀 Machine Learning Framework Explorer - PyQt6 Edition")
    print("=" * 60)
    print()
    print("🎯 New Features:")
    print("   ✅ Modern PyQt6 interface with professional styling")
    print("   ✅ Categorized algorithm display (Supervised/Unsupervised/Semi-Supervised)")
    print("   ✅ Detailed algorithm information with examples and use cases")
    print("   ✅ Quick-run panel for testing algorithms with synthetic data")
    print("   ✅ Status indicators showing implementation progress")
    print("   ✅ Rich text formatting for better readability")
    print()
    print("📊 Algorithm Categories:")
    
    try:
        from machine_learning_model.gui.main_window_pyqt6 import AlgorithmDatabase
        
        supervised = AlgorithmDatabase.SUPERVISED_ALGORITHMS
        unsupervised = AlgorithmDatabase.UNSUPERVISED_ALGORITHMS
        semi_supervised = AlgorithmDatabase.SEMI_SUPERVISED_ALGORITHMS
        
        print(f"   🎯 Supervised Learning: {len(supervised)} algorithms")
        for name, info in supervised.items():
            status_icon = "✅" if "Complete" in info['status'] else "🔄" if "Next" in info['status'] else "📋"
            print(f"      {status_icon} {name}")
        
        print(f"   🔍 Unsupervised Learning: {len(unsupervised)} algorithms")
        for name, info in unsupervised.items():
            status_icon = "✅" if "Complete" in info['status'] else "🔄" if "Next" in info['status'] else "📋"
            print(f"      {status_icon} {name}")
        
        print(f"   🎭 Semi-Supervised Learning: {len(semi_supervised)} algorithms")
        for name, info in semi_supervised.items():
            status_icon = "✅" if "Complete" in info['status'] else "🔄" if "Next" in info['status'] else "📋"
            print(f"      {status_icon} {name}")
        
        total = len(supervised) + len(unsupervised) + len(semi_supervised)
        print(f"\n📈 Total: {total} algorithms across all categories")
        
    except Exception as e:
        print(f"   ❌ Could not load algorithm database: {e}")
    
    print()
    print("🔥 Quick Run Features:")
    print("   • Test Decision Trees and Random Forest with synthetic data")
    print("   • Switch between Classification and Regression tasks")
    print("   • See real-time accuracy and performance metrics")
    print("   • Immediate feedback on algorithm behavior")
    print()
    print("🎨 Interface Improvements:")
    print("   • Modern tab-based navigation")
    print("   • Responsive layout with resizable panels")
    print("   • Rich HTML formatting for algorithm details")
    print("   • Professional styling and color scheme")
    print("   • Status indicators and progress tracking")
    print()
    print("🚀 Launching GUI...")
    print("=" * 60)
    
    try:
        from machine_learning_model.gui.main_window_pyqt6 import main as gui_main
        return gui_main()
    except Exception as e:
        print(f"❌ Failed to launch GUI: {e}")
        print("\n💡 Troubleshooting:")
        print("   • Ensure PyQt6 is installed: pip install PyQt6")
        print("   • Check requirements: pip install -r requirements.txt")
        print("   • For Docker: ./run.sh")
        print("   • For fallback: python run_gui.py (will use tkinter)")
        return 1


if __name__ == "__main__":
    sys.exit(main())
