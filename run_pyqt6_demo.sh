#!/bin/bash

# Make script executable
chmod +x run_gui.py
chmod +x run_gui_pyqt6.py  
chmod +x demo_pyqt6_gui.py

echo "🚀 Machine Learning Framework Explorer - PyQt6 GUI"
echo "=================================================="
echo ""
echo "Choose how to run the GUI:"
echo ""
echo "1) PyQt6 GUI (Recommended - Modern Interface)"
echo "   python demo_pyqt6_gui.py"
echo ""
echo "2) PyQt6 GUI (Direct)"  
echo "   python run_gui_pyqt6.py"
echo ""
echo "3) Auto-detect GUI (PyQt6 → tkinter fallback)"
echo "   python run_gui.py"
echo ""
echo "4) Docker Container"
echo "   ./run.sh"
echo ""
echo "🎯 The new PyQt6 GUI features:"
echo "   • Categorized algorithm display (Supervised/Unsupervised/Semi-Supervised)"
echo "   • Rich algorithm details with examples and use cases"  
echo "   • Quick-run panel for testing with synthetic data"
echo "   • Modern tabbed interface with professional styling"
echo "   • 15 algorithms across 3 categories"
echo ""
echo "💡 If PyQt6 is not installed:"
echo "   pip install PyQt6"
echo "   # OR"
echo "   pip install -r requirements.txt"
echo ""

# Try to run the demo automatically
echo "🔥 Launching PyQt6 GUI demo..."
python demo_pyqt6_gui.py
