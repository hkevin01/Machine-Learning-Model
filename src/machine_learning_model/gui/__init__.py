"""
GUI module for Machine Learning Framework Explorer.
"""

__version__ = "1.0.0"

try:
    # Try PyQt6 implementation first (modern interface)
    from .main_window_pyqt6 import MLExplorerMainWindow as MainWindow
    GUI_TYPE = "PyQt6"
except ImportError:
    try:
        # Fallback to clean tkinter implementation
        from .main_window_fixed import MainWindow
        GUI_TYPE = "tkinter"
    except ImportError:
        # Last resort - original implementation (may be corrupted)
        from .main_window import MainWindow
        GUI_TYPE = "tkinter-original"

__all__ = ["MainWindow", "GUI_TYPE"]
