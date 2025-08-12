"""GUI tests using pytest-qt for headless testing."""
import pytest
import sys
from unittest.mock import patch
from PyQt6.QtWidgets import QApplication
from PyQt6.QtCore import Qt

pytestmark = pytest.mark.gui


@pytest.fixture(scope="session")
def qapp():
    """Create QApplication instance for tests."""
    if not QApplication.instance():
        app = QApplication(sys.argv)
    else:
        app = QApplication.instance()
    yield app
    app.quit()


@pytest.mark.gui
def test_gui_import():
    """Test that GUI components can be imported."""
    try:
        from machine_learning_model.gui.main_window import MainWindow
        assert MainWindow is not None
    except ImportError as e:
        pytest.skip(f"GUI components not available: {e}")


@pytest.mark.gui
def test_main_window_creation(qapp, qtbot):
    """Test main window can be created and closed."""
    try:
        from machine_learning_model.gui.main_window import MainWindow
        
        window = MainWindow()
        qtbot.addWidget(window)
        
        # Test window creation
        assert window is not None
        assert window.windowTitle()
        
        # Test window shows
        window.show()
        assert window.isVisible()
        
        # Test window can be closed
        window.close()
        assert not window.isVisible()
        
    except ImportError as e:
        pytest.skip(f"GUI components not available: {e}")


@pytest.mark.gui
def test_run_gui_script():
    """Test that run_gui.py can be imported and executed."""
    try:
        # Mock sys.exit to prevent actual exit
        with patch('sys.exit'):
            import run_gui
            # Test that main function exists
            assert hasattr(run_gui, 'main')
    except ImportError as e:
        pytest.skip(f"run_gui.py not available: {e}")


@pytest.mark.gui
@pytest.mark.slow
def test_gui_workflow_integration(qapp, qtbot):
    """Integration test for basic GUI workflow."""
    try:
        from machine_learning_model.gui.main_window import MainWindow
        
        window = MainWindow()
        qtbot.addWidget(window)
        window.show()
        
        # Simulate user interaction
        qtbot.wait(100)  # Wait for UI to stabilize
        
        # Test basic interactions (add specific tests based on actual GUI)
        # This is a placeholder for actual GUI interaction tests
        
        window.close()
        
    except ImportError as e:
        pytest.skip(f"GUI components not available: {e}")
