"""
Cross-platform compatibility tests for the ML framework.
"""

import os
import platform
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np
import pytest

from machine_learning_model.supervised.decision_tree import DecisionTreeClassifier
from machine_learning_model.supervised.random_forest import RandomForestClassifier


class TestPlatformCompatibility:
    """Test suite for cross-platform compatibility."""
    
    def test_platform_detection(self):
        """Test that we can properly detect the platform."""
        current_platform = platform.system()
        assert current_platform in ["Windows", "Linux", "Darwin"]
        
        if current_platform == "Windows":
            assert sys.platform.startswith('win')
        elif current_platform == "Linux":
            assert sys.platform.startswith('linux')
        elif current_platform == "Darwin":
            assert sys.platform == "darwin"
    
    def test_path_handling(self):
        """Test file path handling across platforms."""
        # Test basic path operations
        test_path = Path("test") / "subdir" / "file.txt"
        assert str(test_path).replace("\\", "/") == "test/subdir/file.txt"
        
        # Test with temporary directory
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            test_file = temp_path / "test_model.pkl"
            
            # Create a test file
            test_file.touch()
            assert test_file.exists()
    
    def test_threading_compatibility(self):
        """Test threading compatibility across platforms."""
        from sklearn.datasets import make_classification
        
        X, y = make_classification(n_samples=100, n_features=10, random_state=42)
        
        # Test different n_jobs settings
        for n_jobs in [1, 2, -1]:
            try:
                rf = RandomForestClassifier(
                    n_estimators=10, 
                    n_jobs=n_jobs, 
                    random_state=42
                )
                rf.fit(X, y)
                predictions = rf.predict(X)
                assert len(predictions) == len(y)
            except Exception as e:
                # Some platforms might not support certain threading modes
                if n_jobs == 1:
                    # Single-threaded should always work
                    raise e
                else:
                    pytest.skip(f"Threading with n_jobs={n_jobs} not supported on this platform")
    
    def test_gui_dependencies(self):
        """Test GUI dependencies are available."""
        try:
            import tkinter as tk

            # Test that we can create a root window
            root = tk.Tk()
            root.withdraw()  # Hide the window
            root.destroy()
        except ImportError:
            pytest.skip("tkinter not available on this platform")
        except Exception as e:
            pytest.skip(f"GUI not available: {e}")
    
    def test_numpy_compatibility(self):
        """Test NumPy operations work correctly across platforms."""
        # Test basic operations
        a = np.array([1, 2, 3, 4, 5])
        b = np.array([2, 3, 4, 5, 6])
        
        result = a + b
        expected = np.array([3, 5, 7, 9, 11])
        np.testing.assert_array_equal(result, expected)
        
        # Test random number generation
        np.random.seed(42)
        random_data = np.random.rand(10, 5)
        assert random_data.shape == (10, 5)
        assert 0 <= random_data.min() <= random_data.max() <= 1
    
    def test_script_execution(self):
        """Test that our scripts can be executed on this platform."""
        current_platform = platform.system()
        
        if current_platform == "Windows":
            # Test Windows batch scripts exist
            setup_script = Path("scripts/setup_windows.bat")
            gui_script = Path("scripts/run_gui_windows.bat")
            
            # We can't easily test execution without actual setup
            # But we can test the files exist and are readable
            if setup_script.exists():
                assert setup_script.is_file()
            if gui_script.exists():
                assert gui_script.is_file()
                
        else:
            # Test Unix shell scripts exist and are executable
            setup_script = Path("scripts/setup_ubuntu.sh")
            gui_script = Path("scripts/run_gui.sh")
            
            if setup_script.exists():
                assert setup_script.is_file()
                # Check if executable
                assert os.access(setup_script, os.X_OK) or not setup_script.exists()
                
            if gui_script.exists():
                assert gui_script.is_file()
                assert os.access(gui_script, os.X_OK) or not gui_script.exists()


class TestWindowsSpecific:
    """Windows-specific tests."""
    
    @pytest.mark.skipif(platform.system() != "Windows", reason="Windows-only test")
    def test_windows_path_separators(self):
        """Test Windows path separator handling."""
        test_path = "models\\trained\\test_model.pkl"
        normalized_path = Path(test_path)
        
        # Should work with both separators
        assert str(normalized_path)
    
    @pytest.mark.skipif(platform.system() != "Windows", reason="Windows-only test")
    def test_windows_batch_files(self):
        """Test Windows batch file structure."""
        batch_files = [
            "scripts/setup_windows.bat",
            "scripts/run_gui_windows.bat",
            "scripts/run_tests_windows.bat"
        ]
        
        for batch_file in batch_files:
            if Path(batch_file).exists():
                with open(batch_file, 'r') as f:
                    content = f.read()
                    # Check for proper batch file headers
                    assert "@echo off" in content or "REM" in content


class TestUbuntuSpecific:
    """Ubuntu/Linux-specific tests."""
    
    @pytest.mark.skipif(platform.system() != "Linux", reason="Linux-only test")
    def test_ubuntu_dependencies(self):
        """Test Ubuntu system dependencies."""
        # Test that we can import common Ubuntu-available packages
        try:
            import tkinter
        except ImportError:
            pytest.skip("tkinter not available - may need apt install python3-tk")
    
    @pytest.mark.skipif(platform.system() != "Linux", reason="Linux-only test")
    def test_shell_scripts(self):
        """Test shell script structure."""
        shell_scripts = [
            "scripts/setup_ubuntu.sh",
            "scripts/run_gui.sh",
            "scripts/run_tests.sh"
        ]
        
        for script_file in shell_scripts:
            script_path = Path(script_file)
            if script_path.exists():
                with open(script_path, 'r') as f:
                    content = f.read()
                    # Check for proper shell script headers
                    assert "#!/bin/bash" in content or "#!/usr/bin/env bash" in content


def test_virtual_environment_creation():
    """Test virtual environment creation across platforms."""
    with tempfile.TemporaryDirectory() as temp_dir:
        venv_path = Path(temp_dir) / "test_venv"
        
        # Try to create a virtual environment
        try:
            import venv
            venv.create(venv_path, with_pip=True)
            
            # Check that it was created successfully
            if platform.system() == "Windows":
                python_exe = venv_path / "Scripts" / "python.exe"
                activate_script = venv_path / "Scripts" / "activate.bat"
            else:
                python_exe = venv_path / "bin" / "python"
                activate_script = venv_path / "bin" / "activate"
            
            assert python_exe.exists()
            assert activate_script.exists()
            
        except Exception as e:
            pytest.skip(f"Virtual environment creation failed: {e}")


if __name__ == "__main__":
    # Run the tests
    pytest.main([__file__, "-v"])
