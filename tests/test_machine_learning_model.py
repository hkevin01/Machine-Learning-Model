"""Tests for machine_learning_model"""

import pytest
from src.machine_learning_model import main


def test_main():
    """Test main function."""
    # Add your tests here
    assert main is not None


def test_version():
    """Test version import."""
    from src.machine_learning_model import __version__
    assert __version__ is not None
