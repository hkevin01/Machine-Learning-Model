"""
Pytest configuration and fixtures.
"""

import pytest
from pathlib import Path
import tempfile
import shutil


@pytest.fixture
def temp_dir():
    """Create a temporary directory for tests."""
    temp_path = Path(tempfile.mkdtemp())
    yield temp_path
    shutil.rmtree(temp_path)


@pytest.fixture
def sample_data():
    """Provide sample data for tests."""
    return {
        "name": "test",
        "value": 42,
        "items": ["a", "b", "c"]
    }


@pytest.fixture(scope="session")
def test_config():
    """Test configuration."""
    return {
        "test_mode": True,
        "debug": False
    }


class MockResponse:
    """Mock HTTP response for testing."""
    
    def __init__(self, json_data, status_code):
        self.json_data = json_data
        self.status_code = status_code
    
    def json(self):
        return self.json_data


@pytest.fixture
def mock_requests_get(monkeypatch):
    """Mock requests.get for testing."""
    def mock_get(*args, **kwargs):
        return MockResponse({"key": "value"}, 200)
    
    monkeypatch.setattr("requests.get", mock_get)
