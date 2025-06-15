#!/usr/bin/env python3
"""
Generic Python Project Structure Generator

Creates a standard Python project with:
- src/ directory structure
- .copilot/ directory for AI assistance
- .github/ directory with workflows
- .gitignore with comprehensive templates
- Virtual environment setup
- Common project files (README, requirements.txt, etc.)
- Popular development tools and logger
"""

import os
import sys
import argparse
import subprocess
import shutil
from pathlib import Path


def create_directory(path):
    """Create directory if it doesn't exist."""
    path_obj = Path(path)
    if path_obj.exists():
        print(f"Directory already exists: {path}")
    else:
        path_obj.mkdir(parents=True, exist_ok=True)
        print(f"Created directory: {path}")


def create_file(filepath, content=""):
    """Create file with optional content."""
    file_path = Path(filepath)
    if file_path.exists():
        print(f"File already exists: {filepath}")
    else:
        with open(filepath, 'w', encoding='utf-8') as f:
            f.write(content)
        print(f"Created file: {filepath}")


def run_command(command, cwd=None, check=True):
    """Run a shell command."""
    try:
        result = subprocess.run(
            command, 
            shell=True, 
            cwd=cwd, 
            capture_output=True, 
            text=True, 
            check=check
        )
        return result
    except subprocess.CalledProcessError as e:
        print(f"❌ Error running command: {command}")
        print(f"Error: {e.stderr}")
        return None


def get_project_name_from_cwd():
    """Get project name from current working directory."""
    return Path.cwd().name


def get_gitignore_template():
    """Return comprehensive .gitignore template."""
    return """# Byte-compiled / optimized / DLL files
__pycache__/
*.py[cod]
*$py.class

# C extensions
*.so

# Distribution / packaging
.Python
build/
develop-eggs/
dist/
downloads/
eggs/
.eggs/
lib/
lib64/
parts/
sdist/
var/
wheels/
share/python-wheels/
*.egg-info/
.installed.cfg
*.egg
MANIFEST

# PyInstaller
*.manifest
*.spec

# Installer logs
pip-log.txt
pip-delete-this-directory.txt

# Unit test / coverage reports
htmlcov/
.tox/
.nox/
.coverage
.coverage.*
.cache
nosetests.xml
coverage.xml
*.cover
*.py,cover
.hypothesis/
.pytest_cache/
cover/

# Translations
*.mo
*.pot

# Django stuff:
*.log
local_settings.py
db.sqlite3
db.sqlite3-journal

# Flask stuff:
instance/
.webassets-cache

# Scrapy stuff:
.scrapy

# Sphinx documentation
docs/_build/

# PyBuilder
.pybuilder/
target/

# Jupyter Notebook
.ipynb_checkpoints

# IPython
profile_default/
ipython_config.py

# pyenv
.python-version

# pipenv
Pipfile.lock

# poetry
poetry.lock

# pdm
.pdm.toml

# PEP 582
__pypackages__/

# Celery stuff
celerybeat-schedule
celerybeat.pid

# SageMath parsed files
*.sage.py

# Environments
.env
.venv
env/
venv/
ENV/
env.bak/
venv.bak/

# Spyder project settings
.spyderproject
.spyproject

# Rope project settings
.ropeproject

# mkdocs documentation
/site

# mypy
.mypy_cache/
.dmypy.json
dmypy.json

# Pyre type checker
.pyre/

# pytype static type analyzer
.pytype/

# Cython debug symbols
cython_debug/

# PyCharm
.idea/

# VS Code
.vscode/
!.vscode/settings.json
!.vscode/tasks.json
!.vscode/launch.json
!.vscode/extensions.json

# macOS
.DS_Store

# Windows
Thumbs.db
ehthumbs.db
Desktop.ini

# Data files
*.csv
*.json
*.xml
*.sqlite
*.db

# Model files
*.pkl
*.joblib
*.h5
*.onnx
*.pt
*.pth

# API keys and secrets
.env.local
.env.development.local
.env.test.local
.env.production.local
secrets.json
config.ini

# Logs
logs/
*.log
"""


def get_readme_template(project_name):
    """Return README.md template."""
    return f"""# {project_name}

## Description

Brief description of your project.

## Installation

```bash
pip install -r requirements.txt
```

## Usage

```python
from src.{project_name.lower().replace('-', '_').replace(' ', '_')} import main

main()
```

## Project Structure

```
{project_name}/
├── src/
│   └── {project_name.lower().replace('-', '_').replace(' ', '_')}/
│       ├── __init__.py
│       └── main.py
├── tests/
│   └── __init__.py
├── docs/
├── scripts/
├── .github/
│   └── workflows/
├── .copilot/
├── requirements.txt
├── setup.py
├── README.md
└── .gitignore
```

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Run tests
5. Submit a pull request

## License

MIT License
"""


def get_requirements_template():
    """Return requirements.txt template."""
    return """# Core dependencies
numpy>=1.21.0
pandas>=1.3.0
requests>=2.28.0

# Logging
loguru>=0.7.0

# Configuration
python-dotenv>=1.0.0
pydantic>=2.0.0

# Optional: Machine Learning
# scikit-learn>=1.3.0
# tensorflow>=2.13.0
# torch>=2.0.0

# Optional: Data visualization
# matplotlib>=3.7.0
# seaborn>=0.12.0
# plotly>=5.15.0

# Optional: Web framework
# flask>=2.3.0
# fastapi>=0.100.0
# uvicorn>=0.23.0

# Optional: Database
# sqlalchemy>=2.0.0
# pymongo>=4.4.0
# redis>=4.6.0

# Optional: API and HTTP
# httpx>=0.24.0
# aiohttp>=3.8.0
"""


def get_requirements_dev_template():
    """Return requirements-dev.txt template."""
    return """# Development dependencies
pytest>=7.4.0
pytest-cov>=4.1.0
pytest-mock>=3.11.0
pytest-asyncio>=0.21.0

# Code formatting and linting
black>=23.0.0
isort>=5.12.0
flake8>=6.0.0
pylint>=2.17.0
mypy>=1.5.0

# Pre-commit hooks
pre-commit>=3.3.0

# Documentation
sphinx>=7.1.0
mkdocs>=1.5.0
mkdocs-material>=9.1.0

# Security
bandit>=1.7.0
safety>=2.3.0

# Debugging and profiling
ipdb>=0.13.0
memory-profiler>=0.61.0

# Environment management
python-dotenv>=1.0.0

# Build tools
build>=0.10.0
twine>=4.0.0
"""


def get_pyproject_toml_template(project_name):
    """Return pyproject.toml template."""
    package_name = project_name.lower().replace('-', '_').replace(' ', '_')
    return f"""[build-system]
requires = ["setuptools>=61.0", "wheel"]
build-backend = "setuptools.build_meta"

[project]
name = "{project_name}"
version = "0.1.0"
description = "A brief description of your project"
readme = "README.md"
license = {{file = "LICENSE"}}
authors = [
    {{name = "Your Name", email = "your.email@example.com"}},
]
classifiers = [
    "Development Status :: 3 - Alpha",
    "Intended Audience :: Developers",
    "License :: OSI Approved :: MIT License",
    "Operating System :: OS Independent",
    "Programming Language :: Python :: 3",
    "Programming Language :: Python :: 3.8",
    "Programming Language :: Python :: 3.9",
    "Programming Language :: Python :: 3.10",
    "Programming Language :: Python :: 3.11",
    "Programming Language :: Python :: 3.12",
]
requires-python = ">=3.8"
dependencies = [
    "numpy>=1.21.0",
    "pandas>=1.3.0",
    "requests>=2.28.0",
    "loguru>=0.7.0",
    "python-dotenv>=1.0.0",
    "pydantic>=2.0.0",
    "click>=8.1.0",
    "rich>=13.0.0",
    "typer>=0.9.0",
]

[project.optional-dependencies]
dev = [
    "pytest>=7.4.0",
    "pytest-cov>=4.1.0",
    "black>=23.0.0",
    "isort>=5.12.0",
    "flake8>=6.0.0",
    "mypy>=1.5.0",
    "pre-commit>=3.3.0",
    "ruff>=0.0.290",
    "commitizen>=3.0.0",
    "coverage[toml]>=7.0.0",
]
ml = [
    "scikit-learn>=1.3.0",
    "tensorflow>=2.13.0",
    "torch>=2.0.0",
    "xgboost>=1.7.0",
    "lightgbm>=4.0.0",
]
viz = [
    "matplotlib>=3.7.0",
    "seaborn>=0.12.0",
    "plotly>=5.15.0",
    "bokeh>=3.0.0",
]
docs = [
    "sphinx>=7.1.0",
    "sphinx-rtd-theme>=1.3.0",
    "myst-parser>=2.0.0",
    "sphinx-autodoc-typehints>=1.24.0",
]
test = [
    "pytest>=7.4.0",
    "pytest-cov>=4.1.0",
    "pytest-mock>=3.11.0",
    "pytest-xdist>=3.3.0",
    "hypothesis>=6.82.0",
    "factory-boy>=3.3.0",
]

[project.scripts]
{package_name} = "{package_name}.cli:main"

[project.urls]
Homepage = "https://github.com/yourusername/{project_name}"
Repository = "https://github.com/yourusername/{project_name}"
Issues = "https://github.com/yourusername/{project_name}/issues"
Documentation = "https://{project_name}.readthedocs.io/"
Changelog = "https://github.com/yourusername/{project_name}/blob/main/CHANGELOG.md"

[tool.setuptools.packages.find]
where = ["src"]

[tool.setuptools.package-dir]
"" = "src"

[tool.black]
line-length = 88
target-version = ["py38", "py39", "py310", "py311"]
include = "\\.pyi?$"

[tool.isort]
profile = "black"
multi_line_output = 3
line_length = 88
known_first_party = ["{package_name}"]
known_third_party = ["pytest", "click", "typer", "rich"]

[tool.ruff]
line-length = 88
target-version = "py38"
select = [
    "E",
    "W", 
    "F",
    "I",
    "C",
    "B",
    "UP",
]
ignore = [
    "E501",
    "B008",
    "C901",
]

[tool.ruff.per-file-ignores]
"__init__.py" = ["F401"]

[tool.mypy]
python_version = "3.8"
warn_return_any = true
warn_unused_configs = true
disallow_untyped_defs = true
disallow_incomplete_defs = true
check_untyped_defs = true
no_implicit_optional = true
warn_redundant_casts = true
warn_unused_ignores = true
show_error_codes = true
namespace_packages = true
explicit_package_bases = true

[[tool.mypy.overrides]]
module = "tests.*"
disallow_untyped_defs = false

[tool.pytest.ini_options]
minversion = "7.0"
testpaths = ["tests"]
python_files = ["test_*.py", "*_test.py"]
python_classes = ["Test*"]
python_functions = ["test_*"]
addopts = [
    "--strict-markers",
    "--strict-config",
    "--cov=src",
    "--cov-report=term-missing",
    "--cov-report=html",
    "--cov-report=xml",
    "--cov-fail-under=80",
    "--durations=10",
]
markers = [
    "slow: marks tests as slow (deselect with '-m \\"not slow\\"')",
    "integration: marks tests as integration tests",
    "unit: marks tests as unit tests",
    "smoke: marks tests as smoke tests",
]
filterwarnings = [
    "error",
    "ignore::UserWarning",
    "ignore::DeprecationWarning",
]

[tool.coverage.run]
source = ["src"]
omit = ["*/tests/*", "*/test_*"]
branch = true

[tool.coverage.report]
exclude_lines = [
    "pragma: no cover",
    "def __repr__",
    "if self.debug:",
    "if settings.DEBUG",
    "raise AssertionError",
    "raise NotImplementedError",
    "if 0:",
    "if __name__ == .__main__.:",
    "class .*\\\\bProtocol\\\\):",
    "@(abc\\\\.)?abstractmethod",
]
show_missing = true
precision = 2

[tool.commitizen]
name = "cz_conventional_commits"
version = "0.1.0"
tag_format = "v$version"
"""


def get_cli_template(package_name):
    """Return CLI module template using typer/click."""
    return f'''"""
Command Line Interface for {package_name}
"""

import typer
from rich.console import Console
from rich.table import Table
from loguru import logger
from typing import Optional

from .main import main as run_main
from .__init__ import __version__

app = typer.Typer(
    name="{package_name}",
    help="A Python package for [description]",
    add_completion=False,
)
console = Console()


def version_callback(value: bool):
    """Show version and exit."""
    if value:
        console.print(f"{package_name} version: {{__version__}}")
        raise typer.Exit()


@app.callback()
def main(
    version: Optional[bool] = typer.Option(
        None, 
        "--version", 
        "-v",
        callback=version_callback,
        is_eager=True,
        help="Show version and exit."
    ),
    verbose: bool = typer.Option(
        False, 
        "--verbose", 
        help="Enable verbose output."
    ),
    quiet: bool = typer.Option(
        False, 
        "--quiet", 
        help="Suppress output."
    ),
):
    """
    {package_name} - A Python package for [description]
    """
    if verbose:
        logger.info("Verbose mode enabled")
    if quiet:
        logger.remove()


@app.command()
def run(
    config: Optional[str] = typer.Option(
        None,
        "--config",
        "-c", 
        help="Path to configuration file."
    ),
    dry_run: bool = typer.Option(
        False,
        "--dry-run",
        help="Show what would be done without executing."
    ),
):
    """
    Run the main application.
    """
    if dry_run:
        console.print("[yellow]DRY RUN MODE - No changes will be made[/yellow]")
    
    try:
        run_main()
        console.print("[green]✅ Operation completed successfully![/green]")
    except Exception as e:
        console.print(f"[red]❌ Error: {{e}}[/red]")
        raise typer.Exit(code=1)


@app.command()
def info():
    """
    Show package information.
    """
    table = Table(title=f"{package_name} Information")
    table.add_column("Property", style="cyan")
    table.add_column("Value", style="green")
    
    table.add_row("Package", "{package_name}")
    table.add_row("Version", __version__)
    table.add_row("Python", ">=3.8")
    
    console.print(table)


if __name__ == "__main__":
    app()
'''


def get_init_py_template(package_name):
    """Return __init__.py template."""
    return f'''"""
{package_name} package

A modern Python package with best practices.
"""

__version__ = "0.1.0"
__author__ = "Your Name"
__email__ = "your.email@example.com"
__license__ = "MIT"

from .main import main

__all__ = ["main", "__version__"]
'''


def get_conftest_py_template():
    """Return pytest conftest.py template."""
    return '''"""
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
'''


def get_changelog_template():
    """Return CHANGELOG.md template."""
    return """# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added
- Initial project structure
- Basic functionality
- Documentation
- Tests

### Changed

### Deprecated

### Removed

### Fixed

### Security

## [0.1.0] - 2023-XX-XX

### Added
- Initial release
- Basic project structure
- Core functionality
- Documentation
- Test suite
- CI/CD pipeline

[Unreleased]: https://github.com/username/project/compare/v0.1.0...HEAD
[0.1.0]: https://github.com/username/project/releases/tag/v0.1.0
"""


def get_contributing_template():
    """Return CONTRIBUTING.md template."""
    return """# Contributing

We love your input! We want to make contributing to this project as easy and transparent as possible, whether it's:

- Reporting a bug
- Discussing the current state of the code
- Submitting a fix
- Proposing new features
- Becoming a maintainer

## Development Process

We use GitHub to host code, to track issues and feature requests, as well as accept pull requests.

### Pull Requests

1. Fork the repo and create your branch from `main`.
2. If you've added code that should be tested, add tests.
3. If you've changed APIs, update the documentation.
4. Ensure the test suite passes.
5. Make sure your code lints.
6. Issue that pull request!

### Development Setup

```bash
# Clone your fork
git clone https://github.com/yourusername/project.git
cd project

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\\Scripts\\activate

# Install development dependencies
make install-dev

# Install pre-commit hooks
pre-commit install

# Run tests
make test

# Check code quality
make check
```

### Code Style

We use several tools to maintain code quality:

- **Black** for code formatting
- **isort** for import sorting
- **flake8** for linting
- **mypy** for type checking
- **pytest** for testing

Run all checks with:
```bash
make check
```

### Commit Message Convention

We follow the [Conventional Commits](https://www.conventionalcommits.org/) specification:

```
<type>[optional scope]: <description>

[optional body]

[optional footer(s)]
```

Types:
- `feat`: A new feature
- `fix`: A bug fix
- `docs`: Documentation only changes
- `style`: Changes that do not affect the meaning of the code
- `refactor`: A code change that neither fixes a bug nor adds a feature
- `perf`: A code change that improves performance
- `test`: Adding missing tests or correcting existing tests
- `chore`: Changes to the build process or auxiliary tools

### Testing

- Write tests for new features
- Ensure all tests pass
- Aim for high test coverage
- Include both unit and integration tests

### Documentation

- Update docstrings for new functions/classes
- Update README.md if needed
- Add examples for new features

## License

By contributing, you agree that your contributions will be licensed under the same license as the project.
"""


def get_docker_template(package_name):
    """Return Dockerfile template."""
    return f"""# Use Python 3.11 slim image
FROM python:3.11-slim

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE=1 \\
    PYTHONUNBUFFERED=1 \\
    PYTHONPATH=/app \\
    PIP_NO_CACHE_DIR=1 \\
    PIP_DISABLE_PIP_VERSION_CHECK=1

# Set work directory
WORKDIR /app

# Install system dependencies
RUN apt-get update \\
    && apt-get install -y --no-install-recommends \\
        build-essential \\
        git \\
    && rm -rf /var/lib/apt/lists/*

# Copy dependency files
COPY requirements.txt .
COPY requirements-dev.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy project
COPY . .

# Install project in editable mode
RUN pip install -e .

# Create non-root user
RUN groupadd -r appuser && useradd -r -g appuser appuser
RUN chown -R appuser:appuser /app
USER appuser

# Expose port (if needed)
# EXPOSE 8000

# Command to run the application
CMD ["{package_name}"]
"""


def get_docker_compose_template():
    """Return docker-compose.yml template."""
    return """version: '3.8'

services:
  app:
    build: .
    container_name: app
    volumes:
      - .:/app
      - /app/venv  # Exclude venv from mount
    environment:
      - PYTHONPATH=/app
      - LOG_LEVEL=INFO
    # ports:
    #   - "8000:8000"
    depends_on:
      - redis
      - postgres

  postgres:
    image: postgres:15-alpine
    container_name: postgres
    environment:
      POSTGRES_DB: app_db
      POSTGRES_USER: app_user
      POSTGRES_PASSWORD: app_password
    volumes:
      - postgres_data:/var/lib/postgresql/data
    ports:
      - "5432:5432"

  redis:
    image: redis:7-alpine
    container_name: redis
    ports:
      - "6379:6379"
    volumes:
      - redis_data:/data

volumes:
  postgres_data:
  redis_data:
"""


def get_github_workflow_template():
    """Return GitHub Actions workflow template."""
    return """name: CI/CD

on:
  push:
    branches: [ main, develop ]
  pull_request:
    branches: [ main ]
  release:
    types: [ published ]

env:
  PYTHON_VERSION: "3.11"

jobs:
  test:
    runs-on: ${{ matrix.os }}
    strategy:
      matrix:
        os: [ubuntu-latest, windows-latest, macos-latest]
        python-version: ["3.8", "3.9", "3.10", "3.11", "3.12"]

    steps:
    - uses: actions/checkout@v4
    
    - name: Set up Python ${{ matrix.python-version }}
      uses: actions/setup-python@v4
      with:
        python-version: ${{ matrix.python-version }}
        cache: 'pip'
    
    - name: Install dependencies
      run: |
        python -m pip install --upgrade pip
        pip install -e ".[dev,test]"
    
    - name: Lint with ruff
      run: ruff check src tests
    
    - name: Format check with black
      run: black --check src tests
    
    - name: Type check with mypy
      run: mypy src
    
    - name: Test with pytest
      run: |
        pytest tests/ -v \\
          --cov=src \\
          --cov-report=xml \\
          --cov-report=term-missing \\
          --junit-xml=pytest.xml
    
    - name: Upload coverage to Codecov
      if: matrix.python-version == env.PYTHON_VERSION && matrix.os == 'ubuntu-latest'
      uses: codecov/codecov-action@v3
      with:
        file: ./coverage.xml
        flags: unittests
        name: codecov-umbrella

  security:
    runs-on: ubuntu-latest
    steps:
    - uses: actions/checkout@v4
    
    - name: Set up Python
      uses: actions/setup-python@v4
      with:
        python-version: ${{ env.PYTHON_VERSION }}
    
    - name: Install dependencies
      run: |
        python -m pip install --upgrade pip
        pip install bandit safety
    
    - name: Run security checks
      run: |
        bandit -r src/
        safety check

  build:
    needs: [test, security]
    runs-on: ubuntu-latest
    if: github.event_name == 'release'
    
    steps:
    - uses: actions/checkout@v4
    
    - name: Set up Python
      uses: actions/setup-python@v4
      with:
        python-version: ${{ env.PYTHON_VERSION }}
    
    - name: Install build dependencies
      run: |
        python -m pip install --upgrade pip
        pip install build twine
    
    - name: Build package
      run: python -m build
    
    - name: Publish to PyPI
      env:
        TWINE_USERNAME: __token__
        TWINE_PASSWORD: ${{ secrets.PYPI_API_TOKEN }}
      run: twine upload dist/*
"""


def get_copilot_config():
    """Return Copilot configuration."""
    return """{
  "completions": {
    "enable": true,
    "suggestions": {
      "enable": true,
      "count": 3
    }
  },
  "chat": {
    "enable": true
  },
  "language": {
    "python": {
      "enable": true,
      "suggestions": true
    }
  }
}
"""


def create_virtual_environment(project_path):
    """Create and setup virtual environment."""
    venv_path = project_path / "venv"
    
    if venv_path.exists():
        print(f"✅ Virtual environment already exists: {venv_path}")
        # Determine the correct activation script and python executable
        if os.name == 'nt':  # Windows
            activate_script = venv_path / "Scripts" / "activate"
            python_exe = venv_path / "Scripts" / "python.exe"
            pip_exe = venv_path / "Scripts" / "pip.exe"
        else:  # Unix/Linux/macOS
            activate_script = venv_path / "bin" / "activate"
            python_exe = venv_path / "bin" / "python"
            pip_exe = venv_path / "bin" / "pip"
        
        return str(activate_script), str(python_exe), str(pip_exe)
    
    print("🐍 Creating virtual environment...")
    
    # Create virtual environment
    result = run_command(f"python -m venv {venv_path}", cwd=project_path)
    if result and result.returncode == 0:
        print("✅ Virtual environment created")
        
        # Determine the correct activation script and python executable
        if os.name == 'nt':  # Windows
            activate_script = venv_path / "Scripts" / "activate"
            python_exe = venv_path / "Scripts" / "python.exe"
            pip_exe = venv_path / "Scripts" / "pip.exe"
        else:  # Unix/Linux/macOS
            activate_script = venv_path / "bin" / "activate"
            python_exe = venv_path / "bin" / "python"
            pip_exe = venv_path / "bin" / "pip"
        
        # Upgrade pip in virtual environment
        if python_exe.exists():
            print("📦 Upgrading pip in virtual environment...")
            upgrade_result = run_command(f'"{python_exe}" -m pip install --upgrade pip', cwd=project_path)
            if upgrade_result and upgrade_result.returncode == 0:
                print("✅ Pip upgraded successfully")
        
        return str(activate_script), str(python_exe), str(pip_exe)
    else:
        print("❌ Failed to create virtual environment")
        return None, None, None


def install_dependencies(project_path, pip_exe):
    """Install project dependencies in virtual environment."""
    print("📦 Installing dependencies...")
    
    # Check if pyproject.toml exists and is valid before installing
    pyproject_file = project_path / "pyproject.toml"
    if pyproject_file.exists():
        try:
            # Try to install in editable mode
            result = run_command(f'"{pip_exe}" install -e .', cwd=project_path, check=False)
            if result and result.returncode == 0:
                print("✅ Main dependencies installed")
            else:
                print("⚠️  Failed to install in editable mode, trying basic requirements...")
                # Fallback to just requirements.txt
                req_result = run_command(f'"{pip_exe}" install -r requirements.txt', cwd=project_path, check=False)
                if req_result and req_result.returncode == 0:
                    print("✅ Basic requirements installed")
        except Exception as e:
            print(f"⚠️  Error installing dependencies: {e}")
    
    # Install development dependencies
    dev_result = run_command(f'"{pip_exe}" install -r requirements-dev.txt', cwd=project_path, check=False)
    if dev_result and dev_result.returncode == 0:
        print("✅ Development dependencies installed")
        
        # Setup pre-commit hooks
        precommit_result = run_command(f'"{pip_exe}" install pre-commit', cwd=project_path, check=False)
        if precommit_result and precommit_result.returncode == 0:
            # Install pre-commit hooks
            if os.name == 'nt':  # Windows
                precommit_exe = project_path / "venv" / "Scripts" / "pre-commit.exe"
            else:  # Unix/Linux/macOS
                precommit_exe = project_path / "venv" / "bin" / "pre-commit"
            
            if precommit_exe.exists():
                run_command(f'"{precommit_exe}" install', cwd=project_path, check=False)


def get_pyproject_toml_template(project_name):
    """Return pyproject.toml template."""
    package_name = project_name.lower().replace('-', '_').replace(' ', '_')
    return f"""[build-system]
requires = ["setuptools>=61.0", "wheel"]
build-backend = "setuptools.build_meta"

[project]
name = "{project_name}"
version = "0.1.0"
description = "A brief description of your project"
readme = "README.md"
license = {{file = "LICENSE"}}
authors = [
    {{name = "Your Name", email = "your.email@example.com"}},
]
classifiers = [
    "Development Status :: 3 - Alpha",
    "Intended Audience :: Developers",
    "License :: OSI Approved :: MIT License",
    "Operating System :: OS Independent",
    "Programming Language :: Python :: 3",
    "Programming Language :: Python :: 3.8",
    "Programming Language :: Python :: 3.9",
    "Programming Language :: Python :: 3.10",
    "Programming Language :: Python :: 3.11",
    "Programming Language :: Python :: 3.12",
]
requires-python = ">=3.8"
dependencies = [
    "numpy>=1.21.0",
    "pandas>=1.3.0",
    "requests>=2.28.0",
    "loguru>=0.7.0",
    "python-dotenv>=1.0.0",
    "pydantic>=2.0.0",
    "click>=8.1.0",
    "rich>=13.0.0",
    "typer>=0.9.0",
]

[project.optional-dependencies]
dev = [
    "pytest>=7.4.0",
    "pytest-cov>=4.1.0",
    "black>=23.0.0",
    "isort>=5.12.0",
    "flake8>=6.0.0",
    "mypy>=1.5.0",
    "pre-commit>=3.3.0",
    "ruff>=0.0.290",
    "commitizen>=3.0.0",
    "coverage[toml]>=7.0.0",
]
ml = [
    "scikit-learn>=1.3.0",
    "tensorflow>=2.13.0",
    "torch>=2.0.0",
    "xgboost>=1.7.0",
    "lightgbm>=4.0.0",
]
viz = [
    "matplotlib>=3.7.0",
    "seaborn>=0.12.0",
    "plotly>=5.15.0",
    "bokeh>=3.0.0",
]
docs = [
    "sphinx>=7.1.0",
    "sphinx-rtd-theme>=1.3.0",
    "myst-parser>=2.0.0",
    "sphinx-autodoc-typehints>=1.24.0",
]
test = [
    "pytest>=7.4.0",
    "pytest-cov>=4.1.0",
    "pytest-mock>=3.11.0",
    "pytest-xdist>=3.3.0",
    "hypothesis>=6.82.0",
    "factory-boy>=3.3.0",
]

[project.scripts]
{package_name} = "{package_name}.cli:main"

[project.urls]
Homepage = "https://github.com/yourusername/{project_name}"
Repository = "https://github.com/yourusername/{project_name}"
Issues = "https://github.com/yourusername/{project_name}/issues"
Documentation = "https://{project_name}.readthedocs.io/"
Changelog = "https://github.com/yourusername/{project_name}/blob/main/CHANGELOG.md"

[tool.setuptools.packages.find]
where = ["src"]

[tool.setuptools.package-dir]
"" = "src"

[tool.black]
line-length = 88
target-version = ["py38", "py39", "py310", "py311"]
include = "\\.pyi?$"

[tool.isort]
profile = "black"
multi_line_output = 3
line_length = 88
known_first_party = ["{package_name}"]
known_third_party = ["pytest", "click", "typer", "rich"]

[tool.ruff]
line-length = 88
target-version = "py38"
select = [
    "E",
    "W", 
    "F",
    "I",
    "C",
    "B",
    "UP",
]
ignore = [
    "E501",
    "B008",
    "C901",
]

[tool.ruff.per-file-ignores]
"__init__.py" = ["F401"]

[tool.mypy]
python_version = "3.8"
warn_return_any = true
warn_unused_configs = true
disallow_untyped_defs = true
disallow_incomplete_defs = true
check_untyped_defs = true
no_implicit_optional = true
warn_redundant_casts = true
warn_unused_ignores = true
show_error_codes = true
namespace_packages = true
explicit_package_bases = true

[[tool.mypy.overrides]]
module = "tests.*"
disallow_untyped_defs = false

[tool.pytest.ini_options]
minversion = "7.0"
testpaths = ["tests"]
python_files = ["test_*.py", "*_test.py"]
python_classes = ["Test*"]
python_functions = ["test_*"]
addopts = [
    "--strict-markers",
    "--strict-config",
    "--cov=src",
    "--cov-report=term-missing",
    "--cov-report=html",
    "--cov-report=xml",
    "--cov-fail-under=80",
    "--durations=10",
]
markers = [
    "slow: marks tests as slow (deselect with '-m \\"not slow\\"')",
    "integration: marks tests as integration tests",
    "unit: marks tests as unit tests",
    "smoke: marks tests as smoke tests",
]
filterwarnings = [
    "error",
    "ignore::UserWarning",
    "ignore::DeprecationWarning",
]

[tool.coverage.run]
source = ["src"]
omit = ["*/tests/*", "*/test_*"]
branch = true

[tool.coverage.report]
exclude_lines = [
    "pragma: no cover",
    "def __repr__",
    "if self.debug:",
    "if settings.DEBUG",
    "raise AssertionError",
    "raise NotImplementedError",
    "if 0:",
    "if __name__ == .__main__.:",
    "class .*\\\\bProtocol\\\\):",
    "@(abc\\\\.)?abstractmethod",
]
show_missing = true
precision = 2

[tool.commitizen]
name = "cz_conventional_commits"
version = "0.1.0"
tag_format = "v$version"
"""


def get_main_py_template(package_name):
    """Return main.py template with logging."""
    return f'''"""
Main module for {package_name}
"""

import sys
from pathlib import Path
from loguru import logger
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Configure logger
logger.remove()  # Remove default handler
logger.add(
    sys.stderr,
    format="<green>{{time:YYYY-MM-DD HH:mm:ss}}</green> | <level>{{level: <8}}</level> | <cyan>{{name}}</cyan>:<cyan>{{function}}</cyan>:<cyan>{{line}}</cyan> - <level>{{message}}</level>",
    level="INFO"
)
logger.add(
    "logs/app.log",
    rotation="10 MB",
    retention="10 days",
    level="DEBUG",
    format="{{time:YYYY-MM-DD HH:mm:ss}} | {{level: <8}} | {{name}}:{{function}}:{{line}} - {{message}}"
)


def setup_logging():
    """Setup logging configuration."""
    logs_dir = Path("logs")
    logs_dir.mkdir(exist_ok=True)
    logger.info("Logging setup complete")


def main():
    """Main entry point for the application."""
    setup_logging()
    logger.info("Starting {package_name} application")
    
    try:
        logger.info("Hello from {package_name}!")
        # Your application logic here
        
    except Exception as e:
        logger.error(f"Application error: {{e}}")
        sys.exit(1)
    
    logger.info("Application completed successfully")


if __name__ == "__main__":
    main()
'''


def get_precommit_template():
    """Return .pre-commit-config.yaml template."""
    return """repos:
  - repo: https://github.com/pre-commit/pre-commit-hooks
    rev: v4.4.0
    hooks:
      - id: trailing-whitespace
      - id: end-of-file-fixer
      - id: check-yaml
      - id: check-added-large-files
      - id: check-merge-conflict
      - id: debug-statements

  - repo: https://github.com/psf/black
    rev: 23.7.0
    hooks:
      - id: black
        language_version: python3

  - repo: https://github.com/pycqa/isort
    rev: 5.12.0
    hooks:
      - id: isort
        args: ["--profile", "black"]

  - repo: https://github.com/pycqa/flake8
    rev: 6.0.0
    hooks:
      - id: flake8
        additional_dependencies: [flake8-docstrings]

  - repo: https://github.com/pre-commit/mirrors-mypy
    rev: v1.5.1
    hooks:
      - id: mypy
        additional_dependencies: [types-requests]
"""


def get_makefile_template(package_name):
    """Return Makefile template."""
    return f""".PHONY: help install install-dev test lint format type-check clean build

help:  ## Show this help
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | sort | awk 'BEGIN {{FS = ":.*?## "}}; {{printf "\\033[36m%-20s\\033[0m %s\\n", $$1, $$2}}'

install:  ## Install production dependencies
	pip install -e .

install-dev:  ## Install development dependencies
	pip install -e ".[dev]"
	pip install -r requirements-dev.txt
	pre-commit install

test:  ## Run tests
	pytest tests/ -v --cov=src --cov-report=term-missing

test-fast:  ## Run tests without coverage
	pytest tests/ -v

lint:  ## Run linting
	flake8 src tests
	pylint src

format:  ## Format code
	black src tests
	isort src tests

type-check:  ## Run type checking
	mypy src

check:  ## Run all checks (lint, type-check, test)
	$(MAKE) format
	$(MAKE) lint
	$(MAKE) type-check
	$(MAKE) test

clean:  ## Clean build artifacts
	rm -rf build/
	rm -rf dist/
	rm -rf *.egg-info/
	rm -rf .coverage
	rm -rf htmlcov/
	rm -rf .pytest_cache/
	rm -rf .mypy_cache/
	find . -type d -name __pycache__ -delete
	find . -type f -name "*.pyc" -delete

build:  ## Build package
	python -m build

venv:  ## Create virtual environment
	python -m venv venv
	@echo "Virtual environment created. Activate with:"
	@echo "source venv/bin/activate  # Linux/Mac"
	@echo "venv\\Scripts\\activate     # Windows"

setup:  ## Full project setup
	$(MAKE) venv
	@echo "Please activate the virtual environment and run 'make install-dev'"
"""

def run_setup_steps(project_path, activate_script, pip_exe):
    """Run the automated setup steps."""
    print("\n🔄 Running automated setup steps...")
    
    # Step 1: Git initialization
    print("\n📝 Initializing Git repository...")
    git_init_result = run_command("git init", cwd=project_path, check=False)
    if git_init_result and git_init_result.returncode == 0:
        print("✅ Git repository initialized")
        
        # Add all files
        git_add_result = run_command("git add .", cwd=project_path, check=False)
        if git_add_result and git_add_result.returncode == 0:
            print("✅ Files added to git")
            
            # Initial commit
            commit_result = run_command('git commit -m "Initial commit"', cwd=project_path, check=False)
            if commit_result and commit_result.returncode == 0:
                print("✅ Initial commit created")
            else:
                print("⚠️  Failed to create initial commit")
        else:
            print("⚠️  Failed to add files to git")
    else:
        print("⚠️  Failed to initialize git repository")
    
    # Step 2: Install development dependencies
    if pip_exe:
        print("\n📦 Installing development dependencies...")
        dev_install_result = run_command(f'"{pip_exe}" install -e ".[dev]"', cwd=project_path, check=False)
        if dev_install_result and dev_install_result.returncode == 0:
            print("✅ Development dependencies installed")
        else:
            print("⚠️  Failed to install development dependencies, trying fallback...")
            # Fallback: install requirements-dev.txt
            fallback_result = run_command(f'"{pip_exe}" install -r requirements-dev.txt', cwd=project_path, check=False)
            if fallback_result and fallback_result.returncode == 0:
                print("✅ Development dependencies installed (fallback)")
    
    # Step 3: Setup pre-commit hooks
    if pip_exe:
        print("\n🔧 Setting up pre-commit hooks...")
        precommit_install_result = run_command(f'"{pip_exe}" install pre-commit', cwd=project_path, check=False)
        if precommit_install_result and precommit_install_result.returncode == 0:
            # Install pre-commit hooks
            if os.name == 'nt':  # Windows
                precommit_exe = project_path / "venv" / "Scripts" / "pre-commit.exe"
            else:  # Unix/Linux/macOS
                precommit_exe = project_path / "venv" / "bin" / "pre-commit"
            
            if precommit_exe.exists():
                hooks_result = run_command(f'"{precommit_exe}" install', cwd=project_path, check=False)
                if hooks_result and hooks_result.returncode == 0:
                    print("✅ Pre-commit hooks installed")
                else:
                    print("⚠️  Failed to install pre-commit hooks")
            else:
                print("⚠️  Pre-commit executable not found")
        else:
            print("⚠️  Failed to install pre-commit")
    
    print("\n🎉 Automated setup completed!")


def create_python_project(project_name, base_path=".", create_venv=True, auto_setup=False):
    """Create a complete Python project structure."""
    # Fix: If project name matches current directory name, use current directory
    current_dir_name = Path.cwd().name
    
    if project_name == current_dir_name and base_path == ".":
        # Use current directory as project root
        project_path = Path.cwd()
        print(f"Using current directory as project root: {project_name}")
    else:
        # Create new subdirectory
        project_path = Path(base_path) / project_name
        print(f"Creating new project directory: {project_name}")
    
    package_name = project_name.lower().replace('-', '_').replace(' ', '_')
    
    print(f"Creating Python project: {project_name}")
    print(f"Location: {project_path.absolute()}")
    
    # Only create main project directory if it doesn't exist and we're not using current dir
    if not project_path.exists():
        create_directory(project_path)
    elif project_path == Path.cwd():
        print(f"Using existing directory: {project_path}")

    # Create directory structure
    directories = [
        "src",
        f"src/{package_name}",
        "tests",
        "docs",
        "scripts", 
        "logs",
        ".github/workflows",
        ".copilot",
        "data",
        "notebooks",
        "config",
        "examples",
        "assets",
    ]
    
    for directory in directories:
        create_directory(project_path / directory)
    
    # Create files
    files_to_create = [
        (".gitignore", get_gitignore_template()),
        ("README.md", get_readme_template(project_name)),
        ("requirements.txt", get_requirements_template()),
        ("requirements-dev.txt", get_requirements_dev_template()),
        ("pyproject.toml", get_pyproject_toml_template(project_name)),
        (".pre-commit-config.yaml", get_precommit_template()),
        ("Makefile", get_makefile_template(package_name)),
        ("CHANGELOG.md", get_changelog_template()),
        ("CONTRIBUTING.md", get_contributing_template()),
        ("Dockerfile", get_docker_template(package_name)),
        ("docker-compose.yml", get_docker_compose_template()),
        (f"src/{package_name}/__init__.py", get_init_py_template(package_name)),
        (f"src/{package_name}/main.py", get_main_py_template(package_name)),
        (f"src/{package_name}/cli.py", get_cli_template(package_name)),
        ("tests/__init__.py", ""),
        ("tests/conftest.py", get_conftest_py_template()),
        (f"tests/test_{package_name}.py", f'"""Tests for {package_name}"""\n\nimport pytest\nfrom src.{package_name} import main\n\n\ndef test_main():\n    """Test main function."""\n    # Add your tests here\n    assert main is not None\n\n\ndef test_version():\n    """Test version import."""\n    from src.{package_name} import __version__\n    assert __version__ is not None\n'),
        (".github/workflows/ci.yml", get_github_workflow_template()),
        (".copilot/config.json", get_copilot_config()),
        ("docs/README.md", f"# {project_name} Documentation\n\nDocumentation for {project_name} will go here.\n"),
        ("scripts/README.md", "# Scripts\n\nUtility scripts for the project.\n"),
        ("data/README.md", "# Data\n\nData files and datasets.\n"),
        ("notebooks/README.md", "# Notebooks\n\nJupyter notebooks for analysis and experimentation.\n"),
        ("config/README.md", "# Configuration\n\nConfiguration files for the project.\n"),
        ("examples/README.md", "# Examples\n\nUsage examples and tutorials.\n"),
        (".env.example", "# Environment variables\n# Copy to .env and fill in your values\n\n# Database\n# DATABASE_URL=sqlite:///app.db\n\n# API Keys\n# API_KEY=your_api_key_here\n\n# Logging\n# LOG_LEVEL=INFO\n"),
        ("LICENSE", "MIT License\n\nCopyright (c) 2023 Your Name\n\nPermission is hereby granted, free of charge, to any person obtaining a copy\nof this software and associated documentation files (the \"Software\"), to deal\nin the Software without restriction, including without limitation the rights\nto use, copy, modify, merge, publish, distribute, sublicense, and/or sell\ncopies of the Software, and to permit persons to whom the Software is\nfurnished to do so, subject to the following conditions:\n\nThe above copyright notice and this permission notice shall be included in all\ncopies or substantial portions of the Software.\n\nTHE SOFTWARE IS PROVIDED \"AS IS\", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR\nIMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,\nFITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE\nAUTHORs OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER\nLIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,\nOUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE\nSOFTWARE.\n"),
    ]
    
    for filename, content in files_to_create:
        create_file(project_path / filename, content)
    
    # Create virtual environment if requested
    activate_script = None
    python_exe = None
    pip_exe = None
    
    if create_venv:
        activate_script, python_exe, pip_exe = create_virtual_environment(project_path)
        if pip_exe:
            install_dependencies(project_path, pip_exe)
    
    # Run automated setup if requested
    if auto_setup and activate_script and pip_exe:
        run_setup_steps(project_path, activate_script, pip_exe)
    
    print(f"\n✅ Successfully created Python project: {project_name}")
    print(f"📁 Location: {project_path.absolute()}")
    
    # Show next steps or completion message
    if auto_setup and activate_script:
        print("\n🎯 Project setup completed! You can now:")
        print("1. Activate the virtual environment:")
        if os.name == 'nt':  # Windows
            print(f'   "{activate_script}"')
        else:  # Unix/Linux/macOS
            print(f'   source "{activate_script}"')
        print("2. Start developing:")
        print("   make test        # Run tests")
        print("   make format      # Format code")
        print("   make check       # Run all checks")
    else:
        print("\n🚀 Next steps:")
        
        if create_venv and activate_script:
            if os.name == 'nt':  # Windows
                print(f'1. "{activate_script}"')
            else:  # Unix/Linux/macOS
                print(f'1. source "{activate_script}"')
            print("2. make install-dev  # or pip install -e '.[dev]'")
            print("3. pre-commit install")
        else:
            print("1. python -m venv venv")
            print('2. source "venv/bin/activate"  # On Windows: "venv\\Scripts\\activate"')
            print("3. make install-dev")
        
        print("4. git init && git add . && git commit -m 'Initial commit'")
        print("\n📚 Available make commands:")
        print("  make help        # Show all available commands")
        print("  make test        # Run tests")
        print("  make format      # Format code")
        print("  make check       # Run all checks")
    
    # Add helpful notes for directories with spaces
    if " " in str(project_path):
        print("\n⚠️  Note: Your project path contains spaces.")
        print("   Always use quotes when referencing paths in commands:")
        print(f'   cd "{project_path}"')
        if activate_script:
            print(f'   source "{activate_script}"')
        print("\n💡 Tip: Consider using underscores instead of spaces in directory names for easier command-line usage.")


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description="Create a generic Python project structure")
    parser.add_argument("project_name", nargs='?', help="Name of the project to create (default: current directory name)")
    parser.add_argument("--path", default=".", help="Base path where to create the project")
    parser.add_argument("--no-venv", action="store_true", help="Skip virtual environment creation")
    parser.add_argument("--force-subdir", action="store_true", help="Force creation of subdirectory even if project name matches current directory")
    parser.add_argument("--auto-setup", action="store_true", help="Automatically run setup steps (git init, install deps, pre-commit)")
    parser.add_argument("--minimal", action="store_true", help="Create minimal project structure without optional files")
    
    args = parser.parse_args()
    
    # Use current directory name if no project name provided
    if not args.project_name:
        args.project_name = get_project_name_from_cwd()
        print(f"Using current directory name as project name: {args.project_name}")
    
    try:
        if args.force_subdir:
            # Force subdirectory creation
            project_path = Path(args.path) / args.project_name
            package_name = args.project_name.lower().replace('-', '_').replace(' ', '_')
            
            print(f"Creating Python project in subdirectory: {args.project_name}")
            print(f"Location: {project_path.absolute()}")
            
            # Always create subdirectory
            create_directory(project_path)
            
            # Use the standard project creation but with forced path
            create_python_project(
                args.project_name, 
                str(project_path.parent), 
                create_venv=not args.no_venv,
                auto_setup=args.auto_setup
            )
        else:
            create_python_project(
                args.project_name, 
                args.path, 
                create_venv=not args.no_venv,
                auto_setup=args.auto_setup
            )
    except Exception as e:
        print(f"❌ Error creating project: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()