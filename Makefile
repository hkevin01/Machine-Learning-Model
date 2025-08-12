.PHONY: help install install-dev test lint format type-check clean build gui agent docs compile-deps monitor

help:  ## Show this help
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | sort | awk 'BEGIN {FS = ":.*?## "}; {printf "\033[36m%-20s\033[0m %s\n", $$1, $$2}'

compile-deps:  ## Compile (pin) requirements from .in files (bootstraps pip-tools if absent)
	@echo "[deps] Ensuring pip-tools is installed..."
	@python -c "import piptools" 2>/dev/null || python -m pip install --upgrade pip && python -m pip install pip-tools
	@echo "[deps] Compiling base requirements -> requirements.txt"
	pip-compile --resolver=backtracking --quiet --output-file requirements.txt requirements.in
	@echo "[deps] Compiling dev requirements -> requirements-dev.txt"
	pip-compile --resolver=backtracking --quiet --output-file requirements-dev.txt requirements-dev.in
	@echo "[deps] Done. To apply updated pins: pip install -r requirements.txt -r requirements-dev.txt"

install:  ## Install production dependencies
	pip install -r requirements.txt
	pip install -e .

install-dev:  ## Install development dependencies
	pip install -r requirements-dev.txt
	pip install -e .
	pre-commit install

test:  ## Run tests
	pytest --config-file=config/dev/pytest.ini tests/ -v --cov=src --cov-report=term-missing

test-fast:  ## Run tests without coverage
	pytest --config-file=config/dev/pytest.ini tests/ -v

test-gui:  ## Run GUI tests (requires X11)
	pytest --config-file=config/dev/pytest.ini tests/gui -v -m "gui"

test-agent:  ## Run Agent Mode tests
	pytest --config-file=config/dev/pytest.ini tests/agent -v -m "agent"

lint:  ## Run linting
	flake8 --config=config/dev/.flake8 src tests
	pylint src

format:  ## Format code
	black src tests
	isort src tests

type-check:  ## Run type checking
	mypy --config-file=config/dev/mypy.ini src

check:  ## Run all checks (lint, type-check, test)
	$(MAKE) format
	$(MAKE) lint
	$(MAKE) type-check
	$(MAKE) test

monitor:  ## Generate monitoring reports
	python scripts/monitoring/generate_reports.py --use-sample-data

mlflow-ui:  ## Launch MLflow UI (requires mlflow installed)
	mlflow ui --port 5000

dvc-init:  ## Initialize DVC (no-scm for lightweight example)
	dvc init --no-scm --force || true

dvc-run:  ## Execute DVC pipeline (prepare + train)
	dvc repro

docs:  ## Build documentation
	mkdocs build

docs-serve:  ## Serve documentation locally
	mkdocs serve

clean:  ## Clean build artifacts
	rm -rf build/
	rm -rf dist/
	rm -rf *.egg-info/
	rm -rf .coverage
	rm -rf htmlcov/
	rm -rf .pytest_cache/
	rm -rf .mypy_cache/
	rm -rf site/
	find . -type d -name __pycache__ -delete
	find . -type f -name "*.pyc" -delete

build:  ## Build package
	python -m build

gui:  ## Run PyQt6 GUI in Docker container
	./scripts/run_gui_docker.sh

agent:  ## Run Agent Mode in Docker container
	./scripts/run_agent_docker.sh

venv:  ## Create virtual environment
	python -m venv venv
	@echo "Virtual environment created. Activate with:"
	@printf '%s\n' "source venv/bin/activate  # Linux/Mac"
	@printf '%s\n' "venv\\Scripts\\activate     # Windows"

setup:  ## Full project setup
	$(MAKE) venv
	@echo "Please activate the virtual environment and run 'make install-dev'"
