.PHONY: help install install-dev test lint format type-check clean build gui agent

help:  ## Show this help
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | sort | awk 'BEGIN {FS = ":.*?## "}; {printf "\033[36m%-20s\033[0m %s\n", $$1, $$2}'

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

gui:  ## Run PyQt6 GUI in Docker container
	./scripts/run_gui_docker.sh

agent:  ## Run Agent Mode in Docker container
	./scripts/run_agent_docker.sh

venv:  ## Create virtual environment
	python -m venv venv
	@echo "Virtual environment created. Activate with:"
	@echo "source venv/bin/activate  # Linux/Mac"
	@echo "venv\Scripts\activate     # Windows"

setup:  ## Full project setup
	$(MAKE) venv
	@echo "Please activate the virtual environment and run 'make install-dev'"
