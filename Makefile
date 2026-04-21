.PHONY: help build test test-fast lint format type-check clean docs docs-serve \
        docker-app docker-test docker-gui compile-deps

# ── help ──────────────────────────────────────────────────────────────────────
help: ## Show available targets
@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | sort | \
  awk 'BEGIN {FS = ":.*?## "}; {printf "\033[36m%-20s\033[0m %s\n", $$1, $$2}'

# ── docker (primary workflow) ─────────────────────────────────────────────────
docker-app: ## Build & run the ML CLI app in Docker
./run.sh app

docker-test: ## Build & run the full test suite in Docker
./run.sh test

docker-gui: ## Build & launch the PyQt6 GUI in Docker (requires X11)
./run.sh gui

build: ## Build all Docker images
./run.sh build

# ── local dev (no Docker) ─────────────────────────────────────────────────────
install: ## Install production dependencies locally
pip install -r requirements.txt && pip install -e .

install-dev: ## Install all dependencies + pre-commit hooks locally
pip install -r requirements.txt -r requirements-dev.txt
pip install -e .
pre-commit install

test: ## Run pytest locally (fast, no coverage)
PYTHONPATH=src python -m pytest tests/ --ignore=tests/gui -q --no-cov --tb=short

test-cov: ## Run pytest locally with coverage report
PYTHONPATH=src python -m pytest tests/ --ignore=tests/gui \
  --cov=src --cov-report=term-missing --cov-report=html

lint: ## Run flake8 linter
flake8 src tests

format: ## Auto-format with black + isort
black src tests
isort src tests

type-check: ## Run mypy type checker
mypy src

check: ## format → lint → type-check → test
$(MAKE) format lint type-check test

# ── data / ML pipeline ────────────────────────────────────────────────────────
prepare-data: ## Generate sample datasets from sklearn
PYTHONPATH=src python scripts/prepare_sample_data.py

train: ## Train the sample model
PYTHONPATH=src python scripts/train_sample_model.py

dvc-run: ## Execute DVC pipeline
dvc repro

mlflow-ui: ## Launch MLflow UI on port 5000
mlflow ui --port 5000

monitor: ## Generate monitoring reports
PYTHONPATH=src python scripts/monitoring/generate_reports.py --use-sample-data

# ── docs ──────────────────────────────────────────────────────────────────────
docs: ## Build MkDocs documentation
mkdocs build

docs-serve: ## Serve docs locally at http://localhost:8000
mkdocs serve

# ── dependency pinning ────────────────────────────────────────────────────────
compile-deps: ## Re-pin requirements.txt from requirements.in (needs pip-tools)
pip-compile --resolver=backtracking --quiet --output-file requirements.txt requirements.in
pip-compile --resolver=backtracking --quiet --output-file requirements-dev.txt requirements-dev.in

# ── clean ─────────────────────────────────────────────────────────────────────
clean: ## Remove build artifacts and caches
rm -rf build/ dist/ *.egg-info/ .coverage htmlcov/ .pytest_cache/ .mypy_cache/ site/
find . -type d -name __pycache__ -delete
find . -type f -name "*.pyc" -delete
