<div align="center">
  <h1>🤖 Machine Learning Model</h1>
  <p><em>Agent-guided ML pipeline framework with PyQt6 GUI, experiment tracking, and production-ready model automation.</em></p>
</div>

<div align="center">

[![License](https://img.shields.io/github/license/hkevin01/Machine-Learning-Model?style=flat-square)](LICENSE)
[![GitHub Stars](https://img.shields.io/github/stars/hkevin01/Machine-Learning-Model?style=flat-square)](https://github.com/hkevin01/Machine-Learning-Model/stargazers)
[![GitHub Forks](https://img.shields.io/github/forks/hkevin01/Machine-Learning-Model?style=flat-square)](https://github.com/hkevin01/Machine-Learning-Model/network)
[![Last Commit](https://img.shields.io/github/last-commit/hkevin01/Machine-Learning-Model?style=flat-square)](https://github.com/hkevin01/Machine-Learning-Model/commits/main)
[![Repo Size](https://img.shields.io/github/repo-size/hkevin01/Machine-Learning-Model?style=flat-square)](https://github.com/hkevin01/Machine-Learning-Model)
[![Issues](https://img.shields.io/github/issues/hkevin01/Machine-Learning-Model?style=flat-square)](https://github.com/hkevin01/Machine-Learning-Model/issues)
[![Python](https://img.shields.io/badge/python-3.8%2B-blue?style=flat-square&logo=python)](https://python.org)
[![PyQt6](https://img.shields.io/badge/GUI-PyQt6-41cd52?style=flat-square&logo=qt)](https://www.riverbankcomputing.com/software/pyqt/)
[![scikit-learn](https://img.shields.io/badge/ML-scikit--learn-f7931e?style=flat-square&logo=scikitlearn)](https://scikit-learn.org)
[![MLflow](https://img.shields.io/badge/tracking-MLflow-0194e2?style=flat-square)](https://mlflow.org)
[![DVC](https://img.shields.io/badge/versioning-DVC-945dd6?style=flat-square)](https://dvc.org)
[![Docker](https://img.shields.io/badge/docker-ready-2496ed?style=flat-square&logo=docker)](https://www.docker.com)

</div>

---

## Table of Contents

- [Overview](#overview)
- [Key Features](#key-features)
- [Architecture](#architecture)
- [Usage Flow](#usage-flow)
- [Algorithm Coverage](#algorithm-coverage)
- [Technology Stack](#technology-stack)
- [Setup & Installation](#setup--installation)
- [Usage](#usage)
- [Core Capabilities](#core-capabilities)
- [Experiment Tracking](#experiment-tracking)
- [Data Versioning](#data-versioning)
- [Roadmap](#roadmap)
- [Development Status](#development-status)
- [Contributing](#contributing)
- [License](#license)

---

## Overview

**Machine Learning Model** is a comprehensive, agent-driven ML framework that automates the full machine learning lifecycle — from raw data ingestion to production model deployment — through a 13-step guided pipeline. It targets data scientists, ML engineers, and developers who want structured, reproducible ML workflows without sacrificing flexibility.

The framework pairs a PyQt6 graphical interface with an intelligent ML agent that provides context-aware recommendations at every pipeline stage. Traditional algorithm exploration and AI-guided automation coexist in a single, unified environment.

> [!IMPORTANT]
> Agent Mode is the primary workflow entry point. It guides you step-by-step through the entire ML pipeline with automatic state persistence so you can pause and resume at any stage.

<p align="right">(<a href="#top">back to top ↑</a>)</p>

---

## Key Features

| Icon | Feature | Description | Impact | Status |
|------|---------|-------------|--------|--------|
| 🤖 | **ML Agent** | AI-powered assistant navigating the 13-step pipeline | Critical | ✅ Stable |
| 🖥️ | **PyQt6 GUI** | Interactive workflow navigator with real-time progress | High | ✅ Stable |
| 💾 | **State Persistence** | Auto save/load of workflow progress across sessions | High | ✅ Stable |
| 📊 | **Enhanced Results** | Execution timing, hyperparameters, smart recommendations | High | ✅ Stable |
| 🧪 | **MLflow Tracking** | Experiment logging: params, metrics, feature importances | Medium | ✅ Stable |
| 🗃️ | **DVC Versioning** | Reproducible data & model pipelines via DVC | Medium | ✅ Stable |
| 🐳 | **Docker Support** | GUI-in-container with X11 forwarding and font rendering | Medium | ✅ Stable |
| ⚙️ | **Hyperparameter Tuning** | Automated optimization integrated into pipeline | High | 🟡 Beta |
| 📡 | **Drift Monitoring** | Continuous learning and model drift detection | Medium | 🟡 Beta |

**Highlights:**

- **13-step automated pipeline**: Data Collection → Preprocessing → EDA → Feature Engineering → Splitting → Algorithm Selection → Training → Evaluation → Tuning → Deployment → Monitoring → Experiment Tracking → Data Versioning
- **Rich algorithm output**: every algorithm run returns execution time, full hyperparameter config, performance category (Excellent/Good/Fair/Poor), and actionable recommendations
- **Cross-platform**: Linux, Windows, and basic macOS support with both shell and batch launchers

<p align="right">(<a href="#top">back to top ↑</a>)</p>

---

## Architecture

### System Architecture

```mermaid
flowchart TD
    User([👤 User]) --> Entry{Entry Point}
    Entry -->|Agent Mode| Agent[🤖 ML Agent\nml_agent.py]
    Entry -->|GUI Mode| GUI[🖥️ PyQt6 GUI\nmain_window_pyqt6.py]
    Entry -->|CLI Mode| CLI[⌨️ CLI\ncli.py]

    Agent --> Workflow[📋 ML Workflow\nml_workflow.py]
    GUI --> Workflow
    Workflow --> Steps[🔧 Step Implementations\nstep_implementations.py]

    Steps --> DataLayer[📦 Data Layer]
    DataLayer --> Loader[Data Loader]
    DataLayer --> Validator[Data Validator]

    Steps --> Supervised[🌲 Supervised Algorithms]
    Supervised --> DT[Decision Tree]
    Supervised --> RF[Random Forest]
    Supervised --> SKLearn[scikit-learn Suite]

    Steps --> Eval[📈 Evaluation\nMetrics & Reports]
    Steps --> Track[🧪 MLflow Tracking]
    Steps --> DVC[🗃️ DVC Versioning]

    Eval --> Results[EnhancedResult\nTiming + Recommendations]
    Results --> Viz[📊 Visualization\nmatplotlib / plotly]
```

### Component Responsibilities

| Component | Location | Responsibility |
|-----------|----------|----------------|
| `ML Agent` | `workflow/ml_agent.py` | Orchestrates pipeline steps, provides context-aware recommendations |
| `ML Workflow` | `workflow/ml_workflow.py` | State machine managing 13-step progression and persistence |
| `Step Implementations` | `workflow/step_implementations.py` | Concrete logic for each pipeline stage |
| `PyQt6 GUI` | `gui/main_window_pyqt6.py` | Interactive dashboard, progress tracking, real-time output |
| `CLI` | `cli.py` | Typer-based command-line interface |
| `Supervised` | `supervised/` | Decision Tree, Random Forest with enhanced result output |
| `Tracking` | `tracking/` | MLflow integration for experiment logging |
| `Visualization` | `visualization/` | matplotlib, seaborn, plotly chart generation |

> [!NOTE]
> All pipeline state is automatically serialized to disk so sessions survive crashes or intentional exits. Resume by re-launching — the agent picks up where you left off.

<p align="right">(<a href="#top">back to top ↑</a>)</p>

---

## Usage Flow

### End-to-End Interaction Sequence

```mermaid
sequenceDiagram
    participant Dev as 👤 Developer
    participant GUI as 🖥️ PyQt6 GUI
    participant Agent as 🤖 ML Agent
    participant Pipeline as 📋 Workflow
    participant MLflow as 🧪 MLflow
    participant DVC as 🗃️ DVC

    Dev->>GUI: Launch application
    GUI->>Agent: Initialize agent session
    Agent->>Pipeline: Load or create workflow state
    Pipeline-->>Agent: Current step (e.g. Step 1: Data Collection)
    Agent-->>GUI: Display step + recommendations

    Dev->>GUI: Load dataset
    GUI->>Pipeline: execute_step(data_collection)
    Pipeline->>DVC: Track raw data file
    DVC-->>Pipeline: ✅ data versioned
    Pipeline-->>GUI: Step complete → advance to Step 2

    Dev->>GUI: Run model training (Step 7)
    GUI->>Pipeline: execute_step(model_training)
    Pipeline->>MLflow: log_params(), log_metrics()
    MLflow-->>Pipeline: Run ID logged
    Pipeline-->>GUI: EnhancedResult{timing, metrics, recommendations}
    GUI-->>Dev: Display results + performance category
```

<p align="right">(<a href="#top">back to top ↑</a>)</p>

---

## Algorithm Coverage

### Supported Algorithm Distribution

```mermaid
pie title Algorithm Coverage by Category
    "Supervised Classification" : 40
    "Supervised Regression" : 30
    "Ensemble Methods" : 20
    "Unsupervised (planned)" : 10
```

| Category | Algorithms | Status |
|----------|-----------|--------|
| Supervised Classification | Decision Tree, Random Forest, SVM, KNN, Logistic Regression | ✅ Stable |
| Supervised Regression | Linear Regression, Decision Tree Regressor, Random Forest Regressor | ✅ Stable |
| Ensemble Methods | Random Forest, XGBoost, LightGBM | ✅ Stable |
| Unsupervised Clustering | K-Means, DBSCAN | 🟡 Planned |
| Neural Networks | scikit-learn MLPClassifier | 🟡 Planned |

<p align="right">(<a href="#top">back to top ↑</a>)</p>

---

## Technology Stack

| Technology | Purpose | Why Chosen | Alternatives Considered |
|------------|---------|------------|------------------------|
| **Python 3.8+** | Core runtime | Ubiquitous ML ecosystem, broad OS support | Julia, R |
| **scikit-learn** | ML algorithms | Battle-tested, consistent API, rich estimator library | PyTorch, TensorFlow |
| **XGBoost / LightGBM** | Gradient boosting | State-of-the-art tabular performance | CatBoost |
| **PyQt6** | Desktop GUI | Native look/feel, rich widget set, Linux/Win/Mac | Tkinter, Dear PyGui |
| **MLflow** | Experiment tracking | Self-hostable, rich UI, scikit-learn autolog | Weights & Biases, Neptune |
| **DVC** | Data versioning | Git-native, storage-agnostic, pipeline support | LakeFS, Pachyderm |
| **Docker** | Containerization | Reproducible GUI environment, CI isolation | Podman |
| **pytest** | Testing | Fixture system, coverage plugins, hypothesis | unittest |
| **loguru** | Logging | Structured logs, rotation, zero-boilerplate | standard logging |
| **Typer + Rich** | CLI | Auto-help generation, colored output | Click, argparse |

<p align="right">(<a href="#top">back to top ↑</a>)</p>

---

## Setup & Installation

### Prerequisites

- Python **3.8 – 3.12**
- Git
- Docker *(optional, for containerized GUI)*
- A display server *(X11 or Wayland for GUI)*

### Clone & Install

```bash
git clone https://github.com/hkevin01/Machine-Learning-Model.git
cd Machine-Learning-Model
```

**Linux / macOS:**

```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
# Dev + ML + viz extras
pip install -r requirements-dev.txt
```

**Windows:**

```batch
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
```

### Environment Variables

Copy the example environment file and configure as needed:

```bash
cp .env.example .env
```

```env
# .env
MLFLOW_TRACKING_URI=http://localhost:5000
MLFLOW_EXPERIMENT_NAME=default
```

### Verify Setup

```bash
python scripts/validate_setup.py
```

> [!TIP]
> Run `make mlflow-ui` after installing dev dependencies to open the MLflow experiment dashboard at [http://localhost:5000](http://localhost:5000).

<p align="right">(<a href="#top">back to top ↑</a>)</p>

---

## Usage

### Option 1 — Agent Mode (Recommended)

```bash
# Linux / macOS
./run_agent.sh

# Windows
run_agent.bat
```

The agent launches an interactive CLI + GUI session and guides you through all 13 pipeline steps.

### Option 2 — PyQt6 GUI

```bash
# Unified launcher (Docker or local)
./run.sh               # Launch GUI in Docker
./run.sh --local       # Launch GUI natively
./run.sh --headless    # Headless import smoke-test
./run.sh --rebuild     # Force rebuild Docker image
./run.sh --healthcheck # Environment & ML diagnostics
```

### Option 3 — CLI

```bash
python -m machine_learning_model --help
```

### Option 4 — Python API

```python
from machine_learning_model.workflow.ml_agent import MLAgent

agent = MLAgent()
agent.run()  # Starts the guided 13-step pipeline
```

**Enhanced Algorithm Output:**

```python
from machine_learning_model.supervised.random_forest import run_algorithm

result = run_algorithm("Random Forest", "classification", spec)
print(f"Execution Time : {result.execution_time:.4f}s")
print(f"Performance    : {result.performance_summary}")   # "Accuracy: 0.934 (Excellent)"
print(f"Recommendations: {result.recommendations}")       # ["Try cross-validation", ...]
```

<p align="right">(<a href="#top">back to top ↑</a>)</p>

---

## Core Capabilities

### 🤖 Agent Mode Pipeline

The ML Agent executes a deterministic 13-step workflow. Each step is independently resumable:

| # | Step | Description |
|---|------|-------------|
| 1 | **Data Collection** | Automated dataset loading and schema validation |
| 2 | **Data Preprocessing** | Cleaning, null handling, encoding, type coercion |
| 3 | **Exploratory Data Analysis** | Automated statistical summary and distribution plots |
| 4 | **Feature Engineering** | Scaling, polynomial features, selection |
| 5 | **Data Splitting** | Stratified train / validation / test splitting |
| 6 | **Algorithm Selection** | Automatic algorithm recommendation based on data profile |
| 7 | **Model Training** | Multi-algorithm training with MLflow logging |
| 8 | **Model Evaluation** | Accuracy, F1, ROC-AUC, R², MSE with visual reports |
| 9 | **Hyperparameter Tuning** | Grid/random search with cross-validation |
| 10 | **Model Deployment** | Pickle + ONNX export, production-ready persistence |
| 11 | **Monitoring** | Drift detection and continuous learning hooks |
| 12 | **Experiment Tracking** | MLflow run comparison and artifact logging |
| 13 | **Data Versioning** | DVC pipeline for fully reproducible data & model history |

### 📊 Enhanced Algorithm Results

Every algorithm execution returns an `EnhancedResult` object:

```python
@dataclass
class EnhancedResult:
    execution_time: float          # Precise wall-clock timing
    model_params: dict             # Full hyperparameter configuration
    performance_summary: str       # "Accuracy: 0.934 (Excellent)"
    recommendations: list[str]     # Context-aware next-step suggestions
    extended_metrics: dict         # AUC, F1-macro, confusion matrix, etc.
    model_insights: dict           # Algorithm-specific info (feature importances, etc.)
```

> [!WARNING]
> Performance categories (Excellent/Good/Fair/Poor) are heuristic thresholds. Always validate against your domain's acceptable error bounds before deployment.

### 🖥️ PyQt6 GUI Features

- Real-time pipeline step progress tracker
- Side-by-side algorithm comparison panel
- Integrated log viewer with severity filtering
- Decision boundary and feature importance charts
- Keyboard shortcuts for power users:
  - Press <kbd>Ctrl</kbd>+<kbd>R</kbd> to run the current pipeline step
  - Press <kbd>Ctrl</kbd>+<kbd>N</kbd> to advance to the next step
  - Press <kbd>Ctrl</kbd>+<kbd>S</kbd> to save workflow state

<p align="right">(<a href="#top">back to top ↑</a>)</p>

---

## Experiment Tracking

MLflow is integrated into Steps 7–9 of the pipeline. Enable it with:

```bash
pip install -r requirements-dev.txt
make mlflow-ui        # Opens http://localhost:5000
```

Configure in `.env`:

```env
MLFLOW_TRACKING_URI=http://localhost:5000
MLFLOW_EXPERIMENT_NAME=default
```

When enabled, all built-in algorithms automatically log:
- Hyperparameters (`log_params`)
- Evaluation metrics (`log_metrics`)
- Feature importances (`log_artifact`)
- Trained model artifacts (`mlflow.sklearn.log_model`)

<p align="right">(<a href="#top">back to top ↑</a>)</p>

---

## Data Versioning

A minimal DVC pipeline is defined in `dvc.yaml` with two stages: **prepare** and **train**.

```bash
pip install -r requirements-dev.txt
make dvc-init
dvc repro              # Executes the full pipeline
```

Add a remote storage backend (optional):

```bash
dvc remote add -d origin <remote-url>   # S3, GCS, SSH, local path
dvc push
```

<p align="right">(<a href="#top">back to top ↑</a>)</p>

---

## Roadmap

```mermaid
gantt
    title Machine Learning Model — Roadmap
    dateFormat  YYYY-MM-DD
    section Foundation
        Core pipeline & agent mode    :done,    f1, 2025-01-01, 2025-06-01
        PyQt6 GUI                     :done,    f2, 2025-04-01, 2025-08-01
        MLflow + DVC integration      :done,    f3, 2025-06-01, 2025-10-01
    section Enhancement
        Enhanced algorithm results    :done,    e1, 2025-09-01, 2026-01-01
        Docker GUI support            :done,    e2, 2025-11-01, 2026-02-01
        Hyperparameter tuning engine  :active,  e3, 2026-01-01, 2026-05-01
    section Expansion
        Unsupervised algorithms       :         x1, 2026-05-01, 2026-08-01
        Neural network support        :         x2, 2026-06-01, 2026-10-01
        REST API / model serving      :         x3, 2026-08-01, 2026-12-01
```

| Phase | Goals | Target | Status |
|-------|-------|--------|--------|
| **Foundation** | Core pipeline, Agent Mode, PyQt6 GUI | Q2 2025 | ✅ Complete |
| **Enhancement** | Enhanced results, Docker, MLflow/DVC | Q1 2026 | ✅ Complete |
| **Tuning** | Hyperparameter engine, drift monitoring | Q2 2026 | 🟡 In Progress |
| **Expansion** | Unsupervised algorithms, neural nets | Q3 2026 | ⭕ Planned |
| **Serving** | REST API, model serving, cloud export | Q4 2026 | ⭕ Planned |

<p align="right">(<a href="#top">back to top ↑</a>)</p>

---

## Development Status

| Version | Stability | Test Coverage | Known Limitations |
|---------|-----------|--------------|-------------------|
| 0.1.0 | Alpha | Growing | macOS untested, neural nets planned |

### Testing

```bash
# Run full test suite
python -m pytest tests/ -v

# With coverage report
python -m pytest tests/ --cov=src/machine_learning_model --cov-report=html

# Cross-platform compatibility
python -m pytest tests/test_platform_compatibility.py -v

# Linux / macOS convenience script
./scripts/run_comprehensive_tests.sh
```

**Development Tools:**

| Tool | Purpose |
|------|---------|
| `pytest` + `pytest-cov` | Test runner and coverage |
| `black` | Code formatting |
| `isort` | Import ordering |
| `flake8` | Linting |
| `mypy` | Static type checking |
| `ruff` | Fast linting |
| `pre-commit` | Git hook automation |
| `commitizen` | Conventional commits |

<p align="right">(<a href="#top">back to top ↑</a>)</p>

---

## Platform Support

| Platform | Support Level | Notes |
|----------|--------------|-------|
| ✅ Linux (Ubuntu 18.04+) | Full | Primary development target |
| ✅ Windows 10/11 | Full | Batch scripts provided |
| ⚠️ macOS | Basic | Untested — use Linux scripts |

---

## Contributing

1. Fork the repository
2. Create a feature branch: `git checkout -b feat/my-feature`
3. Commit using conventional commits: `git commit -m "feat: add new algorithm"`
4. Ensure tests pass: `./scripts/run_comprehensive_tests.sh`
5. Open a Pull Request

<details>
<summary>📋 Detailed Contribution Guidelines</summary>

### Code Style

- **Formatter**: `black` — run `black src/ tests/` before committing
- **Imports**: `isort` — run `isort src/ tests/`
- **Linting**: `flake8 src/ tests/`
- **Type hints**: all public functions must have type annotations

### Testing Requirements

- New features require unit tests in `tests/`
- Bug fixes require a regression test
- Run `pytest tests/ --cov=src/machine_learning_model` and ensure coverage does not decrease

### Branch Naming

| Type | Pattern | Example |
|------|---------|---------|
| Feature | `feat/*` | `feat/add-kmeans` |
| Bug fix | `fix/*` | `fix/workflow-resume` |
| Documentation | `docs/*` | `docs/update-readme` |
| Chore | `chore/*` | `chore/bump-deps` |

### Commit Format

Follow [Conventional Commits](https://www.conventionalcommits.org/):

```
feat(agent): add drift detection to monitoring step
fix(gui): resolve PyQt6 thread crash on large datasets
docs(readme): add mermaid architecture diagram
```

</details>

<details>
<summary>🐳 Docker Development Workflow</summary>

```bash
# Build GUI image
docker build -f Dockerfile.gui -t ml-model-gui .

# Run with X11 forwarding (Linux)
docker run -e DISPLAY=$DISPLAY \
           -v /tmp/.X11-unix:/tmp/.X11-unix \
           ml-model-gui

# Use docker-compose
docker-compose up
```

</details>

<details>
<summary>📦 Full Dependency List</summary>

**Core (`requirements.txt`)**
- `numpy`, `pandas` — data manipulation
- `scikit-learn`, `xgboost`, `lightgbm` — ML algorithms
- `matplotlib`, `seaborn`, `plotly` — visualization
- `PyQt6` — desktop GUI
- `loguru` — structured logging
- `python-dotenv` — environment management
- `pydantic` — data validation
- `typer`, `rich`, `click` — CLI

**Dev (`requirements-dev.txt`)**
- `pytest`, `pytest-cov`, `hypothesis` — testing
- `black`, `isort`, `flake8`, `ruff`, `mypy` — code quality
- `pre-commit`, `commitizen` — git automation
- `mlflow` — experiment tracking
- `dvc` — data versioning
- `mkdocs` — documentation site

</details>

<p align="right">(<a href="#top">back to top ↑</a>)</p>

---

## License

This project is licensed under the **MIT License** — you are free to use, modify, and distribute it with attribution. See the [LICENSE](LICENSE) file for full terms.

---

<div align="center">
  <p>Built with ❤️ by <a href="https://github.com/hkevin01">hkevin01</a></p>
  <p>
    <a href="https://github.com/hkevin01/Machine-Learning-Model/issues">Report Bug</a> ·
    <a href="https://github.com/hkevin01/Machine-Learning-Model/issues">Request Feature</a>
  </p>
</div>
