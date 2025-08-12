# Machine Learning Framework - Agent Mode 🤖

A comprehensive, **agent-based machine learning framework** with intelligent workflow guidance and complete pipeline automation. Features both traditional algorithm exploration and AI-guided workflow navigation.

## 🌟 Key Features

### 🤖 **Agent Mode** (NEW!)
- **Intelligent ML Agent**: AI-powered assistant that guides you through the complete ML pipeline
- **Step-by-Step Workflow**: Automated progression from data collection to model deployment
- **Smart Recommendations**: Context-aware suggestions for each workflow step
- **Interactive Navigator**: Comprehensive GUI with real-time progress tracking
- **State Persistence**: Automatic saving/loading of workflow progress

### 🔬 **Traditional Mode**
- **Algorithm Explorer**: Individual algorithm testing and comparison
- **Performance Visualization**: Decision boundaries, feature importance, metrics
- **Educational Examples**: Complete learning resources with explanations

### 📊 **Complete ML Pipeline**
1. **Data Collection** → Automated dataset loading and validation
2. **Data Preprocessing** → Intelligent cleaning, encoding, and transformation
3. **Exploratory Data Analysis** → Automated visualization and statistical analysis
4. **Feature Engineering** → Smart feature scaling and selection
5. **Data Splitting** → Intelligent train/validation/test splitting
6. **Algorithm Selection** → Automatic algorithm recommendation
7. **Model Training** → Automated training with multiple algorithms
8. **Model Evaluation** → Comprehensive performance analysis
9. **Hyperparameter Tuning** → Automated optimization framework
10. **Model Deployment** → Production-ready model persistence
11. **Monitoring** → Continuous learning and drift detection
12. **Experiment Tracking** → MLflow integrated lightweight logging
13. **Data Versioning** → DVC pipeline for reproducible data & models

## ✨ Enhanced Algorithm Results (NEW!)

All algorithms now provide rich, detailed output including:

- **⏱️ Execution Timing**: Precise performance measurement for each algorithm run
- **🧠 Model Parameters**: Complete hyperparameter and configuration details
- **📊 Performance Analysis**: Intelligent performance categorization (Excellent/Good/Fair/Poor)
- **💡 Smart Recommendations**: Context-aware suggestions based on results:
  - Feature engineering suggestions for low performance
  - Overfitting warnings for suspiciously high accuracy
  - Optimization recommendations for slow execution
  - Cross-validation reminders and ensemble method suggestions
- **📈 Detailed Metrics**: Extended metrics beyond basic accuracy/MSE
- **🔍 Model Insights**: Algorithm-specific information (cluster counts, explained variance, etc.)

### Example Enhanced Output:
```python
result = run_algorithm("Linear Regression", "regression", spec)
print(f"Execution Time: {result.execution_time:.4f}s")
print(f"Performance: {result.performance_summary}")  # "R² score: 0.847 (Good fit)"
print(f"Recommendations: {result.recommendations}")  # ["Validate results with cross-validation", ...]
```

## 🚀 Quick Start - Agent Mode

### Option 1: Launch Agent Mode (Recommended)
```bash
# Clone and setup
git clone <repository-url>
cd "Machine Learning Model"

# Launch agent mode directly
./run_agent.sh        # Linux/Mac
# or
run_agent.bat         # Windows
```

### Option 2: Traditional GUI
```bash
# Launch traditional algorithm explorer
./run_gui.sh          # Linux/Mac
# or
scripts\run_gui_windows.bat  # Windows
```

## 🛠️ Installation

### Quick Setup Scripts

**Windows:**

```batch
scripts\setup_windows.bat
```

**Ubuntu/Linux:**

```bash
chmod +x scripts/*.sh
./scripts/setup_ubuntu.sh
python3 -m venv venv
source venv/bin/activate
```

### Manual Dependency Installation

```bash
pip install -r requirements.txt
# Dev extras
pip install -r requirements-dev.txt
```

### Validate Setup

```bash
python scripts/validate_setup.py
```

## 🧪 Testing

### Run All Tests

```bash
# Windows
scripts\run_tests_windows.bat

# Ubuntu/Linux
./scripts/run_tests.sh

# Manual
python -m pytest tests/ -v
```

### Cross-Platform Compatibility Tests

```bash
python -m pytest tests/test_platform_compatibility.py -v
```

## 🖥️ Platform Support

### ✅ Windows

- Windows 10/11
- Python 3.8+
- Batch scripts for automation
- GUI support with tkinter

### ✅ Ubuntu/Linux

- Ubuntu 18.04+, other Linux distributions
- Python 3.8+
- Shell scripts for automation
- GUI support with tkinter

### ⚠️ macOS

- Basic support (not fully tested)
- Use Linux/Unix scripts

## 📁 Project Structure

```text
Machine Learning Model/
├── src/machine_learning_model/
│   ├── data/              # Data loading and validation
│   ├── supervised/        # Supervised learning algorithms
│   ├── gui/              # PyQt6 GUI application with enhanced results
│   └── main.py           # Main application entry
├── data/                 # Datasets (raw, processed)
├── docs/                 # Documentation (PROJECT_CLEANUP_SUMMARY.md, PYQT6_GUI_GUIDE.md)
├── config/               # Configuration files
│   ├── .flake8           # Linting configuration
│   ├── mypy.ini          # Type checking configuration
│   └── pytest.ini       # Testing configuration
├── tests/                # Test suite
├── scripts/              # Organized utility scripts
│   ├── gui/              # GUI launchers & demos (run_gui.py, demo_pyqt6_gui.py)
│   ├── agent/            # Agent mode scripts (run_agent.sh, quick_test_agent.py)
│   ├── docker/           # Docker wrapper scripts (run.sh)
│   ├── env/              # Environment activation helpers (activate_venv.sh)
│   ├── testing/          # Test scripts (test_*.py, validate_enhanced_algorithms.py)
│   └── output/           # Test outputs and artifacts
├── run_gui.py            # Backward-compatible stub → scripts/gui/run_gui.py
├── run_agent.sh          # Backward-compatible stub → scripts/agent/run_agent.sh
├── activate_venv.sh      # Backward-compatible stub → scripts/env/activate_venv.sh
├── .flake8               # Backward-compatible stub → config/.flake8
├── mypy.ini              # Backward-compatible stub → config/mypy.ini
├── pytest.ini           # Backward-compatible stub → config/pytest.ini
└── requirements.txt     # Dependencies
```

## Development Tools

- **Testing**: pytest with coverage, Hypothesis property-based tests
- **Linting**: flake8
- **Formatting**: black
- **Import Sorting**: isort
- **GUI**: PyQt6 (container includes DejaVu + Noto Emoji fonts for icon rendering)
- **ML**: scikit-learn, xgboost, matplotlib, seaborn, plotly

## Experiment Tracking (MLflow)

Enable MLflow by installing dev dependencies and (optionally) starting a local UI:

```bash
pip install -r requirements-dev.txt
make mlflow-ui  # opens at http://localhost:5000
```

Configure via `.env` (copy from `.env.example`):

```
MLFLOW_TRACKING_URI=http://localhost:5000
MLFLOW_EXPERIMENT_NAME=default
When enabled, built-in algorithms log params & feature importances automatically.

## Data Versioning (DVC)

A minimal pipeline is defined in `dvc.yaml` (stages: prepare, train):

```bash
pip install -r requirements-dev.txt
make dvc-init
dvc repro
```

Add a remote (optional):

```bash
dvc remote add -d origin <remote-url>
dvc push
```

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Run tests: `./scripts/run_comprehensive_tests.sh`
5. Submit a pull request

## License

This project is licensed under the MIT License - see the LICENSE file for details.
