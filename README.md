# Machine Learning Framework

A comprehensive machine learning framework with implementations of supervised, unsupervised, and semi-supervised learning algorithms.

## 🚀 Quick Start

### Windows Setup
```batch
# Clone the repository
git clone <repository-url>
cd "Machine Learning Model"

# Run Windows setup script
scripts\setup_windows.bat

# Launch GUI
scripts\run_gui_windows.bat
```

### Ubuntu/Linux Setup
```bash
# Clone the repository
git clone <repository-url>
cd "Machine Learning Model"

# Make scripts executable
chmod +x scripts/*.sh

# Run Ubuntu setup script
./scripts/setup_ubuntu.sh

# Launch GUI
./scripts/run_gui.sh
```

## 🔧 Manual Setup

### Prerequisites
- Python 3.8 or higher
- pip package manager
- Git

### Installation Steps

1. **Create Virtual Environment**
   ```bash
   # Windows
   python -m venv venv
   venv\Scripts\activate.bat
   
   # Ubuntu/Linux
   python3 -m venv venv
   source venv/bin/activate
   ```

2. **Install Dependencies**
   ```bash
   pip install numpy pandas scikit-learn matplotlib seaborn plotly
   pip install pytest pytest-cov black isort mypy flake8  # Dev dependencies
   ```

3. **Validate Setup**
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
```
Machine Learning Model/
├── src/machine_learning_model/
│   ├── data/              # Data loading and validation
│   ├── supervised/        # Supervised learning algorithms
│   ├── gui/              # PyQt6 GUI application
│   └── main.py           # Main application entry
├── data/                 # Datasets (raw, processed)
├── tests/                # Test suite
├── scripts/              # Utility scripts
├── run_gui.py           # GUI launcher script
└── requirements.txt     # Dependencies
```

## Development Tools

- **Testing**: pytest with coverage
- **Linting**: flake8
- **Formatting**: black
- **Import Sorting**: isort
- **GUI**: PyQt6
- **ML**: scikit-learn, xgboost, matplotlib, seaborn, plotly

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Run tests: `./scripts/run_comprehensive_tests.sh`
5. Submit a pull request

## License

This project is licensed under the MIT License - see the LICENSE file for details.
