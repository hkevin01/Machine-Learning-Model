# Machine Learning Model

A comprehensive machine learning framework with interactive GUI for exploring different algorithms and datasets.

## Features

- 🤖 **Interactive GUI**: PyQt6-based interface with emoji icons
- 📊 **Dataset Explorer**: Load and validate datasets (Iris, Wine, California Housing)
- 🎯 **Supervised Learning**: Decision Trees, Random Forest (SVM, XGBoost coming soon)
- 🔍 **Unsupervised Learning**: Coming soon (K-means, DBSCAN, PCA)
- 🔄 **Semi-Supervised Learning**: Coming soon (Label Propagation, Semi-Supervised SVM)
- ✅ **Data Validation**: Comprehensive validation framework
- 🧪 **Testing**: Full test suite with coverage reporting

## Quick Start

### 1. Setup Environment

```bash
# Clone the repository
git clone <repository-url>
cd "Machine Learning Model"

# Create and activate virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### 2. Run the GUI

```bash
# Run the GUI application
python run_gui.py

# Or run directly (alternative method)
python -m src.machine_learning_model.gui.main_window
```

### 3. Run Tests

```bash
# Run comprehensive test suite
./scripts/run_comprehensive_tests.sh

# Or run tests manually
python -m pytest tests/ -v
```

## GUI Features

### 🤖 Main Window
- **Window Title**: "🤖 Machine Learning Framework Explorer"
- **Custom Icon**: Robot-themed application icon
- **Emoji Tabs**: Visual indicators for different sections

### 📊 Dataset Explorer Tab
- Load and preview datasets
- Real-time data validation
- Dataset information display
- Data quality checks

### 🎯 Supervised Learning Tab
- Model selection (Decision Tree, Random Forest)
- Parameter configuration
- Training with progress indication
- Results display and evaluation

### 🔍 Unsupervised Learning Tab
- Coming soon: K-means, DBSCAN, PCA

### 🔄 Semi-Supervised Learning Tab
- Coming soon: Label Propagation, Semi-Supervised SVM

## Project Structure

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
