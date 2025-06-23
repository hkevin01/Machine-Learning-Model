"""Main PyQt GUI window for machine learning framework exploration."""

import sys
import traceback
from typing import Any, Dict, Optional

import numpy as np
import pandas as pd

try:
    from PyQt6.QtCore import Qt, QThread, QTimer, pyqtSignal
    from PyQt6.QtGui import QColor, QFont, QIcon, QPainter, QPalette, QPixmap
    from PyQt6.QtWidgets import (
        QApplication,
        QComboBox,
        QDoubleSpinBox,
        QFrame,
        QGridLayout,
        QGroupBox,
        QHBoxLayout,
        QLabel,
        QLineEdit,
        QMainWindow,
        QMessageBox,
        QProgressBar,
        QPushButton,
        QScrollArea,
        QSpinBox,
        QSplitter,
        QTableWidget,
        QTableWidgetItem,
        QTabWidget,
        QTextEdit,
        QVBoxLayout,
        QWidget,
    )
except ImportError:
    print("PyQt6 not available. Please install PyQt6 for GUI functionality.")
    sys.exit(1)

from ..data.loaders import load_california_housing, load_iris_dataset, load_wine_dataset
from ..data.validators import (
    validate_housing_dataset,
    validate_iris_dataset,
    validate_wine_dataset,
)
from ..supervised.decision_tree import DecisionTreeClassifier
from ..supervised.random_forest import RandomForestClassifier

# TODO: Import SVMClassifier, XGBoostClassifier when implemented


class ModelTrainingThread(QThread):
    """Thread for model training to prevent GUI freezing."""
    
    training_complete = pyqtSignal(dict)
    training_error = pyqtSignal(str)
    
    def __init__(self, model_class, model_params, X, y):
        super().__init__()
        self.model_class = model_class
        self.model_params = model_params
        self.X = X
        self.y = y
    
    def run(self):
        try:
            # Create and train model
            model = self.model_class(**self.model_params)
            model.fit(self.X, self.y)
            
            # Evaluate model
            results = model.evaluate(self.X, self.y)
            model_info = model.get_model_info()
            
            self.training_complete.emit({
                'model': model,
                'results': results,
                'model_info': model_info
            })
        except Exception as e:
            self.training_error.emit(str(e))


class DatasetTab(QWidget):
    """Tab for dataset exploration and validation."""
    
    def __init__(self):
        super().__init__()
        self.datasets = {
            'Iris': load_iris_dataset,
            'Wine': load_wine_dataset,
            'California Housing': load_california_housing
        }
        self.validators = {
            'Iris': validate_iris_dataset,
            'Wine': validate_wine_dataset,
            'California Housing': validate_housing_dataset
        }
        self.current_data = None
        self.init_ui()
    
    def init_ui(self):
        """Initialize the user interface."""
        layout = QVBoxLayout()
        
        # Dataset selection
        dataset_group = QGroupBox("Dataset Selection")
        dataset_layout = QHBoxLayout()
        
        self.dataset_combo = QComboBox()
        self.dataset_combo.addItems(list(self.datasets.keys()))
        self.dataset_combo.currentTextChanged.connect(self.load_dataset)
        
        self.load_btn = QPushButton("Load Dataset")
        self.load_btn.clicked.connect(self.load_dataset)
        
        dataset_layout.addWidget(QLabel("Select Dataset:"))
        dataset_layout.addWidget(self.dataset_combo)
        dataset_layout.addWidget(self.load_btn)
        dataset_layout.addStretch()
        dataset_group.setLayout(dataset_layout)
        
        # Dataset info
        info_group = QGroupBox("Dataset Information")
        info_layout = QVBoxLayout()
        
        self.info_text = QTextEdit()
        self.info_text.setMaximumHeight(150)
        self.info_text.setReadOnly(True)
        info_layout.addWidget(self.info_text)
        info_group.setLayout(info_layout)
        
        # Validation results
        validation_group = QGroupBox("Data Validation")
        validation_layout = QVBoxLayout()
        
        self.validation_text = QTextEdit()
        self.validation_text.setMaximumHeight(200)
        self.validation_text.setReadOnly(True)
        validation_layout.addWidget(self.validation_text)
        validation_group.setLayout(validation_layout)
        
        # Data preview
        preview_group = QGroupBox("Data Preview")
        preview_layout = QVBoxLayout()
        
        self.data_table = QTableWidget()
        self.data_table.setMaximumHeight(300)
        preview_layout.addWidget(self.data_table)
        preview_group.setLayout(preview_layout)
        
        # Add all groups to main layout
        layout.addWidget(dataset_group)
        layout.addWidget(info_group)
        layout.addWidget(validation_group)
        layout.addWidget(preview_group)
        
        self.setLayout(layout)
        
        # Load initial dataset
        self.load_dataset()
    
    def load_dataset(self):
        """Load the selected dataset."""
        try:
            dataset_name = self.dataset_combo.currentText()
            load_func = self.datasets[dataset_name]
            
            self.current_data = load_func()
            
            # Update info
            self.update_dataset_info()
            
            # Update validation
            self.update_validation_results()
            
            # Update preview
            self.update_data_preview()
            
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to load dataset: {str(e)}")
    
    def update_dataset_info(self):
        """Update dataset information display."""
        if self.current_data is None:
            return
        
        info = f"""
Dataset: {self.dataset_combo.currentText()}
Shape: {self.current_data.shape}
Columns: {list(self.current_data.columns)}
Data Types:
{self.current_data.dtypes.to_string()}

First 5 rows:
{self.current_data.head().to_string()}
        """
        self.info_text.setText(info)
    
    def update_validation_results(self):
        """Update validation results display."""
        if self.current_data is None:
            return
        
        try:
            dataset_name = self.dataset_combo.currentText()
            validator_func = self.validators[dataset_name]
            
            validation_results = validator_func(self.current_data)
            
            # Format validation results
            validation_text = "Validation Results:\n\n"
            
            for result in validation_results:
                status = "✅ PASS" if result.is_valid else "❌ FAIL"
                validation_text += f"{status}: {result.message}\n"
                if result.details:
                    validation_text += f"    Details: {result.details}\n"
                validation_text += "\n"
            
            self.validation_text.setText(validation_text)
            
        except Exception as e:
            self.validation_text.setText(f"Validation error: {str(e)}")
    
    def update_data_preview(self):
        """Update data preview table."""
        if self.current_data is None:
            return
        
        # Set up table
        self.data_table.setRowCount(min(10, len(self.current_data)))
        self.data_table.setColumnCount(len(self.current_data.columns))
        self.data_table.setHorizontalHeaderLabels(self.current_data.columns)
        
        # Populate table
        for i in range(min(10, len(self.current_data))):
            for j, col in enumerate(self.current_data.columns):
                value = str(self.current_data.iloc[i, j])
                self.data_table.setItem(i, j, QTableWidgetItem(value))


class SupervisedLearningTab(QWidget):
    """Tab for supervised learning algorithms."""
    
    def __init__(self):
        super().__init__()
        self.current_model = None
        self.training_thread = None
        self.init_ui()
    
    def init_ui(self):
        """Initialize the user interface."""
        layout = QVBoxLayout()
        
        # Model selection
        model_group = QGroupBox("Model Selection")
        model_layout = QHBoxLayout()
        
        self.model_combo = QComboBox()
        self.model_combo.addItems(["Decision Tree", "Random Forest", "SVM", "XGBoost"])
        self.model_combo.currentTextChanged.connect(self.update_model_params)
        
        model_layout.addWidget(QLabel("Select Model:"))
        model_layout.addWidget(self.model_combo)
        model_layout.addStretch()
        model_group.setLayout(model_layout)
        
        # Model parameters
        self.params_group = QGroupBox("Model Parameters")
        self.params_layout = QGridLayout()
        self.params_group.setLayout(self.params_layout)
        
        # Training controls
        training_group = QGroupBox("Training")
        training_layout = QHBoxLayout()
        
        self.train_btn = QPushButton("Train Model")
        self.train_btn.clicked.connect(self.train_model)
        
        self.progress_bar = QProgressBar()
        self.progress_bar.setVisible(False)
        
        training_layout.addWidget(self.train_btn)
        training_layout.addWidget(self.progress_bar)
        training_layout.addStretch()
        training_group.setLayout(training_layout)
        
        # Results
        results_group = QGroupBox("Results")
        results_layout = QVBoxLayout()
        
        self.results_text = QTextEdit()
        self.results_text.setReadOnly(True)
        results_layout.addWidget(self.results_text)
        results_group.setLayout(results_layout)
        
        # Add all groups to main layout
        layout.addWidget(model_group)
        layout.addWidget(self.params_group)
        layout.addWidget(training_group)
        layout.addWidget(results_group)
        
        self.setLayout(layout)
        
        # Initialize model parameters
        self.update_model_params()
    
    def update_model_params(self):
        """Update model parameter controls based on selected model."""
        # Clear existing parameters
        for i in reversed(range(self.params_layout.count())):
            self.params_layout.itemAt(i).widget().setParent(None)
        
        model_name = self.model_combo.currentText()
        
        if model_name == "Decision Tree":
            self.add_decision_tree_params()
        elif model_name == "Random Forest":
            self.add_random_forest_params()
        elif model_name == "SVM":
            self.add_svm_params()
        elif model_name == "XGBoost":
            self.add_xgboost_params()
    
    def add_decision_tree_params(self):
        """Add Decision Tree parameters."""
        row = 0
        
        # Max depth
        self.params_layout.addWidget(QLabel("Max Depth:"), row, 0)
        self.max_depth_spin = QSpinBox()
        self.max_depth_spin.setRange(1, 20)
        self.max_depth_spin.setValue(3)
        self.max_depth_spin.setSpecialValueText("None")
        self.params_layout.addWidget(self.max_depth_spin, row, 1)
        
        row += 1
        
        # Min samples split
        self.params_layout.addWidget(QLabel("Min Samples Split:"), row, 0)
        self.min_samples_split_spin = QSpinBox()
        self.min_samples_split_spin.setRange(2, 20)
        self.min_samples_split_spin.setValue(2)
        self.params_layout.addWidget(self.min_samples_split_spin, row, 1)
        
        row += 1
        
        # Criterion
        self.params_layout.addWidget(QLabel("Criterion:"), row, 0)
        self.criterion_combo = QComboBox()
        self.criterion_combo.addItems(["gini", "entropy"])
        self.params_layout.addWidget(self.criterion_combo, row, 1)
    
    def add_random_forest_params(self):
        """Add Random Forest parameters."""
        row = 0
        
        # N estimators
        self.params_layout.addWidget(QLabel("Number of Trees:"), row, 0)
        self.n_estimators_spin = QSpinBox()
        self.n_estimators_spin.setRange(10, 500)
        self.n_estimators_spin.setValue(100)
        self.params_layout.addWidget(self.n_estimators_spin, row, 1)
        
        row += 1
        
        # Max depth
        self.params_layout.addWidget(QLabel("Max Depth:"), row, 0)
        self.max_depth_spin = QSpinBox()
        self.max_depth_spin.setRange(1, 20)
        self.max_depth_spin.setValue(5)
        self.max_depth_spin.setSpecialValueText("None")
        self.params_layout.addWidget(self.max_depth_spin, row, 1)
        
        row += 1
        
        # Criterion
        self.params_layout.addWidget(QLabel("Criterion:"), row, 0)
        self.criterion_combo = QComboBox()
        self.criterion_combo.addItems(["gini", "entropy"])
        self.params_layout.addWidget(self.criterion_combo, row, 1)
    
    def add_svm_params(self):
        """Add SVM parameters."""
        row = 0
        
        # C parameter
        self.params_layout.addWidget(QLabel("C Parameter:"), row, 0)
        self.c_param_spin = QDoubleSpinBox()
        self.c_param_spin.setRange(0.1, 100.0)
        self.c_param_spin.setValue(1.0)
        self.c_param_spin.setDecimals(2)
        self.params_layout.addWidget(self.c_param_spin, row, 1)
        
        row += 1
        
        # Kernel
        self.params_layout.addWidget(QLabel("Kernel:"), row, 0)
        self.kernel_combo = QComboBox()
        self.kernel_combo.addItems(["rbf", "linear", "poly", "sigmoid"])
        self.params_layout.addWidget(self.kernel_combo, row, 1)
    
    def add_xgboost_params(self):
        """Add XGBoost parameters."""
        row = 0
        
        # N estimators
        self.params_layout.addWidget(QLabel("Number of Trees:"), row, 0)
        self.n_estimators_spin = QSpinBox()
        self.n_estimators_spin.setRange(10, 500)
        self.n_estimators_spin.setValue(100)
        self.params_layout.addWidget(self.n_estimators_spin, row, 1)
        
        row += 1
        
        # Learning rate
        self.params_layout.addWidget(QLabel("Learning Rate:"), row, 0)
        self.learning_rate_spin = QDoubleSpinBox()
        self.learning_rate_spin.setRange(0.01, 1.0)
        self.learning_rate_spin.setValue(0.1)
        self.learning_rate_spin.setDecimals(3)
        self.params_layout.addWidget(self.learning_rate_spin, row, 1)
        
        row += 1
        
        # Max depth
        self.params_layout.addWidget(QLabel("Max Depth:"), row, 0)
        self.max_depth_spin = QSpinBox()
        self.max_depth_spin.setRange(1, 20)
        self.max_depth_spin.setValue(3)
        self.params_layout.addWidget(self.max_depth_spin, row, 1)
    
    def train_model(self):
        """Train the selected model."""
        try:
            # Get dataset
            iris_data = load_iris_dataset()
            X = iris_data.drop('species', axis=1)
            y = iris_data['species']
            
            # Get model parameters
            model_name = self.model_combo.currentText()
            params = self.get_model_params(model_name)
            
            # Disable training button and show progress
            self.train_btn.setEnabled(False)
            self.progress_bar.setVisible(True)
            self.progress_bar.setRange(0, 0)  # Indeterminate progress
            
            # Start training in separate thread
            self.training_thread = ModelTrainingThread(
                self.get_model_class(model_name), params, X, y
            )
            self.training_thread.training_complete.connect(self.on_training_complete)
            self.training_thread.training_error.connect(self.on_training_error)
            self.training_thread.start()
            
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to start training: {str(e)}")
            self.reset_training_ui()
    
    def get_model_class(self, model_name: str):
        """Get the model class for the selected model."""
        if model_name == "Decision Tree":
            return DecisionTreeClassifier
        elif model_name == "Random Forest":
            return RandomForestClassifier
        # TODO: Add SVM and XGBoost when implemented
        else:
            # Placeholder fallback
            class PlaceholderModel:
                def __init__(self, **kwargs):
                    self.params = kwargs
                def fit(self, X, y):
                    pass
                def evaluate(self, X, y):
                    return {'accuracy': 0.95, 'message': 'Placeholder results'}
                def get_model_info(self):
                    return {'model_type': model_name, 'status': 'Placeholder'}
            return PlaceholderModel
    
    def get_model_params(self, model_name: str) -> Dict[str, Any]:
        """Get model parameters from UI controls."""
        params = {}
        
        if model_name == "Decision Tree":
            params['max_depth'] = self.max_depth_spin.value() if self.max_depth_spin.value() > 0 else None
            params['min_samples_split'] = self.min_samples_split_spin.value()
            params['criterion'] = self.criterion_combo.currentText()
        
        elif model_name == "Random Forest":
            params['n_estimators'] = self.n_estimators_spin.value()
            params['max_depth'] = self.max_depth_spin.value() if self.max_depth_spin.value() > 0 else None
            params['criterion'] = self.criterion_combo.currentText()
        
        elif model_name == "SVM":
            params['C'] = self.c_param_spin.value()
            params['kernel'] = self.kernel_combo.currentText()
        
        elif model_name == "XGBoost":
            params['n_estimators'] = self.n_estimators_spin.value()
            params['learning_rate'] = self.learning_rate_spin.value()
            params['max_depth'] = self.max_depth_spin.value()
        
        return params
    
    def on_training_complete(self, results: Dict[str, Any]):
        """Handle training completion."""
        self.current_model = results['model']
        
        # Display results
        results_text = f"""
Training Complete!

Model Information:
{results['model_info']}

Evaluation Results:
{results['results']}
        """
        self.results_text.setText(results_text)
        
        self.reset_training_ui()
    
    def on_training_error(self, error_msg: str):
        """Handle training error."""
        QMessageBox.critical(self, "Training Error", f"Training failed: {error_msg}")
        self.reset_training_ui()
    
    def reset_training_ui(self):
        """Reset training UI elements."""
        self.train_btn.setEnabled(True)
        self.progress_bar.setVisible(False)


class UnsupervisedLearningTab(QWidget):
    """Tab for unsupervised learning algorithms."""
    
    def __init__(self):
        super().__init__()
        self.init_ui()
    
    def init_ui(self):
        """Initialize the user interface."""
        layout = QVBoxLayout()
        
        # Placeholder content
        layout.addWidget(QLabel("Unsupervised Learning Algorithms"))
        layout.addWidget(QLabel("Coming soon: K-means, DBSCAN, PCA"))
        layout.addStretch()
        
        self.setLayout(layout)


class SemiSupervisedLearningTab(QWidget):
    """Tab for semi-supervised learning algorithms."""
    
    def __init__(self):
        super().__init__()
        self.init_ui()
    
    def init_ui(self):
        """Initialize the user interface."""
        layout = QVBoxLayout()
        
        # Placeholder content
        layout.addWidget(QLabel("Semi-Supervised Learning Algorithms"))
        layout.addWidget(QLabel("Coming soon: Label Propagation, Semi-Supervised SVM"))
        layout.addStretch()
        
        self.setLayout(layout)


class MLExplorerGUI(QMainWindow):
    """Main GUI window for machine learning framework exploration."""
    
    def __init__(self):
        super().__init__()
        self.init_ui()
    
    def init_ui(self):
        """Initialize the user interface."""
        self.setWindowTitle("🤖 Machine Learning Framework Explorer")
        self.setGeometry(100, 100, 1200, 800)
        
        # Set window icon (emoji)
        self.setWindowIcon(self.create_emoji_icon())
        
        # Create central widget and layout
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        
        # Create tab widget
        self.tab_widget = QTabWidget()
        
        # Add tabs
        self.tab_widget.addTab(DatasetTab(), "📊 Dataset Explorer")
        self.tab_widget.addTab(SupervisedLearningTab(), "🎯 Supervised Learning")
        self.tab_widget.addTab(UnsupervisedLearningTab(), "🔍 Unsupervised Learning")
        self.tab_widget.addTab(SemiSupervisedLearningTab(), "🔄 Semi-Supervised Learning")
        
        # Set up main layout
        layout = QVBoxLayout()
        layout.addWidget(self.tab_widget)
        central_widget.setLayout(layout)
        
        # Set up menu bar
        self.setup_menu_bar()
        
        # Apply styling
        self.apply_styling()
    
    def create_emoji_icon(self):
        """Create a simple emoji icon for the window."""
        # Create a simple colored icon
        pixmap = QPixmap(32, 32)
        pixmap.fill(QColor(0, 120, 212))  # Blue background
        
        # Create a simple robot-like icon
        painter = QPainter(pixmap)
        painter.setPen(QColor(255, 255, 255))  # White pen
        painter.setBrush(QColor(255, 255, 255))  # White brush
        
        # Draw a simple robot face
        painter.drawEllipse(8, 8, 16, 16)  # Head
        painter.drawRect(12, 12, 2, 2)     # Left eye
        painter.drawRect(18, 12, 2, 2)     # Right eye
        painter.drawRect(14, 18, 4, 2)     # Mouth
        
        painter.end()
        
        return QIcon(pixmap)
    
    def setup_menu_bar(self):
        """Set up the menu bar."""
        menubar = self.menuBar()
        
        # File menu
        file_menu = menubar.addMenu("📁 File")
        
        exit_action = file_menu.addAction("🚪 Exit")
        exit_action.triggered.connect(self.close)
        
        # Help menu
        help_menu = menubar.addMenu("❓ Help")
        
        about_action = help_menu.addAction("ℹ️ About")
        about_action.triggered.connect(self.show_about)
    
    def apply_styling(self):
        """Apply custom styling to the GUI."""
        # Set application style
        self.setStyleSheet("""
            QMainWindow {
                background-color: #f8f9fa;
            }
            QTabWidget::pane {
                border: 1px solid #dee2e6;
                background-color: white;
                border-radius: 4px;
            }
            QTabBar::tab {
                background-color: #e9ecef;
                color: #495057;
                padding: 10px 20px;
                margin-right: 2px;
                border-top-left-radius: 4px;
                border-top-right-radius: 4px;
                font-weight: 500;
                font-size: 12px;
            }
            QTabBar::tab:selected {
                background-color: white;
                color: #0078d4;
                border-bottom: 3px solid #0078d4;
                font-weight: 600;
            }
            QTabBar::tab:hover {
                background-color: #f8f9fa;
                color: #212529;
            }
            QGroupBox {
                font-size: 9px;
                font-weight: 600;
                color: #212529;
                border: 2px solid #dee2e6;
                border-radius: 6px;
                margin-top: 8px;
                padding-top: 8px;
            }
            QGroupBox::title {
                subcontrol-origin: margin;
                left: 10px;
                padding: 0 5px 0 5px;
                background-color: white;
            }
            QPushButton {
                background-color: #007bff;
                color: white;
                border: none;
                padding: 8px 16px;
                border-radius: 6px;
                font-size: 9px;
                font-weight: 500;
                min-height: 20px;
            }
            QPushButton:hover {
                background-color: #0056b3;
            }
            QPushButton:pressed {
                background-color: #004085;
            }
            QPushButton:disabled {
                background-color: #6c757d;
                color: #adb5bd;
            }
            QTextEdit {
                border: 1px solid #c0c0c0;
                border-radius: 3px;
                padding: 8px 12px;
                background-color: white;
                color: #212529;
                font-family: 'Consolas', 'Monaco', 'Courier New', monospace;
                font-size: 9px;
                line-height: 1.4;
                margin-right: 4px;
                selection-background-color: #0078d4;
                selection-color: white;
            }
            QTextEdit:focus {
                border-color: #0078d4;
                outline: none;
            }
            QTextEdit:disabled {
                background-color: #f5f5f5;
                color: #6c757d;
            }
            QLabel {
                color: #212529;
                font-size: 9px;
                font-weight: 500;
            }
            QComboBox {
                border: 1px solid #c0c0c0;
                border-radius: 3px;
                padding: 4px 8px;
                background-color: white;
                color: #212529;
                font-size: 9px;
                min-height: 18px;
            }
            QComboBox:focus {
                border-color: #0078d4;
                outline: none;
            }
            QComboBox:disabled {
                background-color: #f5f5f5;
                color: #6c757d;
            }
            QComboBox::drop-down {
                border: none;
                width: 20px;
                background-color: transparent;
            }
            QComboBox::down-arrow {
                image: none;
                border-left: 4px solid transparent;
                border-right: 4px solid transparent;
                border-top: 4px solid #666666;
                margin-right: 4px;
            }
            QComboBox::down-arrow:disabled {
                border-top-color: #999999;
            }
            QComboBox QAbstractItemView {
                border: 1px solid #c0c0c0;
                border-radius: 3px;
                background-color: white;
                color: #212529;
                selection-background-color: #0078d4;
                selection-color: white;
                outline: none;
                padding: 2px;
            }
            QComboBox QAbstractItemView::item {
                padding: 4px 8px;
                background-color: transparent;
                color: #212529;
                border-radius: 2px;
            }
            QComboBox QAbstractItemView::item:hover {
                background-color: #f0f0f0;
                color: #212529;
            }
            QComboBox QAbstractItemView::item:selected {
                background-color: #0078d4;
                color: white;
            }
            QTableWidget {
                border: 2px solid #dee2e6;
                border-radius: 6px;
                background-color: white;
                gridline-color: #dee2e6;
                color: #212529;
                font-size: 9px;
                padding: 4px;
                margin-right: 4px;
                alternate-background-color: #f8f9fa;
                selection-background-color: #e3f2fd;
                selection-color: #212529;
                outline: none;
            }
            QTableWidget::item {
                padding: 4px;
                border: none;
                background-color: white;
                color: #212529;
            }
            QTableWidget::item:selected {
                background-color: #e3f2fd;
                color: #212529;
            }
            QTableWidget::item:alternate {
                background-color: #f8f9fa;
                color: #212529;
            }
            QHeaderView::section {
                background-color: #f8f9fa;
                color: #212529;
                padding: 6px;
                border: 1px solid #dee2e6;
                font-size: 9px;
                font-weight: 600;
            }
            QHeaderView::section:hover {
                background-color: #e9ecef;
            }
            QHeaderView::section:pressed {
                background-color: #dee2e6;
            }
            QHeaderView {
                background-color: #f8f9fa;
                border: none;
                outline: none;
            }
            QTableCornerButton::section {
                background-color: #f8f9fa;
                border: 1px solid #dee2e6;
                border-radius: 0px;
            }
            QProgressBar {
                border: 2px solid #dee2e6;
                border-radius: 6px;
                background-color: #f8f9fa;
                color: #212529;
                font-size: 9px;
                text-align: center;
                min-height: 20px;
            }
            QProgressBar::chunk {
                background-color: #28a745;
                border-radius: 4px;
            }
            QMenuBar {
                background-color: #f8f9fa;
                color: #212529;
                border-bottom: 1px solid #dee2e6;
                font-size: 9px;
            }
            QMenuBar::item {
                background-color: transparent;
                color: #212529;
                padding: 6px 10px;
            }
            QMenuBar::item:selected {
                background-color: #e3f2fd;
                color: #212529;
            }
            QMenuBar::item:pressed {
                background-color: #dee2e6;
                color: #212529;
            }
            QMenu {
                background-color: white;
                color: #212529;
                border: 2px solid #dee2e6;
                border-radius: 6px;
                padding: 4px;
                font-size: 9px;
            }
            QMenu::item {
                background-color: transparent;
                color: #212529;
                padding: 6px 20px;
                border-radius: 4px;
            }
            QMenu::item:selected {
                background-color: #e3f2fd;
                color: #212529;
            }
            QMenu::separator {
                height: 1px;
                background-color: #dee2e6;
                margin: 4px 0px;
            }
            QScrollBar:vertical {
                background-color: #f0f0f0;
                width: 12px;
                border-radius: 6px;
                margin: 0px;
            }
            QScrollBar::handle:vertical {
                background-color: #c0c0c0;
                border-radius: 6px;
                min-height: 20px;
                margin: 1px;
            }
            QScrollBar::handle:vertical:hover {
                background-color: #a0a0a0;
            }
            QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical {
                height: 12px;
                background-color: #f0f0f0;
                border: 1px solid #d0d0d0;
                border-radius: 2px;
            }
            QScrollBar::add-line:vertical:hover, QScrollBar::sub-line:vertical:hover {
                background-color: #e0e0e0;
            }
            QScrollBar::add-page:vertical, QScrollBar::sub-page:vertical {
                background-color: #f8f9fa;
            }
            QScrollBar:horizontal {
                background-color: #f0f0f0;
                height: 12px;
                border-radius: 6px;
                margin: 0px;
            }
            QScrollBar::handle:horizontal {
                background-color: #c0c0c0;
                border-radius: 6px;
                min-width: 20px;
                margin: 1px;
            }
            QScrollBar::handle:horizontal:hover {
                background-color: #a0a0a0;
            }
            QScrollBar::add-line:horizontal, QScrollBar::sub-line:horizontal {
                width: 12px;
                background-color: #f0f0f0;
                border: 1px solid #d0d0d0;
                border-radius: 2px;
            }
            QScrollBar::add-line:horizontal:hover, QScrollBar::sub-line:horizontal:hover {
                background-color: #e0e0e0;
            }
            QScrollBar::add-page:horizontal, QScrollBar::sub-page:horizontal {
                background-color: #f8f9fa;
            }
            QLineEdit {
                border: 1px solid #c0c0c0;
                border-radius: 3px;
                padding: 4px 8px;
                background-color: white;
                color: #212529;
                font-size: 9px;
                min-height: 18px;
            }
            QLineEdit:focus {
                border-color: #0078d4;
                outline: none;
            }
            QLineEdit:disabled {
                background-color: #f5f5f5;
                color: #6c757d;
            }
            QSpinBox, QDoubleSpinBox {
                border: 1px solid #c0c0c0;
                border-radius: 3px;
                padding: 4px 8px;
                background-color: white;
                color: #212529;
                font-size: 9px;
                min-height: 18px;
                min-width: 80px;
            }
            QSpinBox:focus, QDoubleSpinBox:focus {
                border-color: #0078d4;
                outline: none;
            }
            QSpinBox:disabled, QDoubleSpinBox:disabled {
                background-color: #f5f5f5;
                color: #6c757d;
            }
            QSpinBox::up-button, QDoubleSpinBox::up-button {
                background-color: #f0f0f0;
                border: 1px solid #c0c0c0;
                border-radius: 2px;
                width: 16px;
                height: 10px;
            }
            QSpinBox::up-button:hover, QDoubleSpinBox::up-button:hover {
                background-color: #e0e0e0;
            }
            QSpinBox::down-button, QDoubleSpinBox::down-button {
                background-color: #f0f0f0;
                border: 1px solid #c0c0c0;
                border-radius: 2px;
                width: 16px;
                height: 10px;
            }
            QSpinBox::down-button:hover, QDoubleSpinBox::down-button:hover {
                background-color: #e0e0e0;
            }
            QSpinBox::up-arrow, QDoubleSpinBox::up-arrow {
                image: none;
                border-left: 3px solid transparent;
                border-right: 3px solid transparent;
                border-bottom: 3px solid #666666;
            }
            QSpinBox::down-arrow, QDoubleSpinBox::down-arrow {
                image: none;
                border-left: 3px solid transparent;
                border-right: 3px solid transparent;
                border-top: 3px solid #666666;
            }
        """)
    
    def show_about(self):
        """Show about dialog."""
        QMessageBox.about(
            self,
            "ℹ️ About ML Framework Explorer",
            "🤖 Machine Learning Framework Explorer\n\n"
            "A comprehensive GUI for exploring different machine learning algorithms "
            "and frameworks with interactive examples and validation tools.\n\n"
            "Version 1.0.0"
        )


def main():
    """Main function to run the GUI application."""
    app = QApplication(sys.argv)
    
    # Set application properties
    app.setApplicationName("ML Framework Explorer")
    app.setApplicationVersion("1.0.0")
    
    # Create and show main window
    window = MLExplorerGUI()
    window.show()
    
    # Start event loop
    sys.exit(app.exec())


if __name__ == "__main__":
    main() 