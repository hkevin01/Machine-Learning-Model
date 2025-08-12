"""
Modern PyQt6-based GUI for Machine Learning Framework Explorer.
Provides categorized interface for exploring supervised, unsupervised, and semi-supervised algorithms.
"""
import sys
from typing import Dict, Any, Optional
import numpy as np

from PyQt6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, 
    QTabWidget, QListWidget, QTextEdit, QLabel, QPushButton, 
    QButtonGroup, QRadioButton, QFrame, QSplitter, QListWidgetItem
)
from PyQt6.QtCore import Qt, pyqtSignal
from PyQt6.QtGui import QFont, QIcon, QPixmap

try:
    from machine_learning_model.supervised.decision_tree import DecisionTreeClassifier, DecisionTreeRegressor
    from machine_learning_model.supervised.random_forest import RandomForestClassifier, RandomForestRegressor
except ImportError:
    # Fallback for development/testing
    class DecisionTreeClassifier:
        def fit(self, X, y): pass
        def predict(self, X): return np.zeros(len(X))
    class DecisionTreeRegressor:
        def fit(self, X, y): pass  
        def predict(self, X): return np.zeros(len(X))
    class RandomForestClassifier:
        def __init__(self, n_estimators=25): pass
        def fit(self, X, y): pass
        def predict(self, X): return np.zeros(len(X))
    class RandomForestRegressor:
        def __init__(self, n_estimators=25): pass
        def fit(self, X, y): pass
        def predict(self, X): return np.zeros(len(X))


class AlgorithmDatabase:
    """Contains comprehensive information about ML algorithms categorized by learning type."""
    
    SUPERVISED_ALGORITHMS = {
        "Linear Regression": {
            "description": "A fundamental algorithm for predicting continuous values by finding the best linear relationship between features and target variables.",
            "use_cases": "House price prediction, sales forecasting, stock price analysis, risk assessment, medical dosage prediction",
            "pros": "Simple and interpretable, fast training, no hyperparameters to tune, works well with small datasets, provides confidence intervals",
            "cons": "Assumes linear relationship, sensitive to outliers, may underfit complex patterns, requires feature scaling",
            "complexity": "Low",
            "type": "Regression",
            "status": "✅ Ready for Implementation",
            "implementation_status": "ready",
            "examples": "Predicting house prices based on size, location, and age; forecasting sales revenue from advertising spend"
        },
        "Logistic Regression": {
            "description": "Classification algorithm using the logistic function to model the probability of class membership with a sigmoid curve.",
            "use_cases": "Email spam detection, medical diagnosis, marketing response prediction, click-through rate estimation",
            "pros": "Probabilistic output, highly interpretable, handles categorical features well, fast training, built-in regularization",
            "cons": "Assumes linear decision boundary, sensitive to outliers, requires feature scaling, struggles with complex relationships",
            "complexity": "Low",
            "type": "Classification", 
            "status": "✅ Ready for Implementation",
            "implementation_status": "ready",
            "examples": "Detecting spam emails, predicting customer churn, medical diagnosis based on symptoms"
        },
        "Decision Trees": {
            "description": "Tree-like model making decisions by recursively splitting data based on feature values to maximize information gain or minimize impurity.",
            "use_cases": "Credit approval systems, medical diagnosis, feature selection, rule extraction, fraud detection",
            "pros": "Highly interpretable, handles mixed data types, no assumptions about data distribution, automatic feature selection, handles missing values",
            "cons": "Prone to overfitting, unstable with small data changes, biased toward features with more levels, can create complex trees",
            "complexity": "Medium",
            "type": "Both",
            "status": "✅ Complete - Production Ready",
            "implementation_status": "complete",
            "examples": "Credit card approval based on income and credit history, medical diagnosis trees, customer segmentation rules"
        },
        "Random Forest": {
            "description": "Ensemble method combining multiple decision trees with bootstrap aggregating (bagging) and voting/averaging to improve accuracy and reduce overfitting.",
            "use_cases": "Feature importance ranking, general-purpose prediction, biomedical research, image classification, genomics",
            "pros": "Reduces overfitting, handles missing values naturally, provides feature importance, built-in cross-validation (OOB), robust to outliers",
            "cons": "Less interpretable than single trees, can overfit with very noisy data, memory intensive, slower than single trees",
            "complexity": "Medium", 
            "type": "Both",
            "status": "✅ Complete - Production Ready",
            "implementation_status": "complete",
            "examples": "Predicting disease outcomes from patient data, ranking feature importance in genetic studies, stock market prediction"
        },
        "Support Vector Machine": {
            "description": "Finds optimal hyperplane to separate classes or predict values by maximizing margin between data points, with kernel trick for non-linear patterns.",
            "use_cases": "Text classification, image recognition, gene classification, high-dimensional data, document classification",
            "pros": "Effective in high dimensions, memory efficient, versatile with different kernels, robust to overfitting in high dimensions",
            "cons": "Slow on large datasets, sensitive to feature scaling, no probabilistic output, difficult to interpret, sensitive to noise",
            "complexity": "High",
            "type": "Both",
            "status": "🔄 Next - Starting this week",
            "implementation_status": "planned",
            "examples": "Text document classification, face recognition, protein fold prediction, handwritten digit recognition"
        },
        "XGBoost": {
            "description": "Advanced gradient boosting framework optimized for speed and performance with regularization to prevent overfitting.",
            "use_cases": "Kaggle competitions, structured data prediction, feature selection, ranking problems, large-scale machine learning",
            "pros": "State-of-the-art performance, built-in regularization, handles missing values, parallel processing, cross-validation support",
            "cons": "Many hyperparameters to tune, computationally intensive, requires careful tuning, can overfit, black box model",
            "complexity": "High",
            "type": "Both", 
            "status": "📋 Planned - Advanced Phase",
            "implementation_status": "future",
            "examples": "Kaggle competition winning models, customer lifetime value prediction, ad click prediction, risk modeling"
        },
        "Neural Networks": {
            "description": "Multi-layered networks of interconnected nodes mimicking brain neurons for complex pattern recognition and function approximation.",
            "use_cases": "Image recognition, natural language processing, speech recognition, game playing, time series forecasting",
            "pros": "Universal approximator, handles complex non-linear relationships, automatic feature learning, scalable to large datasets",
            "cons": "Requires large datasets, computationally expensive, black box, prone to overfitting, many hyperparameters",
            "complexity": "High",
            "type": "Both",
            "status": "📋 Planned - Advanced Phase", 
            "implementation_status": "future",
            "examples": "Image classification with CNNs, language translation with transformers, game AI with deep reinforcement learning"
        }
    }

    UNSUPERVISED_ALGORITHMS = {
        "K-Means Clustering": {
            "description": "Partitions data into k clusters by iteratively minimizing within-cluster sum of squares using centroid-based approach.",
            "use_cases": "Customer segmentation, image compression, market research, data compression, anomaly detection preprocessing",
            "pros": "Simple and fast, works well with spherical clusters, scales to large datasets, guaranteed convergence",
            "cons": "Must specify k beforehand, sensitive to initialization and outliers, assumes spherical clusters, struggles with varying densities",
            "complexity": "Medium",
            "type": "Clustering",
            "status": "📋 Planned - Phase 3",
            "implementation_status": "planned",
            "examples": "Customer segmentation for marketing, color quantization in images, organizing news articles by topic"
        },
        "DBSCAN": {
            "description": "Density-based clustering that groups together points in high-density areas and marks outliers in low-density regions.",
            "use_cases": "Anomaly detection, image processing, social network analysis, fraud detection, spatial data analysis",
            "pros": "Automatically determines number of clusters, handles noise and outliers well, finds arbitrary shaped clusters",
            "cons": "Sensitive to hyperparameters (eps, min_samples), struggles with varying densities, difficult to use in high dimensions",
            "complexity": "Medium",
            "type": "Clustering",
            "status": "📋 Planned - Phase 3", 
            "implementation_status": "planned",
            "examples": "Detecting fraudulent transactions, finding crime hotspots, identifying communities in social networks"
        },
        "Principal Component Analysis": {
            "description": "Dimensionality reduction technique that projects data onto principal components that capture maximum variance in the data.",
            "use_cases": "Data visualization, feature reduction, noise reduction, data compression, exploratory data analysis",
            "pros": "Reduces overfitting, speeds up training, removes multicollinearity, provides interpretable components, reduces storage",
            "cons": "Loses interpretability of original features, linear transformation only, sensitive to scaling, may lose important information",
            "complexity": "Medium",
            "type": "Dimensionality Reduction",
            "status": "📋 Planned - Phase 3",
            "implementation_status": "planned", 
            "examples": "Visualizing high-dimensional data in 2D/3D, reducing features before classification, image compression"
        },
        "Hierarchical Clustering": {
            "description": "Creates tree-like cluster hierarchy using linkage criteria, either agglomerative (bottom-up) or divisive (top-down) approach.",
            "use_cases": "Phylogenetic analysis, social network analysis, image segmentation, organizing product catalogs, taxonomy creation",
            "pros": "No need to specify number of clusters, creates interpretable hierarchy, deterministic results, handles any distance metric",
            "cons": "Computationally expensive O(n³), sensitive to noise and outliers, difficult to handle large datasets, sensitive to metric choice",
            "complexity": "High",
            "type": "Clustering", 
            "status": "📋 Planned - Phase 3",
            "implementation_status": "planned",
            "examples": "Building species evolution trees, organizing company departments, creating product category hierarchies"
        }
    }

    SEMI_SUPERVISED_ALGORITHMS = {
        "Label Propagation": {
            "description": "Graph-based algorithm that propagates labels from labeled to unlabeled data through similarity graphs using diffusion processes.",
            "use_cases": "Text classification with few labels, image annotation, social media analysis, web page classification, protein function prediction", 
            "pros": "Works effectively with few labeled examples, natural uncertainty estimation, captures data manifold structure",
            "cons": "Requires good similarity metric, computationally expensive for large graphs, sensitive to graph construction, memory intensive",
            "complexity": "High",
            "type": "Classification",
            "status": "📋 Planned - Phase 4",
            "implementation_status": "future",
            "examples": "Classifying documents with few labeled examples, image tagging with minimal supervision, social network node classification"
        },
        "Self-Training": {
            "description": "Iteratively trains on labeled data, predicts unlabeled data, adds most confident predictions to training set, and retrains the model.",
            "use_cases": "NLP with limited annotations, medical diagnosis with few expert labels, web page classification, speech recognition",
            "pros": "Simple to implement, works with any base classifier, intuitive approach, can significantly improve performance",
            "cons": "Can amplify errors, requires good confidence estimation, may drift from true distribution, sensitive to initial model quality",
            "complexity": "Medium",
            "type": "Classification",
            "status": "📋 Planned - Phase 4",
            "implementation_status": "future",
            "examples": "Email spam detection with few labeled emails, medical image diagnosis with limited expert annotations"
        },
        "Co-Training": {
            "description": "Uses two different views of data to train separate classifiers that teach each other by adding confident predictions to the other's training set.",
            "use_cases": "Web page classification, email classification, multi-modal learning, document classification, bioinformatics",
            "pros": "Leverages multiple feature views, reduces overfitting through diversity, works well with independent views",
            "cons": "Requires conditionally independent views, complex setup, sensitive to view quality, may not work without good views",
            "complexity": "High", 
            "type": "Classification",
            "status": "📋 Planned - Phase 4",
            "implementation_status": "future",
            "examples": "Web page classification using text and link features, email classification using header and body content"
        },
        "Semi-Supervised SVM": {
            "description": "Extends Support Vector Machine to work with both labeled and unlabeled data using transductive learning and margin maximization.",
            "use_cases": "Text mining, bioinformatics, computer vision with limited labels, drug discovery, gene expression analysis",
            "pros": "Leverages unlabeled data effectively, maintains SVM advantages, works well in high dimensions, principled approach",
            "cons": "Non-convex optimization problem, computationally challenging, sensitive to parameters, difficult to scale",
            "complexity": "High",
            "type": "Classification",
            "status": "📋 Planned - Phase 4", 
            "implementation_status": "future",
            "examples": "Protein function prediction, document classification with few labels, medical image analysis with limited annotations"
        }
    }


class AlgorithmListWidget(QListWidget):
    """Custom list widget for displaying algorithms with status icons and formatting."""
    
    algorithmSelected = pyqtSignal(str, dict)
    
    def __init__(self, algorithms: Dict[str, Dict[str, Any]], parent=None):
        super().__init__(parent)
        self.algorithms = algorithms
        self.setStyleSheet("""
            QListWidget {
                background-color: #f8f9fa;
                border: 1px solid #dee2e6;
                border-radius: 5px;
                font-size: 12px;
                padding: 5px;
                color: black;
            }
            QListWidget::item {
                padding: 8px;
                border-bottom: 1px solid #e9ecef;
                margin: 2px;
                border-radius: 3px;
                color: black;
            }
            QListWidget::item:hover {
                background-color: #e3f2fd;
                color: black;
            }
            QListWidget::item:selected {
                background-color: #b3d9ff;
                color: black;
            }
        """)
        self.populate_algorithms()
        
    def populate_algorithms(self):
        """Populate the list with algorithms and their status."""
        for name, info in self.algorithms.items():
            # Create status icon based on implementation status
            status = info.get("status", "")
            if "Complete" in status or "✅" in status:
                icon = "✅"
            elif "Next" in status or "🔄" in status:
                icon = "🔄"
            elif "Planned" in status or "📋" in status:
                icon = "📋"
            else:
                icon = "❓"
            
            item_text = f"{icon} {name}"
            item = QListWidgetItem(item_text)
            item.setData(Qt.ItemDataRole.UserRole, name)  # Store algorithm name
            self.addItem(item)
        
        # Connect selection change
        self.itemSelectionChanged.connect(self.on_selection_changed)
    
    def on_selection_changed(self):
        """Handle algorithm selection."""
        current_item = self.currentItem()
        if current_item:
            algorithm_name = current_item.data(Qt.ItemDataRole.UserRole)
            algorithm_info = self.algorithms[algorithm_name]
            self.algorithmSelected.emit(algorithm_name, algorithm_info)


class AlgorithmDetailWidget(QTextEdit):
    """Widget for displaying detailed algorithm information with rich formatting."""
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setReadOnly(True)
        self.setStyleSheet("""
            QTextEdit {
                background-color: white;
                border: 1px solid #dee2e6;
                border-radius: 5px;
                padding: 15px;
                font-family: 'Segoe UI', Arial, sans-serif;
                font-size: 11px;
                line-height: 1.4;
                color: black;
            }
        """)
        
    def display_algorithm_details(self, name: str, info: Dict[str, Any], category: str):
        """Display comprehensive algorithm information with formatting."""
        
        # Map categories to readable names
        category_names = {
            "supervised": "Supervised Learning",
            "unsupervised": "Unsupervised Learning", 
            "semi_supervised": "Semi-Supervised Learning"
        }
        
        # Generate implementation guide based on status
        implementation_status = info.get("implementation_status", "planned")
        if implementation_status == "complete":
            implementation_guide = """
🚀 IMPLEMENTATION AVAILABLE!

📁 Location: src/machine_learning_model/supervised/
📖 Examples: examples/supervised_examples/
🧪 Tests: tests/test_supervised/

⚡ Quick Start:
```python
from machine_learning_model.supervised import DecisionTree, RandomForest
model = DecisionTree()  # or RandomForest()
model.fit(X_train, y_train)
predictions = model.predict(X_test)
```

🔬 Use the Quick Run panel below to test with synthetic data!
"""
        elif implementation_status == "ready":
            implementation_guide = """
📋 READY FOR IMPLEMENTATION
✅ Documentation and API design complete
🔧 Implementation starting soon
📚 Research and design phase completed
"""
        else:
            implementation_guide = """
⏳ PLANNED FOR FUTURE IMPLEMENTATION
📅 Scheduled for upcoming development phases
🎯 Will be available in future releases
💡 Community contributions welcome!
"""

        # Create rich text content
        details_html = f"""
        <div style="font-family: 'Segoe UI', Arial, sans-serif; line-height: 1.6; color: black;">
            <h2 style="color: black; border-bottom: 2px solid #3498db; padding-bottom: 10px;">
                🎯 {name}
            </h2>
            
            <div style="background: #f8f9fa; padding: 10px; border-radius: 5px; margin: 10px 0; color: black;">
                <strong>Category:</strong> {category_names.get(category, category)} •
                <strong>Type:</strong> {info.get('type', 'N/A')} •
                <strong>Complexity:</strong> {info.get('complexity', 'N/A')}
            </div>

            <h3 style="color: black;">📝 Description</h3>
            <p style="color: black;">{info.get('description', 'No description available.')}</p>

            <h3 style="color: black;">🎯 Common Use Cases</h3>
            <p style="color: black;">{info.get('use_cases', 'No use cases listed.')}</p>

            <h3 style="color: black;">✅ Advantages</h3>
            <p style="color: black;">{info.get('pros', 'No advantages listed.')}</p>

            <h3 style="color: black;">❌ Disadvantages</h3>
            <p style="color: black;">{info.get('cons', 'No disadvantages listed.')}</p>

            <h3 style="color: black;">💡 Examples</h3>
            <p style="color: black;">{info.get('examples', 'No examples available.')}</p>

            <h3 style="color: black;">📈 Implementation Status</h3>
            <div style="background: #ecf0f1; padding: 10px; border-radius: 5px; color: black;">
                <strong>{info.get('status', 'Status unknown')}</strong>
            </div>

            <div style="background: #e8f5e8; padding: 15px; border-radius: 5px; margin: 15px 0; color: black;">
                <pre style="white-space: pre-wrap; font-family: monospace; margin: 0; color: black;">{implementation_guide.strip()}</pre>
            </div>

            <div style="background: #fff3cd; padding: 10px; border-radius: 5px; margin-top: 15px; color: black;">
                <h4 style="margin: 0 0 10px 0; color: black;">💡 Quick Tips:</h4>
                <ul style="margin: 0; padding-left: 20px;">
                    <li>Start with simpler algorithms (Linear/Logistic Regression) for baseline performance</li>
                    <li>Use ensemble methods (Random Forest, Gradient Boosting) for improved accuracy</li>
                    <li>Consider your data size and computational resources when choosing algorithms</li>
                    <li>Always validate your model on unseen data using proper cross-validation</li>
                    <li>Feature engineering often has more impact than algorithm choice</li>
                </ul>
            </div>
        </div>
        """
        
        self.setHtml(details_html)


class QuickRunPanel(QFrame):
    """Panel for quick algorithm testing with synthetic data."""
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.current_algorithm = None
        self.current_category = None
        self.setup_ui()
        
    def setup_ui(self):
        """Setup the quick run interface."""
        self.setStyleSheet("""
            QFrame {
                background-color: #f8f9fa;
                border: 1px solid #dee2e6;
                border-radius: 5px;
                padding: 10px;
                color: black;
            }
            QPushButton {
                background-color: #007bff;
                color: black;
                border: none;
                padding: 8px 16px;
                border-radius: 4px;
                font-weight: bold;
            }
            QPushButton:hover {
                background-color: #0056b3;
                color: black;
            }
            QPushButton:disabled {
                background-color: #6c757d;
                color: black;
            }
        """)
        
        layout = QVBoxLayout(self)
        
        # Title
        title = QLabel("🚀 Quick Run (Synthetic Data)")
        title_font = QFont()
        title_font.setBold(True)
        title_font.setPointSize(12)
        title.setFont(title_font)
        layout.addWidget(title)
        
        # Task selection
        task_layout = QHBoxLayout()
        self.task_group = QButtonGroup(self)
        
        self.classification_radio = QRadioButton("Classification")
        self.regression_radio = QRadioButton("Regression")
        self.classification_radio.setChecked(True)
        
        self.task_group.addButton(self.classification_radio, 0)
        self.task_group.addButton(self.regression_radio, 1)
        
        task_layout.addWidget(self.classification_radio)
        task_layout.addWidget(self.regression_radio)
        task_layout.addStretch()
        
        layout.addLayout(task_layout)
        
        # Run button
        self.run_button = QPushButton("Run Selected Algorithm")
        self.run_button.clicked.connect(self.run_algorithm)
        self.run_button.setEnabled(False)
        layout.addWidget(self.run_button)
        
        # Results display
        self.results_label = QLabel("Select an algorithm and click Run to test with synthetic data")
        self.results_label.setStyleSheet("""
            QLabel {
                background-color: white;
                padding: 10px;
                border: 1px solid #dee2e6;
                border-radius: 3px;
                margin-top: 5px;
            }
        """)
        self.results_label.setWordWrap(True)
        layout.addWidget(self.results_label)
        
    def set_current_algorithm(self, name: str, info: Dict[str, Any], category: str):
        """Set the currently selected algorithm."""
        self.current_algorithm = name
        self.current_category = category
        self.current_info = info
        
        # Enable run button for implemented algorithms
        implementation_status = info.get("implementation_status", "planned")
        self.run_button.setEnabled(implementation_status in ["complete", "ready"])
        
        if implementation_status == "complete":
            self.results_label.setText(f"Ready to test {name} with synthetic data. Choose task type and click Run.")
        elif implementation_status == "ready":
            self.results_label.setText(f"{name} is ready for implementation. Synthetic testing will be available soon.")
        else:
            self.results_label.setText(f"{name} is planned for future implementation. Testing not yet available.")
    
    def run_algorithm(self):
        """Execute the selected algorithm with synthetic data."""
        if not self.current_algorithm:
            return
            
        try:
            task = "classification" if self.classification_radio.isChecked() else "regression"
            n_samples = 100
            n_features = 5
            
            # Generate synthetic data
            np.random.seed(42)  # For reproducible results
            X = np.random.randn(n_samples, n_features)
            
            if task == "classification":
                # Create binary classification problem
                y = (X[:, 0] + X[:, 1] * 0.5 - X[:, 2] * 0.3 > 0).astype(int)
            else:
                # Create regression problem with some noise
                y = X[:, 0] * 0.7 + X[:, 1] * 0.3 - X[:, 2] * 0.2 + np.random.randn(n_samples) * 0.1
            
            # Select and run algorithm
            if self.current_algorithm == "Decision Trees":
                if task == "classification":
                    model = DecisionTreeClassifier()
                else:
                    model = DecisionTreeRegressor()
                    
            elif self.current_algorithm == "Random Forest":
                if task == "classification":
                    model = RandomForestClassifier(n_estimators=50)
                else:
                    model = RandomForestRegressor(n_estimators=50)
                    
            elif self.current_algorithm == "Linear Regression" and task == "regression":
                # Simple linear regression using numpy
                coef, residuals, rank, s = np.linalg.lstsq(X, y, rcond=None)
                predictions = X @ coef
                mse = float(np.mean((predictions - y) ** 2))
                r2 = 1 - (np.sum((y - predictions) ** 2) / np.sum((y - np.mean(y)) ** 2))
                
                self.results_label.setText(
                    f"✅ Linear Regression Results:\n"
                    f"MSE: {mse:.4f} | R²: {r2:.4f}\n"
                    f"Samples: {n_samples} | Features: {n_features}"
                )
                return
                
            else:
                self.results_label.setText(f"❌ {self.current_algorithm} not implemented for {task} task")
                return
            
            # Train and evaluate model
            model.fit(X, y)
            predictions = model.predict(X)
            
            if task == "classification":
                accuracy = float(np.mean(predictions == y))
                unique_classes = len(np.unique(y))
                self.results_label.setText(
                    f"✅ {self.current_algorithm} Classification Results:\n"
                    f"Accuracy: {accuracy:.3f} | Classes: {unique_classes}\n"
                    f"Samples: {n_samples} | Features: {n_features}"
                )
            else:
                mse = float(np.mean((predictions - y) ** 2))
                mae = float(np.mean(np.abs(predictions - y)))
                self.results_label.setText(
                    f"✅ {self.current_algorithm} Regression Results:\n"
                    f"MSE: {mse:.4f} | MAE: {mae:.4f}\n"
                    f"Samples: {n_samples} | Features: {n_features}"
                )
                
        except Exception as e:
            self.results_label.setText(f"❌ Error running {self.current_algorithm}: {str(e)}")


class MLExplorerMainWindow(QMainWindow):
    """Main application window with modern PyQt6 interface."""
    
    def __init__(self):
        super().__init__()
        self.setWindowTitle("🤖 Machine Learning Framework Explorer")
        self.setGeometry(100, 100, 1400, 900)
        self.setup_ui()
        
    def setup_ui(self):
        """Setup the main user interface."""
        # Central widget
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        
        # Main layout using vertical splitter for size control
        main_layout = QVBoxLayout(central_widget)
        main_layout.setContentsMargins(0, 0, 0, 0)
        main_layout.setSpacing(0)
        
        # Create vertical splitter for header and content areas
        main_splitter = QSplitter(Qt.Orientation.Vertical)
        main_layout.addWidget(main_splitter)
        
        # Header
        header = self.create_header()
        main_splitter.addWidget(header)
        
        # Main content area with horizontal splitter
        content_splitter = QSplitter(Qt.Orientation.Horizontal)
        main_splitter.addWidget(content_splitter)
        
        # Left panel - Algorithm categories and lists
        left_panel = self.create_left_panel()
        content_splitter.addWidget(left_panel)
        
        # Right panel - Details and quick run
        right_panel = self.create_right_panel()
        content_splitter.addWidget(right_panel)
        
        # Set horizontal splitter proportions (30% left, 70% right)
        content_splitter.setSizes([400, 1000])
        
        # Set vertical splitter proportions (10% header, 90% content)
        # Using window height of 900px: header=90px, content=810px
        main_splitter.setSizes([90, 810])
        
        # Apply global stylesheet
        self.setStyleSheet("""
            QMainWindow {
                background-color: #f5f5f5;
                color: black;
            }
            QTabWidget::pane {
                border: 1px solid #c0c0c0;
                background-color: white;
                color: black;
            }
            QTabWidget::tab-bar {
                alignment: center;
            }
            QTabBar::tab {
                background-color: #e1e1e1;
                padding: 10px 20px;
                margin-right: 2px;
                border-top-left-radius: 4px;
                border-top-right-radius: 4px;
                color: black;
            }
            QTabBar::tab:selected {
                background-color: white;
                font-weight: bold;
                color: black;
            }
            QTabBar::tab:hover {
                background-color: #f0f0f0;
                color: black;
            }
        """)
        
    def create_header(self) -> QWidget:
        """Create the application header."""
        header = QFrame()
        header.setStyleSheet("""
            QFrame {
                background: qlineargradient(x1: 0, y1: 0, x2: 1, y2: 0,
                    stop: 0 #e8e8e8, stop: 1 #f0f0f0);
                color: black;
                padding: 8px 20px;
                border-radius: 5px;
                margin-bottom: 2px;
            }
        """)
        header.setMaximumHeight(50)  # Further reduced for smaller text
        header.setMinimumHeight(40)  # Reduced minimum height
        
        layout = QHBoxLayout(header)
        layout.setContentsMargins(10, 5, 10, 5)  # Reduced margins
        
        title = QLabel("🤖 Machine Learning Framework Explorer")
        title_font = QFont()
        title_font.setPointSize(7)  # Reduced from 14 (half the size)
        title_font.setBold(True)
        title.setFont(title_font)
        title.setStyleSheet("color: black;")
        
        subtitle = QLabel("Explore, Learn, and Implement ML Algorithms")
        subtitle.setStyleSheet("color: black; font-size: 5px;")  # Reduced from 10px (half the size)
        
        title_layout = QVBoxLayout()
        title_layout.setSpacing(2)  # Reduced spacing between title and subtitle
        title_layout.addWidget(title)
        title_layout.addWidget(subtitle)
        
        layout.addLayout(title_layout)
        layout.addStretch()
        
        return header
        
    def create_left_panel(self) -> QWidget:
        """Create the left panel with algorithm categories."""
        panel = QWidget()
        layout = QVBoxLayout(panel)
        
        # Create tab widget for categories
        self.tab_widget = QTabWidget()
        
        # Supervised learning tab
        supervised_widget = AlgorithmListWidget(AlgorithmDatabase.SUPERVISED_ALGORITHMS)
        supervised_widget.algorithmSelected.connect(
            lambda name, info: self.on_algorithm_selected(name, info, "supervised")
        )
        self.tab_widget.addTab(supervised_widget, "🎯 Supervised (7)")
        
        # Unsupervised learning tab
        unsupervised_widget = AlgorithmListWidget(AlgorithmDatabase.UNSUPERVISED_ALGORITHMS)
        unsupervised_widget.algorithmSelected.connect(
            lambda name, info: self.on_algorithm_selected(name, info, "unsupervised")
        )
        self.tab_widget.addTab(unsupervised_widget, "🔍 Unsupervised (4)")
        
        # Semi-supervised learning tab
        semi_supervised_widget = AlgorithmListWidget(AlgorithmDatabase.SEMI_SUPERVISED_ALGORITHMS)
        semi_supervised_widget.algorithmSelected.connect(
            lambda name, info: self.on_algorithm_selected(name, info, "semi_supervised")
        )
        self.tab_widget.addTab(semi_supervised_widget, "🎭 Semi-Supervised (4)")
        
        layout.addWidget(self.tab_widget)
        
        return panel
        
    def create_right_panel(self) -> QWidget:
        """Create the right panel with details and quick run."""
        panel = QWidget()
        layout = QVBoxLayout(panel)
        
        # Algorithm details
        self.details_widget = AlgorithmDetailWidget()
        self.details_widget.setHtml("""
        <div style="text-align: center; padding: 50px; color: #6c757d;">
            <h2>Welcome to ML Framework Explorer!</h2>
            <p>Select an algorithm from the categories on the left to view detailed information, 
               use cases, advantages, disadvantages, and implementation status.</p>
            <p>Use the Quick Run panel below to test implemented algorithms with synthetic data.</p>
        </div>
        """)
        
        # Quick run panel
        self.quick_run_panel = QuickRunPanel()
        
        # Add to layout with splitter for resizing
        right_splitter = QSplitter(Qt.Orientation.Vertical)
        right_splitter.addWidget(self.details_widget)
        right_splitter.addWidget(self.quick_run_panel)
        right_splitter.setSizes([600, 200])  # More space for details
        
        layout.addWidget(right_splitter)
        
        return panel
        
    def on_algorithm_selected(self, name: str, info: Dict[str, Any], category: str):
        """Handle algorithm selection from any category."""
        # Update details view
        self.details_widget.display_algorithm_details(name, info, category)
        
        # Update quick run panel
        self.quick_run_panel.set_current_algorithm(name, info, category)


def main():
    """Main application entry point."""
    app = QApplication(sys.argv)
    
    # Set application properties
    app.setApplicationName("ML Framework Explorer")
    app.setOrganizationName("ML Framework")
    app.setOrganizationDomain("ml-framework.local")
    
    # Create and show main window
    window = MLExplorerMainWindow()
    window.show()
    
    return app.exec()


if __name__ == "__main__":
    sys.exit(main())
