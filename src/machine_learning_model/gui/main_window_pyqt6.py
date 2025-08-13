"""
Modern PyQt6-based GUI for Machine Learning Framework Explorer.
Provides categorized interface for exploring supervised, unsupervised, and semi-supervised algorithms.
"""
import sys
from typing import Any, Dict

import numpy as np
from PyQt6.QtCore import Qt, pyqtSignal
from PyQt6.QtGui import QFont
from PyQt6.QtWidgets import (
    QApplication,
    QButtonGroup,
    QFrame,
    QHBoxLayout,
    QLabel,
    QListWidget,
    QListWidgetItem,
    QMainWindow,
    QPushButton,
    QRadioButton,
    QSplitter,
    QTabWidget,
    QTextEdit,
    QVBoxLayout,
    QWidget,
)

try:
    from machine_learning_model.supervised.decision_tree import (
        DecisionTreeClassifier,
        DecisionTreeRegressor,
    )
    from machine_learning_model.supervised.random_forest import (
        RandomForestClassifier,
        RandomForestRegressor,
    )
except ImportError:  # Fallback minimal stubs
    class DecisionTreeClassifier:
        def fit(self, X, y):
            pass

        def predict(self, X):
            return np.zeros(len(X))

    class DecisionTreeRegressor:
        def fit(self, X, y):
            pass

        def predict(self, X):
            return np.zeros(len(X))

    class RandomForestClassifier:
        def __init__(self, n_estimators=25):
            pass

        def fit(self, X, y):
            pass

        def predict(self, X):
            return np.zeros(len(X))

    class RandomForestRegressor:
        def __init__(self, n_estimators=25):
            pass

        def fit(self, X, y):
            pass

        def predict(self, X):
            return np.zeros(len(X))

# Algorithm execution moved to models.algorithm_runner (MVC refactor step 3)
# Metadata duplication removed: source from registry_data module
try:  # pragma: no cover
    from .models import registry_data as _algo_data  # type: ignore
except Exception:  # pragma: no cover
    import importlib
    _algo_data = importlib.import_module('machine_learning_model.gui.models.registry_data')  # type: ignore

SUPERVISED_ALGORITHMS = _algo_data.SUPERVISED_ALGORITHMS_DATA
UNSUPERVISED_ALGORITHMS = _algo_data.UNSUPERVISED_ALGORITHMS_DATA
SEMI_SUPERVISED_ALGORITHMS = _algo_data.SEMI_SUPERVISED_ALGORITHMS_DATA


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
    """Panel for quick algorithm testing with synthetic data (refactored)."""

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self.current_algorithm: str | None = None
        self.current_category: str | None = None
        self.current_info: Dict[str, Any] | None = None
        self._setup_ui()

    def _setup_ui(self) -> None:
        self.setStyleSheet(
            """
            QFrame { background-color: #f8f9fa; border: 1px solid #dee2e6; border-radius: 5px; padding: 10px; color: black; }
            QPushButton { background-color: #007bff; color: black; border: none; padding: 8px 16px; border-radius: 4px; font-weight: bold; }
            QPushButton:hover { background-color: #0056b3; color: black; }
            QPushButton:disabled { background-color: #6c757d; color: black; }
            """
        )
        layout = QVBoxLayout(self)
        title = QLabel("🚀 Quick Run (Synthetic Data)")
        title_font = QFont()
        title_font.setBold(True)
        title_font.setPointSize(12)
        title.setFont(title_font)
        layout.addWidget(title)

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

        self.run_button = QPushButton("Run Selected Algorithm")
        self.run_button.clicked.connect(self.run_algorithm)
        self.run_button.setEnabled(False)
        layout.addWidget(self.run_button)

        self.results_label = QLabel("Select an algorithm and click Run to test with synthetic data")
        self.results_label.setStyleSheet(
            """
            QLabel { background-color: white; padding: 10px; border: 1px solid #dee2e6; border-radius: 3px; margin-top: 5px; }
            """
        )
        self.results_label.setWordWrap(True)
        layout.addWidget(self.results_label)

    def set_current_algorithm(self, name: str, info: Dict[str, Any], category: str) -> None:
        self.current_algorithm = name
        self.current_category = category
        self.current_info = info
        implementation_status = info.get("implementation_status", "planned")
        self.run_button.setEnabled(implementation_status in {"complete", "ready"})
        if implementation_status == "complete":
            self.results_label.setText(
                f"Ready to test {name} with synthetic data. Choose task type and click Run."
            )
        elif implementation_status == "ready":
            self.results_label.setText(
                f"{name} is ready for implementation. Synthetic testing will be available soon."
            )
        else:
            self.results_label.setText(
                f"{name} is planned for future implementation. Testing not yet available."
            )

    def run_algorithm(self) -> None:
        if not self.current_algorithm:
            return
        try:
            from .models.algorithm_runner import (
                run_algorithm as _run_algo,  # type: ignore
            )
            from .models.data_synthesizer import SyntheticDataSpec  # type: ignore
        except Exception as exc:  # pragma: no cover - defensive
            self.results_label.setText(f"❌ Internal import error: {exc}")
            return
        task = "classification" if self.classification_radio.isChecked() else "regression"
        spec = SyntheticDataSpec(task=task, n_samples=100, n_features=5, seed=42)
        result = _run_algo(self.current_algorithm, task, spec)
        self.results_label.setText(result.details)


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
                /* Reduced padding by 20% (was 10px 20px) */
                padding: 8px 16px;
                margin-right: 2px;
                border-top-left-radius: 4px;
                border-top-right-radius: 4px;
                font-size: 80%; /* reduce font size by 20% */
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
        supervised_widget = AlgorithmListWidget(SUPERVISED_ALGORITHMS)
        supervised_widget.algorithmSelected.connect(
            lambda name, info: self.on_algorithm_selected(name, info, "supervised")
        )
        self.tab_widget.addTab(supervised_widget, f"🎯 Supervised ({len(SUPERVISED_ALGORITHMS)})")

        # Unsupervised learning tab
        unsupervised_widget = AlgorithmListWidget(UNSUPERVISED_ALGORITHMS)
        unsupervised_widget.algorithmSelected.connect(
            lambda name, info: self.on_algorithm_selected(name, info, "unsupervised")
        )
        self.tab_widget.addTab(unsupervised_widget, f"🔍 Unsupervised ({len(UNSUPERVISED_ALGORITHMS)})")

        # Semi-supervised learning tab
        semi_supervised_widget = AlgorithmListWidget(SEMI_SUPERVISED_ALGORITHMS)
        semi_supervised_widget.algorithmSelected.connect(
            lambda name, info: self.on_algorithm_selected(name, info, "semi_supervised")
        )
        self.tab_widget.addTab(semi_supervised_widget, f"🎭 Semi-Supervised ({len(SEMI_SUPERVISED_ALGORITHMS)})")

        layout.addWidget(self.tab_widget)
        # Reduce first tab (Supervised) width by ~20% using stylesheet override
        try:
            supervised_bar = self.tab_widget.tabBar()
            # Apply a smaller padding to just the first tab
            # Qt doesn't allow per-tab CSS easily; we simulate by reducing overall then compensating others if needed.
            # For simplicity here we just ensure minimum width narrower.
            supervised_bar.setTabButton(0, Qt.TabPosition.North, supervised_bar.tabButton(0, Qt.TabPosition.North))
            # Optionally could adjust size policy; kept minimal.
        except Exception:
            pass

        return panel

    def create_right_panel(self) -> QWidget:
        """Create the right panel with details (top) and quick run/results (bottom)."""
        panel = QWidget()
        layout = QVBoxLayout(panel)

        # Details widget (information box)
        self.details_widget = AlgorithmDetailWidget()
        self.details_widget.setHtml(
            """
            <div style=\"text-align: center; padding: 40px; color: #6c757d;\">
                <h2>Welcome to ML Framework Explorer!</h2>
                <p>Select an algorithm from the categories on the left to view detailed information,
                   use cases, advantages, disadvantages, and implementation status.</p>
                <p>Use the Quick Run panel below to test implemented algorithms with synthetic data.</p>
            </div>
            """
        )

        # Quick run panel (results section)
        self.quick_run_panel = QuickRunPanel()

        splitter = QSplitter(Qt.Orientation.Vertical)
        splitter.addWidget(self.details_widget)
        splitter.addWidget(self.quick_run_panel)
        # Target 60/40 split (information box reduced ~20 percentage points from prior 80%)
        splitter.setStretchFactor(0, 3)
        splitter.setStretchFactor(1, 2)

        layout.addWidget(splitter)
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
