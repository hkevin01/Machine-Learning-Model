"""
Main GUI window for Machine Learning Framework Explorer.
Provides an interactive interface for exploring ML algorithms categorized by learning type.
"""
from __future__ import annotations
import tkinter as tk
from tkinter import messagebox, scrolledtext
try:  # pragma: no cover
    from .icon_utils import icon_for_status  # type: ignore
except Exception:  # pragma: no cover
    def icon_for_status(_):  # type: ignore
        return "[?]"
import numpy as np
from machine_learning_model.supervised.decision_tree import DecisionTreeClassifier, DecisionTreeRegressor
from machine_learning_model.supervised.random_forest import RandomForestClassifier, RandomForestRegressor


class MLAlgorithmInfo:
    """Contains detailed information about ML algorithms categorized by learning type."""
    
    SUPERVISED_ALGORITHMS = {
        "Linear Regression": {
            "description": "A fundamental algorithm for predicting continuous values by finding the best linear relationship between features and target.",
            "use_cases": "House price prediction, sales forecasting, risk assessment",
            "pros": "Simple, interpretable, fast training, no hyperparameters",
            "cons": "Assumes linear relationship, sensitive to outliers",
            "complexity": "Low",
            "type": "Regression",
            "status": "✅ Ready for Implementation"
        },
        "Logistic Regression": {
            "description": "Classification algorithm using logistic function to model probability of class membership.",
            "use_cases": "Email spam detection, medical diagnosis, marketing response",
            "pros": "Probabilistic output, interpretable, handles categorical features",
            "cons": "Assumes linear decision boundary, sensitive to outliers",
            "complexity": "Low",
            "type": "Classification",
            "status": "✅ Ready for Implementation"
        },
        "Decision Trees": {
            "description": "Tree-like model making decisions by splitting data based on feature values to maximize information gain.",
            "use_cases": "Credit approval, feature selection, rule extraction",
            "pros": "Highly interpretable, handles mixed data types, no assumptions about data distribution",
            "cons": "Prone to overfitting, unstable with small data changes",
            "complexity": "Medium",
            "type": "Both",
            "status": "✅ Complete - Production Ready"
        },
        "Random Forest": {
            "description": "Ensemble method combining multiple decision trees with voting/averaging to improve accuracy and reduce overfitting.",
            "use_cases": "Feature importance ranking, general-purpose prediction, biomedical research",
            "pros": "Reduces overfitting, handles missing values, provides feature importance, built-in cross-validation (OOB)",
            "cons": "Less interpretable than single trees, can overfit with very noisy data",
            "complexity": "Medium",
            "type": "Both",
            "status": "✅ Complete - Production Ready"
        },
        "Support Vector Machine": {
            "description": "Finds optimal hyperplane to separate classes or predict values by maximizing margin between data points.",
            "use_cases": "Text classification, image recognition, gene classification",
            "pros": "Effective in high dimensions, memory efficient, versatile with kernels",
            "cons": "Slow on large datasets, sensitive to feature scaling, no probabilistic output",
            "complexity": "High",
            "type": "Both",
            "status": "🔄 Next - Starting this week"
        },
        "XGBoost": {
            "description": "Advanced gradient boosting framework optimized for speed and performance with regularization.",
            "use_cases": "Kaggle competitions, structured data prediction, feature selection",
            """Main GUI window for Machine Learning Framework Explorer.
            Provides an interactive interface for exploring ML algorithms categorized by learning type.
            """

            from __future__ import annotations

            import os
            import subprocess
            import sys
            import tkinter as tk
            from tkinter import messagebox, scrolledtext, ttk

            try:  # pragma: no cover - defensive import
                from .icon_utils import icon_for_status  # type: ignore
            except Exception:  # pragma: no cover
                def icon_for_status(_):
                    return "[?]"

            import numpy as np
            from machine_learning_model.supervised.decision_tree import (
                DecisionTreeClassifier,
                DecisionTreeRegressor,
            )
            from machine_learning_model.supervised.random_forest import (
                RandomForestClassifier,
                RandomForestRegressor,
            )


            class MLAlgorithmInfo:
                """Container of algorithm metadata grouped by learning paradigm."""

                SUPERVISED_ALGORITHMS = {
                    "Linear Regression": {
                        "description": "A fundamental algorithm for predicting continuous values by finding the best linear relationship between features and target.",
                        "use_cases": "House price prediction, sales forecasting, risk assessment",
                        "pros": "Simple, interpretable, fast training, no hyperparameters",
                        "cons": "Assumes linear relationship, sensitive to outliers",
                        "complexity": "Low",
                        "type": "Regression",
                        "status": "✅ Ready for Implementation",
                    },
                    "Logistic Regression": {
                        "description": "Classification algorithm using logistic function to model probability of class membership.",
                        "use_cases": "Email spam detection, medical diagnosis, marketing response",
                        "pros": "Probabilistic output, interpretable, handles categorical features",
                        "cons": "Assumes linear decision boundary, sensitive to outliers",
                        "complexity": "Low",
                        "type": "Classification",
                        "status": "✅ Ready for Implementation",
                    },
                    "Decision Trees": {
                        "description": "Tree-like model making decisions by splitting data based on feature values to maximize information gain.",
                        "use_cases": "Credit approval, feature selection, rule extraction",
                        "pros": "Highly interpretable, handles mixed data types, no assumptions about data distribution",
                        "cons": "Prone to overfitting, unstable with small data changes",
                        "complexity": "Medium",
                        "type": "Both",
                        "status": "✅ Complete - Production Ready",
                    },
                    "Random Forest": {
                        "description": "Ensemble method combining multiple decision trees with voting/averaging to improve accuracy and reduce overfitting.",
                        "use_cases": "Feature importance ranking, general-purpose prediction, biomedical research",
                        "pros": "Reduces overfitting, handles missing values, provides feature importance, built-in cross-validation (OOB)",
                        "cons": "Less interpretable than single trees, can overfit with very noisy data",
                        "complexity": "Medium",
                        "type": "Both",
                        "status": "✅ Complete - Production Ready",
                    },
                    "Support Vector Machine": {
                        "description": "Finds optimal hyperplane to separate classes or predict values by maximizing margin between data points.",
                        "use_cases": "Text classification, image recognition, gene classification",
                        "pros": "Effective in high dimensions, memory efficient, versatile with kernels",
                        "cons": "Slow on large datasets, sensitive to feature scaling, no probabilistic output",
                        "complexity": "High",
                        "type": "Both",
                        "status": "🔄 Next - Starting this week",
                    },
                    "XGBoost": {
                        "description": "Advanced gradient boosting framework optimized for speed and performance with regularization.",
                        "use_cases": "Kaggle competitions, structured data prediction, feature selection",
                        "pros": "State-of-the-art performance, built-in regularization, handles missing values",
                        "cons": "Many hyperparameters, computationally intensive, requires tuning",
                        "complexity": "High",
                        "type": "Both",
                        "status": "📋 Planned - Advanced Phase",
                    },
                    "Neural Networks": {
                        "description": "Multi-layered networks of interconnected nodes mimicking brain neurons for complex pattern recognition.",
                        "use_cases": "Image recognition, natural language processing, game playing",
                        "pros": "Universal approximator, handles complex non-linear relationships",
                        "cons": "Requires large datasets, computationally expensive, black box",
                        "complexity": "High",
                        "type": "Both",
                        "status": "📋 Planned - Advanced Phase",
                    },
                }

                UNSUPERVISED_ALGORITHMS = {
                    "K-Means Clustering": {
                        "description": "Partitions data into k clusters by minimizing within-cluster sum of squares.",
                        "use_cases": "Customer segmentation, image compression, market research",
                        "pros": "Simple and fast, works well with spherical clusters",
                        "cons": "Must specify k beforehand, sensitive to initialization and outliers",
                        "complexity": "Medium",
                        "type": "Clustering",
                        "status": "📋 Planned - Phase 3",
                    },
                    "DBSCAN": {
                        "description": "Density-based clustering that groups together points in high-density areas and marks outliers.",
                        "use_cases": "Anomaly detection, image processing, social network analysis",
                        "pros": "Automatically determines clusters, handles noise, finds arbitrary shapes",
                        "cons": "Sensitive to hyperparameters, struggles with varying densities",
                        "complexity": "Medium",
                        "type": "Clustering",
                        "status": "📋 Planned - Phase 3",
                    },
                    "Principal Component Analysis": {
                        "description": "Dimensionality reduction technique that projects data onto principal components capturing maximum variance.",
                        "use_cases": "Data visualization, feature reduction, noise reduction",
                        "pros": "Reduces overfitting, speeds up training, removes multicollinearity",
                        "cons": "Loses interpretability, linear transformation only",
                        "complexity": "Medium",
                        "type": "Dimensionality Reduction",
                        "status": "📋 Planned - Phase 3",
                    },
                    "Hierarchical Clustering": {
                        "description": "Creates tree-like cluster hierarchy using linkage criteria (agglomerative or divisive).",
                        "use_cases": "Phylogenetic analysis, social network analysis, image segmentation",
                        "pros": "No need to specify number of clusters, creates interpretable hierarchy",
                        "cons": "Computationally expensive O(n³), sensitive to noise",
                        "complexity": "High",
                        "type": "Clustering",
                        "status": "📋 Planned - Phase 3",
                    },
                }

                SEMI_SUPERVISED_ALGORITHMS = {
                    "Label Propagation": {
                        "description": "Graph-based algorithm that propagates labels from labeled to unlabeled data through similarity graphs.",
                        "use_cases": "Text classification with few labels, image annotation, social media analysis",
                        "pros": "Works with few labeled examples, natural uncertainty estimation",
                        "cons": "Requires good similarity metric, computationally expensive for large graphs",
                        "complexity": "High",
                        "type": "Classification",
                        "status": "📋 Planned - Phase 4",
                    },
                    "Self-Training": {
                        "description": "Iteratively trains on labeled data, predicts unlabeled data, adds confident predictions to training set.",
                        "use_cases": "NLP with limited annotations, medical diagnosis, web page classification",
                        "pros": "Simple to implement, works with any base classifier",
                        "cons": "Can amplify errors, requires good confidence estimation",
                        "complexity": "Medium",
                        "type": "Classification",
                        "status": "📋 Planned - Phase 4",
                    },
                    "Co-Training": {
                        "description": "Uses two different views of data to train separate classifiers that teach each other.",
                        "use_cases": "Web page classification, email classification, multi-modal learning",
                        "pros": "Leverages multiple feature views, reduces overfitting",
                        "cons": "Requires conditionally independent views, complex setup",
                        "complexity": "High",
                        "type": "Classification",
                        "status": "📋 Planned - Phase 4",
                    },
                    "Semi-Supervised SVM": {
                        "description": "Extends SVM to work with both labeled and unlabeled data using transductive learning.",
                        "use_cases": "Text mining, bioinformatics, computer vision with limited labels",
                        "pros": "Leverages unlabeled data effectively, maintains SVM advantages",
                        "cons": "Non-convex optimization, computationally challenging",
                        "complexity": "High",
                        "type": "Classification",
                        "status": "📋 Planned - Phase 4",
                    },
                }


            class MainWindow:
                """Main application window with categorized algorithm display."""

                def __init__(self) -> None:
                    self.root = tk.Tk()
                    self.root.title("Machine Learning Framework Explorer")
                    self.root.geometry("1400x900")
                    self.root.configure(bg="#f0f0f0")

                    self.current_algorithm: str | None = None
                    self.current_category: str | None = None
                    self.listbox_widgets: dict[str, tk.Listbox] = {}

                    self.create_widgets()

                # ---------------- GUI Construction -----------------
                def create_widgets(self) -> None:
                    header_frame = tk.Frame(self.root, bg="#2c3e50", height=80)
                    header_frame.pack(fill="x", padx=5, pady=5)
                    header_frame.pack_propagate(False)

                    title_label = tk.Label(
                        header_frame,
                        text="🤖 Machine Learning Framework Explorer",
                        font=("Arial", 16, "bold"),
                        bg="#2c3e50",
                        fg="white",
                    )
                    title_label.pack(pady=20)

                    main_frame = tk.Frame(self.root, bg="#f0f0f0")
                    main_frame.pack(fill="both", expand=True, padx=10, pady=5)

                    left_frame = tk.Frame(main_frame, bg="white", width=400)
                    left_frame.pack(side="left", fill="y", padx=(0, 10))
                    left_frame.pack_propagate(False)

                    self.notebook = ttk.Notebook(left_frame)
                    self.notebook.pack(fill="both", expand=True, padx=10, pady=10)
                    self.create_category_tabs()

                    right_frame = tk.Frame(main_frame, bg="white")
                    right_frame.pack(side="right", fill="both", expand=True)

                    self.details_text = scrolledtext.ScrolledText(
                        right_frame,
                        wrap=tk.WORD,
                        font=("Arial", 11),
                        bg="#f8f9fa",
                        padx=15,
                        pady=15,
                    )
                    self.details_text.pack(fill="both", expand=True, padx=10, pady=(10, 5))

                    run_frame = tk.LabelFrame(right_frame, text="Quick Run (synthetic)", bg="white")
                    run_frame.pack(fill="x", padx=10, pady=(0, 10))
                    self.task_var = tk.StringVar(value="classification")
                    tk.Radiobutton(
                        run_frame,
                        text="Classification",
                        variable=self.task_var,
                        value="classification",
                        bg="white",
                    ).grid(row=0, column=0, sticky="w")
                    tk.Radiobutton(
                        run_frame,
                        text="Regression",
                        variable=self.task_var,
                        value="regression",
                        bg="white",
                    ).grid(row=0, column=1, sticky="w")
                    tk.Button(
                        run_frame,
                        text="Run Selected",
                        command=self.run_selected_algorithm,
                        bg="#2c3e50",
                        fg="white",
                    ).grid(row=0, column=2, padx=10)
                    self.run_output = tk.Label(
                        run_frame, text="Select an algorithm then Run", anchor="w", bg="white"
                    )
                    self.run_output.grid(row=1, column=0, columnspan=3, sticky="we", pady=4)
                    for i in range(3):
                        run_frame.grid_columnconfigure(i, weight=1)

                    self.show_welcome_message()

                def create_category_tabs(self) -> None:
                    supervised_frame = ttk.Frame(self.notebook)
                    self.notebook.add(supervised_frame, text="🎯 Supervised (7)")
                    self.create_algorithm_list(
                        supervised_frame, MLAlgorithmInfo.SUPERVISED_ALGORITHMS, "supervised"
                    )

                    unsupervised_frame = ttk.Frame(self.notebook)
                    self.notebook.add(unsupervised_frame, text="🔍 Unsupervised (4)")
                    self.create_algorithm_list(
                        unsupervised_frame, MLAlgorithmInfo.UNSUPERVISED_ALGORITHMS, "unsupervised"
                    )

                    semi_supervised_frame = ttk.Frame(self.notebook)
                    self.notebook.add(semi_supervised_frame, text="🎭 Semi-Supervised (4)")
                    self.create_algorithm_list(
                        semi_supervised_frame,
                        MLAlgorithmInfo.SEMI_SUPERVISED_ALGORITHMS,
                        "semi_supervised",
                    )

                def create_algorithm_list(self, parent, algorithms, category: str) -> None:
                    descriptions = {
                        "supervised": "Uses labeled data to learn patterns and make predictions",
                        "unsupervised": "Finds hidden patterns in data without labels",
                        "semi_supervised": "Combines labeled and unlabeled data for learning",
                    }
                    desc_label = tk.Label(
                        parent,
                        text=descriptions[category],
                        font=("Arial", 10),
                        bg="white",
                        fg="#666",
                        wraplength=350,
                    )
                    desc_label.pack(pady=(10, 5), padx=10)

                    listbox = tk.Listbox(
                        parent,
                        font=("Arial", 10),
                        selectmode="single",
                        bg="#f8f9fa",
                        selectbackground="#007bff",
                        selectforeground="white",
                    )
                    listbox.pack(fill="both", expand=True, padx=10, pady=5)
                    self.listbox_widgets[category] = listbox

                    for algorithm, info in algorithms.items():
                        raw = info["status"]
                        if raw.startswith("✅"):
                            ic = icon_for_status("COMPLETED")
                        elif raw.startswith("🔄"):
                            ic = icon_for_status("IN_PROGRESS")
                        elif raw.startswith("📋"):
                            ic = icon_for_status("NOT_STARTED")
                        else:
                            ic = icon_for_status("NOT_STARTED")
                        listbox.insert(tk.END, f"{ic} {algorithm}")

                    listbox.bind(
                        "<<ListboxSelect>>",
                        lambda event, cat=category, alg_dict=algorithms, lb=listbox: self.on_algorithm_select(
                            event, cat, alg_dict, lb
                        ),
                    )

                def on_algorithm_select(self, _event, category: str, algorithms, listbox) -> None:
                    selection = listbox.curselection()
                    if not selection:
                        return
                    selected_text = listbox.get(selection[0])
                    parts = selected_text.split()
                    algorithm_name = " ".join(parts[1:]) if len(parts) > 1 else selected_text
                    self.current_algorithm = algorithm_name
                    self.current_category = category
                    self.show_algorithm_details(algorithm_name, algorithms, category)

                # ---------------- Quick synthetic run -----------------
                def run_selected_algorithm(self) -> None:
                    if not self.current_algorithm:
                        messagebox.showinfo("Run", "Please select an algorithm first.")
                        return
                    task = self.task_var.get()
                    n = 120
                    try:
                        if task == "classification":
                            X = np.random.randn(n, 5)
                            y = (X[:, 0] + X[:, 1] > 0).astype(int)
                        else:
                            X = np.random.randn(n, 5)
                            y = X[:, 0] * 0.7 + X[:, 1] * -0.2 + np.random.randn(n) * 0.1
                        alg = self.current_algorithm
                        if alg == "Decision Trees":
                            model = (
                                DecisionTreeClassifier()
                                if task == "classification"
                                else DecisionTreeRegressor()
                            )
                        elif alg == "Random Forest":
                            model = (
                                RandomForestClassifier(n_estimators=25)
                                if task == "classification"
                                else RandomForestRegressor(n_estimators=25)
                            )
                        elif alg == "Linear Regression" and task == "regression":
                            coef, *_ = np.linalg.lstsq(X, y, rcond=None)
                            pred = X @ coef
                            mse = float(np.mean((pred - y) ** 2))
                            self.run_output.config(text=f"Linear Regression MSE={mse:.4f}")
                            return
                        else:
                            self.run_output.config(
                                text=f"Run not implemented for {alg} ({task})"
                            )
                            return
                        model.fit(X, y)
                        pred = model.predict(X)
                        if task == "classification":
                            acc = float(np.mean(pred == y))
                            self.run_output.config(text=f"{alg} Accuracy={acc:.3f}")
                        else:
                            mse = float(np.mean((pred - y) ** 2))
                            self.run_output.config(text=f"{alg} MSE={mse:.4f}")
                    except Exception as e:  # pragma: no cover
                        self.run_output.config(text=f"Error: {e}")

                # ---------------- Details / Info panes -----------------
                def show_algorithm_details(self, algorithm_name: str, algorithms, category: str) -> None:
                    if algorithm_name not in algorithms:
                        return
                    info = algorithms[algorithm_name]
                    category_names = {
                        "supervised": "Supervised Learning",
                        "unsupervised": "Unsupervised Learning",
                        "semi_supervised": "Semi-Supervised Learning",
                    }
                    if "Complete" in info["status"]:
                        implementation_guide = f"""
            🚀 IMPLEMENTATION AVAILABLE!

            📁 Location: src/machine_learning_model/supervised/
            📖 Examples: examples/supervised_examples/
            🧪 Tests: tests/test_supervised/

            ⚡ Quick Start:
            from machine_learning_model.supervised import {algorithm_name.replace(' ', '')}
            model = {algorithm_name.replace(' ', '')}()
            model.fit(X_train, y_train)
            predictions = model.predict(X_test)
            """
                    elif "Ready" in info["status"]:
                        implementation_guide = "📋 Ready for implementation - documentation and API design complete."
                    else:
                        implementation_guide = "⏳ Planned for future implementation phases."
                    details_text = f"""
            🎯 {algorithm_name} ({category_names[category]})

            📝 Description:
            {info['description']}

            🎯 Common Use Cases:
            {info['use_cases']}

            ✅ Advantages:
            {info['pros']}

            ❌ Disadvantages:
            {info['cons']}

            📊 Algorithm Type: {info['type']}
            🔧 Complexity: {info['complexity']}
            📈 Status: {info['status']}

            {implementation_guide}

            {'=' * 60}

            💡 Quick Tips:
            • Start with simpler algorithms (Linear/Logistic Regression) for baseline
            • Use ensemble methods (Random Forest, Gradient Boosting) for better accuracy
            • Consider your data size and computational resources
            • Always validate your model on unseen data
            """
                    self.details_text.delete(1.0, tk.END)
                    self.details_text.insert(1.0, details_text)

                def show_welcome_message(self) -> None:
                    welcome_text = (
                        "🎯 Welcome to Machine Learning Framework Explorer!\n\n"
                        "Select an algorithm from the tabs on the left to begin."
                    )
                    self.details_text.delete(1.0, tk.END)
                    self.details_text.insert(1.0, welcome_text)

                # ---------------- Actions / Helpers -----------------
                def get_current_selection(self):
                    if self.current_algorithm and self.current_category:
                        return self.current_algorithm, self.current_category
                    return None, None

                def run_algorithm(self) -> None:
                    algorithm, category = self.get_current_selection()
                    if not algorithm:
                        messagebox.showwarning("No Selection", "Please select an algorithm first!")
                        return
                    if category == "supervised":
                        algorithms = MLAlgorithmInfo.SUPERVISED_ALGORITHMS
                    elif category == "unsupervised":
                        algorithms = MLAlgorithmInfo.UNSUPERVISED_ALGORITHMS
                    else:
                        algorithms = MLAlgorithmInfo.SEMI_SUPERVISED_ALGORITHMS
                    status = algorithms[algorithm]["status"]
                    if "Complete" in status:
                        try:
                            from machine_learning_model.visualization.algorithm_visualizer import (
                                run_algorithm_demo,
                            )
                            messagebox.showinfo(
                                "Running Algorithm",
                                f"Running {algorithm} demo... this may take a moment.",
                            )
                            results = run_algorithm_demo(algorithm)
                            parts = [f"{k}: {v:.3f}" for k, v in results.items() if isinstance(v, (int, float))]
                            messagebox.showinfo(
                                "Algorithm Results",
                                "Completed successfully!\n" + "\n".join(parts),
                            )
                        except Exception as e:  # pragma: no cover
                            messagebox.showerror("Execution Error", str(e))
                    elif "Ready" in status:
                        messagebox.showinfo(
                            "Implementation Coming Soon",
                            f"{algorithm} implementation is ready and will be added soon.",
                        )
                    else:
                        messagebox.showinfo(
                            "Planned", f"{algorithm} is planned for a future phase."
                        )

                def view_examples(self) -> None:
                    algorithm, _category = self.get_current_selection()
                    if not algorithm:
                        messagebox.showwarning("No Selection", "Please select an algorithm first!")
                        return
                    example_mapping = {
                        "Decision Trees": "decision_tree_example.py",
                        "Random Forest": "random_forest_example.py",
                        "Linear Regression": "linear_regression_example.py",
                        "Logistic Regression": "logistic_regression_example.py",
                    }
                    if algorithm not in example_mapping:
                        messagebox.showinfo(
                            "Examples", f"Examples for {algorithm} coming soon."
                        )
                        return
                    example_file = example_mapping[algorithm]
                    example_path = f"examples/supervised_examples/{example_file}"
                    if not os.path.exists(example_path):
                        messagebox.showwarning("Not Found", f"File missing: {example_path}")
                        return
                    if not messagebox.askyesno(
                        "Run Example", f"Run {algorithm} example script?"
                    ):
                        return
                    try:
                        result = subprocess.run(
                            [sys.executable, example_path], capture_output=True, text=True, cwd="."
                        )
                        if result.returncode == 0:
                            messagebox.showinfo(
                                "Example Completed",
                                f"Output (first 200 chars):\n{result.stdout[:200]}...",
                            )
                        else:
                            messagebox.showerror(
                                "Example Failed", result.stderr[:400] + "..."
                            )
                    except Exception as e:  # pragma: no cover
                        messagebox.showerror("Error", str(e))

                def visualize_algorithm(self) -> None:
                    algorithm, _category = self.get_current_selection()
                    if not algorithm:
                        messagebox.showwarning("No Selection", "Please select an algorithm first!")
                        return
                    available = [
                        "Decision Trees",
                        "Random Forest",
                        "Linear Regression",
                        "Logistic Regression",
                    ]
                    if algorithm not in available:
                        messagebox.showinfo(
                            "Visualization", f"Visualization for {algorithm} coming soon."
                        )
                        return
                    try:
                        from machine_learning_model.visualization.algorithm_visualizer import (
                            run_algorithm_demo,
                        )
                        messagebox.showinfo(
                            "Visualization", f"Creating {algorithm} visualization..."
                        )
                        run_algorithm_demo(algorithm)
                        messagebox.showinfo(
                            "Visualization", "Completed and artifact saved to artifacts folder."
                        )
                    except Exception as e:  # pragma: no cover
                        messagebox.showerror("Visualization Error", str(e))

                def compare_performance(self) -> None:
                    messagebox.showinfo(
                        "Compare", "Comprehensive performance comparison coming soon."
                    )

                def run(self) -> None:
                    self.root.mainloop()


            def main() -> None:
                app = MainWindow()
                app.run()


            if __name__ == "__main__":  # pragma: no cover
                main()