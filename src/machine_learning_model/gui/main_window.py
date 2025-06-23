"""
Main GUI window for Machine Learning Framework Explorer.
Provides an interactive interface for exploring ML algorithms categorized by learning type.
"""

import os
import subprocess
import sys
import tkinter as tk
import webbrowser
from tkinter import messagebox, scrolledtext, ttk


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
            "pros": "State-of-the-art performance, built-in regularization, handles missing values",
            "cons": "Many hyperparameters, computationally intensive, requires tuning",
            "complexity": "High",
            "type": "Both",
            "status": "📋 Planned - Advanced Phase"
        },
        "Neural Networks": {
            "description": "Multi-layered networks of interconnected nodes mimicking brain neurons for complex pattern recognition.",
            "use_cases": "Image recognition, natural language processing, game playing",
            "pros": "Universal approximator, handles complex non-linear relationships",
            "cons": "Requires large datasets, computationally expensive, black box",
            "complexity": "High",
            "type": "Both",
            "status": "📋 Planned - Advanced Phase"
        }
    }
    
    UNSUPERVISED_ALGORITHMS = {
        "K-Means Clustering": {
            "description": "Partitions data into k clusters by minimizing within-cluster sum of squares.",
            "use_cases": "Customer segmentation, image compression, market research",
            "pros": "Simple and fast, works well with spherical clusters",
            "cons": "Must specify k beforehand, sensitive to initialization and outliers",
            "complexity": "Medium",
            "type": "Clustering",
            "status": "📋 Planned - Phase 3"
        },
        "DBSCAN": {
            "description": "Density-based clustering that groups together points in high-density areas and marks outliers.",
            "use_cases": "Anomaly detection, image processing, social network analysis",
            "pros": "Automatically determines clusters, handles noise, finds arbitrary shapes",
            "cons": "Sensitive to hyperparameters, struggles with varying densities",
            "complexity": "Medium",
            "type": "Clustering",
            "status": "📋 Planned - Phase 3"
        },
        "Principal Component Analysis": {
            "description": "Dimensionality reduction technique that projects data onto principal components capturing maximum variance.",
            "use_cases": "Data visualization, feature reduction, noise reduction",
            "pros": "Reduces overfitting, speeds up training, removes multicollinearity",
            "cons": "Loses interpretability, linear transformation only",
            "complexity": "Medium",
            "type": "Dimensionality Reduction",
            "status": "📋 Planned - Phase 3"
        },
        "Hierarchical Clustering": {
            "description": "Creates tree-like cluster hierarchy using linkage criteria (agglomerative or divisive).",
            "use_cases": "Phylogenetic analysis, social network analysis, image segmentation",
            "pros": "No need to specify number of clusters, creates interpretable hierarchy",
            "cons": "Computationally expensive O(n³), sensitive to noise",
            "complexity": "High",
            "type": "Clustering",
            "status": "📋 Planned - Phase 3"
        }
    }
    
    SEMI_SUPERVISED_ALGORITHMS = {
        "Label Propagation": {
            "description": "Graph-based algorithm that propagates labels from labeled to unlabeled data through similarity graphs.",
            "use_cases": "Text classification with few labels, image annotation, social media analysis",
            "pros": "Works with few labeled examples, natural uncertainty estimation",
            "cons": "Requires good similarity metric, computationally expensive for large graphs",
            "complexity": "High",
            "type": "Classification",
            "status": "📋 Planned - Phase 4"
        },
        "Self-Training": {
            "description": "Iteratively trains on labeled data, predicts unlabeled data, adds confident predictions to training set.",
            "use_cases": "NLP with limited annotations, medical diagnosis, web page classification",
            "pros": "Simple to implement, works with any base classifier",
            "cons": "Can amplify errors, requires good confidence estimation",
            "complexity": "Medium",
            "type": "Classification",
            "status": "📋 Planned - Phase 4"
        },
        "Co-Training": {
            "description": "Uses two different views of data to train separate classifiers that teach each other.",
            "use_cases": "Web page classification, email classification, multi-modal learning",
            "pros": "Leverages multiple feature views, reduces overfitting",
            "cons": "Requires conditionally independent views, complex setup",
            "complexity": "High",
            "type": "Classification",
            "status": "📋 Planned - Phase 4"
        },
        "Semi-Supervised SVM": {
            "description": "Extends SVM to work with both labeled and unlabeled data using transductive learning.",
            "use_cases": "Text mining, bioinformatics, computer vision with limited labels",
            "pros": "Leverages unlabeled data effectively, maintains SVM advantages",
            "cons": "Non-convex optimization, computationally challenging",
            "complexity": "High",
            "type": "Classification",
            "status": "📋 Planned - Phase 4"
        }
    }

class MainWindow:
    """Main application window with categorized algorithm display."""
    
    def __init__(self):
        self.root = tk.Tk()
        self.root.title("Machine Learning Framework Explorer")
        self.root.geometry("1400x900")
        self.root.configure(bg='#f0f0f0')
        
        # Current selection tracking
        self.current_algorithm = None
        self.current_category = None
        self.listbox_widgets = {}  # Store references to listbox widgets
        
        # Create main interface
        self.create_widgets()
        
    def create_widgets(self):
        """Create and arrange GUI widgets with categorized tabs."""
        
        # Header
        header_frame = tk.Frame(self.root, bg='#2c3e50', height=80)
        header_frame.pack(fill='x', padx=5, pady=5)
        header_frame.pack_propagate(False)
        
        title_label = tk.Label(
            header_frame, 
            text="🤖 Machine Learning Framework Explorer", 
            font=('Arial', 18, 'bold'),
            bg='#2c3e50', 
            fg='white'
        )
        title_label.pack(pady=20)
        
        # Main content area
        main_frame = tk.Frame(self.root, bg='#f0f0f0')
        main_frame.pack(fill='both', expand=True, padx=10, pady=5)
        
        # Left panel - Categorized algorithm tabs
        left_frame = tk.Frame(main_frame, bg='white', width=400)
        left_frame.pack(side='left', fill='y', padx=(0, 10))
        left_frame.pack_propagate(False)
        
        # Create notebook for categories
        self.notebook = ttk.Notebook(left_frame)
        self.notebook.pack(fill='both', expand=True, padx=10, pady=10)
        
        # Create tabs for each category
        self.create_category_tabs()
        
        # Right panel - Algorithm details
        right_frame = tk.Frame(main_frame, bg='white')
        right_frame.pack(side='right', fill='both', expand=True)
        
        # Details text area
        self.details_text = scrolledtext.ScrolledText(
            right_frame,
            wrap=tk.WORD,
            font=('Arial', 11),
            bg='#f8f9fa',
            padx=15,
            pady=15
        )
        self.details_text.pack(fill='both', expand=True, padx=10, pady=10)
        
        # Button frame
        button_frame = tk.Frame(right_frame, bg='white')
        button_frame.pack(fill='x', padx=10, pady=10)
        
        # Action buttons
        ttk.Button(
            button_frame,
            text="🚀 Run Algorithm",
            command=self.run_algorithm
        ).pack(side='left', padx=5)
        
        ttk.Button(
            button_frame,
            text="📊 View Examples",
            command=self.view_examples
        ).pack(side='left', padx=5)
        
        ttk.Button(
            button_frame,
            text="📈 Visualize Algorithm",
            command=self.visualize_algorithm
        ).pack(side='left', padx=5)
        
        ttk.Button(
            button_frame,
            text="🔬 Compare Performance",
            command=self.compare_performance
        ).pack(side='left', padx=5)
        
        # Show initial message
        self.show_welcome_message()
        
    def create_category_tabs(self):
        """Create tabs for different algorithm categories."""
        
        # Supervised Learning Tab
        supervised_frame = ttk.Frame(self.notebook)
        self.notebook.add(supervised_frame, text='🎯 Supervised (7)')
        self.create_algorithm_list(supervised_frame, MLAlgorithmInfo.SUPERVISED_ALGORITHMS, 'supervised')
        
        # Unsupervised Learning Tab
        unsupervised_frame = ttk.Frame(self.notebook)
        self.notebook.add(unsupervised_frame, text='🔍 Unsupervised (4)')
        self.create_algorithm_list(unsupervised_frame, MLAlgorithmInfo.UNSUPERVISED_ALGORITHMS, 'unsupervised')
        
        # Semi-Supervised Learning Tab
        semi_supervised_frame = ttk.Frame(self.notebook)
        self.notebook.add(semi_supervised_frame, text='🎭 Semi-Supervised (4)')
        self.create_algorithm_list(semi_supervised_frame, MLAlgorithmInfo.SEMI_SUPERVISED_ALGORITHMS, 'semi_supervised')
        
    def create_algorithm_list(self, parent, algorithms, category):
        """Create algorithm list for a specific category."""
        
        # Category description
        descriptions = {
            'supervised': "Uses labeled data to learn patterns and make predictions",
            'unsupervised': "Finds hidden patterns in data without labels",
            'semi_supervised': "Combines labeled and unlabeled data for learning"
        }
        
        desc_label = tk.Label(
            parent,
            text=descriptions[category],
            font=('Arial', 10),
            bg='white',
            fg='#666',
            wraplength=350
        )
        desc_label.pack(pady=(10, 5), padx=10)
        
        # Algorithm listbox
        listbox = tk.Listbox(
            parent,
            font=('Arial', 10),
            selectmode='single',
            bg='#f8f9fa',
            selectbackground='#007bff',
            selectforeground='white'
        )
        listbox.pack(fill='both', expand=True, padx=10, pady=5)
        
        # Store reference to listbox
        self.listbox_widgets[category] = listbox
        
        # Populate algorithm list with status
        for algorithm, info in algorithms.items():
            status_icon = info['status'].split()[0]
            display_text = f"{status_icon} {algorithm}"
            listbox.insert(tk.END, display_text)
        
        # Bind selection event
        listbox.bind('<<ListboxSelect>>', 
                    lambda event, cat=category, alg_dict=algorithms, lb=listbox: 
                    self.on_algorithm_select(event, cat, alg_dict, lb))
        
    def on_algorithm_select(self, event, category, algorithms, listbox):
        """Handle algorithm selection."""
        selection = listbox.curselection()
        if not selection:
            return
            
        # Get algorithm name (remove status icon)
        selected_text = listbox.get(selection[0])
        algorithm_name = ' '.join(selected_text.split()[1:])  # Remove first word (status icon)
        
        # Store current selection
        self.current_algorithm = algorithm_name
        self.current_category = category
        
        self.show_algorithm_details(algorithm_name, algorithms, category)
        
    def show_algorithm_details(self, algorithm_name, algorithms, category):
        """Display detailed information about selected algorithm."""
        if algorithm_name not in algorithms:
            return
            
        info = algorithms[algorithm_name]
        
        # Category names for display
        category_names = {
            'supervised': "Supervised Learning",
            'unsupervised': "Unsupervised Learning", 
            'semi_supervised': "Semi-Supervised Learning"
        }
        
        # Enhanced details with implementation status
        implementation_guide = ""
        if "Complete" in info['status']:
            implementation_guide = f"""
🚀 IMPLEMENTATION AVAILABLE!

📁 Location: src/machine_learning_model/supervised/
📖 Examples: examples/supervised_examples/
🧪 Tests: tests/test_supervised/

⚡ Quick Start:
```python
from machine_learning_model.supervised import {algorithm_name.replace(' ', '')}

model = {algorithm_name.replace(' ', '')}()
model.fit(X_train, y_train)
predictions = model.predict(X_test)
```
"""
        elif "Ready" in info['status']:
            implementation_guide = """
📋 Ready for implementation - documentation and API design complete.
"""
        else:
            implementation_guide = """
⏳ Planned for future implementation phases.
"""
        
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

{'='*60}

💡 Quick Tips:
• Start with simpler algorithms (Linear/Logistic Regression) for baseline
• Use ensemble methods (Random Forest, Gradient Boosting) for better accuracy
• Consider your data size and computational resources
• Always validate your model on unseen data

🔍 Want to see this algorithm in action? Click 'Implement Algorithm' below!
        """
        
        self.details_text.delete(1.0, tk.END)
        self.details_text.insert(1.0, details_text)
        
    def show_welcome_message(self):
        """Display welcome message."""
        welcome_text = """
🎯 Welcome to Machine Learning Framework Explorer!

This interactive tool helps you explore and understand different machine learning algorithms.

📋 How to use:
1. Select an algorithm from the tabs on the left
2. Read the detailed description and use cases
3. Use the buttons below to implement or learn more

🔍 Features:
• Detailed algorithm descriptions
• Real-world use cases and examples  
• Pros and cons analysis
• Complexity ratings
• Implementation status tracking

🚀 Current Status:
✅ Decision Trees - Complete and Ready
✅ Random Forest - Complete and Ready
🔄 Support Vector Machine - Next Target

Select an algorithm from the tabs to get started! 🚀
        """
        self.details_text.delete(1.0, tk.END)
        self.details_text.insert(1.0, welcome_text)
        
    def get_current_selection(self):
        """Get the currently selected algorithm and category."""
        if self.current_algorithm and self.current_category:
            return self.current_algorithm, self.current_category
        return None, None
        
    def run_algorithm(self):
        """Run the selected algorithm with demo data."""
        algorithm, category = self.get_current_selection()
        if not algorithm:
            messagebox.showwarning("No Selection", "Please select an algorithm first!")
            return
        
        # Check if implementation is available
        if category == 'supervised':
            algorithms = MLAlgorithmInfo.SUPERVISED_ALGORITHMS
        elif category == 'unsupervised':
            algorithms = MLAlgorithmInfo.UNSUPERVISED_ALGORITHMS
        else:
            algorithms = MLAlgorithmInfo.SEMI_SUPERVISED_ALGORITHMS
            
        status = algorithms[algorithm]['status']
        
        if "Complete" in status:
            try:
                # Import and run algorithm demo
                from machine_learning_model.visualization.algorithm_visualizer import (
                    run_algorithm_demo,
                )
                
                messagebox.showinfo(
                    "Running Algorithm",
                    f"🚀 Running {algorithm} with demo data...\n\n"
                    f"This will:\n"
                    f"• Load appropriate demo dataset\n"
                    f"• Train the {algorithm} model\n"
                    f"• Generate performance visualizations\n"
                    f"• Show results in new window\n\n"
                    f"Please wait..."
                )
                
                # Run the demo
                results = run_algorithm_demo(algorithm)
                
                # Show results
                result_message = f"✅ {algorithm} completed successfully!\n\n"
                
                if 'accuracy' in results:
                    result_message += f"📊 Accuracy: {results['accuracy']:.3f}\n"
                if 'r2_score' in results:
                    result_message += f"📊 R² Score: {results['r2_score']:.3f}\n"
                if 'oob_score' in results:
                    result_message += f"📊 OOB Score: {results['oob_score']:.3f}\n"
                
                result_message += f"\n📁 Visualization saved to:\ntest-outputs/artifacts/{algorithm.lower().replace(' ', '_')}_demo.png"
                
                messagebox.showinfo("Algorithm Results", result_message)
                
            except Exception as e:
                messagebox.showerror(
                    "Execution Error",
                    f"❌ Failed to run {algorithm}:\n\n{str(e)}\n\n"
                    f"Please check that all dependencies are installed."
                )
        
        elif "Ready" in status:
            messagebox.showinfo(
                "Implementation Coming Soon",
                f"📋 {algorithm} implementation is ready!\n\n"
                f"The algorithm will be implemented soon.\n"
                f"Documentation and API design are complete.\n\n"
                f"🔄 Status: Ready for development"
            )
        else:
            messagebox.showinfo(
                "Planned Implementation",
                f"⏳ {algorithm} is planned for future phases.\n\n"
                f"Check the project roadmap for timeline details.\n\n"
                f"📅 Estimated timeline: See project_plan.md"
            )
    
    def view_examples(self):
        """Open and run example files for the selected algorithm."""
        algorithm, category = self.get_current_selection()
        if not algorithm:
            messagebox.showwarning("No Selection", "Please select an algorithm first!")
            return
        
        # Map algorithm names to example files
        example_mapping = {
            'Decision Trees': 'decision_tree_example.py',
            'Random Forest': 'random_forest_example.py', 
            'Linear Regression': 'linear_regression_example.py',
            'Logistic Regression': 'logistic_regression_example.py'
        }
        
        if algorithm in example_mapping:
            example_file = example_mapping[algorithm]
            example_path = f"examples/supervised_examples/{example_file}"
            
            if os.path.exists(example_path):
                # Ask user if they want to run the example
                response = messagebox.askyesno(
                    "Run Example",
                    f"📖 {algorithm} Example Available!\n\n"
                    f"📁 File: {example_path}\n\n"
                    f"Would you like to run this example now?\n\n"
                    f"📊 This will:\n"
                    f"• Load multiple datasets\n"
                    f"• Train and evaluate the algorithm\n"
                    f"• Generate detailed visualizations\n"
                    f"• Show performance analysis"
                )
                
                if response:
                    try:
                        # Run the example script
                        messagebox.showinfo(
                            "Running Example",
                            f"🚀 Running {algorithm} example...\n\n"
                            f"This may take a few moments.\n"
                            f"Check the console for progress and results."
                        )
                        
                        # Execute the example script
                        result = subprocess.run([sys.executable, example_path], 
                                              capture_output=True, text=True, cwd=".")
                        
                        if result.returncode == 0:
                            messagebox.showinfo(
                                "Example Completed",
                                f"✅ {algorithm} example completed successfully!\n\n"
                                f"📁 Check 'test-outputs/artifacts/' for visualizations\n\n"
                                f"🖥️ Console output:\n{result.stdout[:200]}..."
                            )
                        else:
                            messagebox.showerror(
                                "Example Failed",
                                f"❌ Example failed to run:\n\n{result.stderr[:300]}..."
                            )
                    
                    except Exception as e:
                        messagebox.showerror(
                            "Execution Error",
                            f"❌ Failed to run example:\n\n{str(e)}"
                        )
            else:
                messagebox.showwarning(
                    "Example Not Found",
                    f"📁 Example file not found:\n{example_path}\n\n"
                    f"Please ensure the examples directory is properly set up."
                )
        else:
            messagebox.showinfo(
                "Examples Coming Soon", 
                f"📖 Examples for {algorithm}\n\n"
                f"Examples are being developed and will include:\n"
                f"• Jupyter notebooks with step-by-step tutorials\n"
                f"• Python scripts with complete implementations\n"
                f"• Real-world dataset examples\n"
                f"• Performance comparisons\n\n"
                f"Check back soon! 📚"
            )
    
    def visualize_algorithm(self):
        """Create interactive visualizations for the selected algorithm."""
        algorithm, category = self.get_current_selection()
        if not algorithm:
            messagebox.showwarning("No Selection", "Please select an algorithm first!")
            return
        
        # Check if visualization is available
        available_visualizations = ["Decision Trees", "Random Forest", "Linear Regression", "Logistic Regression"]
        
        if algorithm in available_visualizations:
            try:
                # Import and run visualization
                from machine_learning_model.visualization.algorithm_visualizer import (
                    run_algorithm_demo,
                )
                
                messagebox.showinfo(
                    "Creating Visualization",
                    f"📊 Creating {algorithm} visualization...\n\n"
                    f"This will:\n"
                    f"• Load demo dataset\n"
                    f"• Train the algorithm\n"
                    f"• Generate interactive plots\n"
                    f"• Show decision boundaries (if applicable)\n"
                    f"• Display performance metrics\n\n"
                    f"Please wait..."
                )
                
                # Run the visualization
                results = run_algorithm_demo(algorithm)
                
                # Show completion message
                message = f"✅ {algorithm} visualization completed!\n\n"
                message += f"📊 Results:\n"
                
                if 'accuracy' in results:
                    message += f"• Accuracy: {results['accuracy']:.3f}\n"
                if 'r2_score' in results:
                    message += f"• R² Score: {results['r2_score']:.3f}\n"
                if 'oob_score' in results:
                    message += f"• OOB Score: {results['oob_score']:.3f}\n"
                
                message += f"\n📁 Saved to: test-outputs/artifacts/{algorithm.lower().replace(' ', '_')}_demo.png"
                
                messagebox.showinfo("Visualization Complete", message)
                
            except Exception as e:
                messagebox.showerror(
                    "Visualization Error",
                    f"❌ Failed to create visualization:\n\n{str(e)}\n\n"
                    f"Please ensure all dependencies are installed:\n"
                    f"• matplotlib\n• seaborn\n• scikit-learn"
                )
        else:
            messagebox.showinfo(
                "Visualization Coming Soon",
                f"📊 Visualization for {algorithm}\n\n"
                f"Interactive visualizations are being developed and will include:\n"
                f"• Decision boundaries (for classification)\n"
                f"• Feature importance plots\n"
                f"• Performance metrics visualization\n"
                f"• Interactive parameter exploration\n\n"
                f"Available for: {', '.join(available_visualizations)}\n\n"
                f"Feature coming soon for other algorithms! 📈"
            )
    
    def compare_performance(self):
        """Handle compare performance button click."""
        algorithm, category = self.get_current_selection()
        if not algorithm:
            messagebox.showwarning("No Selection", "Please select an algorithm first!")
            return
        
        # Check if comprehensive comparison is available
        import os
        comparison_file = "src/machine_learning_model/evaluation/performance_comparison.py"
        
        if os.path.exists(comparison_file):
            messagebox.showinfo(
                "Performance Comparison Available", 
                f"📈 Performance Comparison for {algorithm}\n\n"
                f"🚀 To run comprehensive comparison:\n"
                f"python -m src.machine_learning_model.evaluation.performance_comparison\n\n"
                f"📊 Available Analysis:\n"
                f"• Cross-validation scores\n"
                f"• Training time comparison\n"
                f"• Multiple dataset benchmarks\n"
                f"• Speed vs accuracy trade-offs\n"
                f"• Comprehensive report generation\n\n"
                f"Results saved to: test-outputs/artifacts/"
            )
        else:
            messagebox.showinfo(
                "Performance Comparison", 
                f"📈 Performance analysis for {algorithm}\n\n"
                f"This will show:\n"
                f"• Benchmark results against scikit-learn\n"
                f"• Performance on different datasets\n"
                f"• Speed and accuracy comparisons\n"
                f"• Hyperparameter sensitivity analysis\n\n"
                f"Feature coming soon! 📊"
            )

    def run(self):
        """Start the GUI application."""
        self.root.mainloop()

def main():
    """Main entry point for the GUI application."""
    app = MainWindow()
    app.run()

if __name__ == "__main__":
    main()