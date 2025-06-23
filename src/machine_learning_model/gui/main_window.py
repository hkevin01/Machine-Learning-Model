"""
Main GUI window for Machine Learning Framework Explorer.
Provides an interactive interface for exploring ML algorithms categorized by learning type.
"""

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
            "status": "📋 Planned - Next Phase"
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
        
        # Current selection
        self.current_algorithm = None
        self.current_category = None
        
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
            text="🚀 Implement Algorithm",
            command=self.implement_algorithm
        ).pack(side='left', padx=5)
        
        ttk.Button(
            button_frame,
            text="📊 View Examples",
            command=self.view_examples
        ).pack(side='left', padx=5)
        
        ttk.Button(
            button_frame,
            text="📈 Compare Performance",
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
        
        # Populate algorithm list with status
        for algorithm, info in algorithms.items():
            status_icon = info['status'].split()[0]
            display_text = f"{status_icon} {algorithm}"
            listbox.insert(tk.END, display_text)
        
        # Bind selection event
        listbox.bind('<<ListboxSelect>>', 
                    lambda event, cat=category, alg_dict=algorithms, lb=listbox: 
                    self.on_algorithm_select(event, cat, alg_dict, lb))
        
    def show_welcome_message(self):
        """Display welcome message."""
        welcome_text = """
🎯 Welcome to Machine Learning Framework Explorer!

This interactive tool helps you explore and understand different machine learning algorithms.

📋 How to use:
1. Select an algorithm from the list on the left
2. Read the detailed description and use cases
3. Use the buttons below to run examples or learn more

🔍 Features:
• Detailed algorithm descriptions
• Real-world use cases and examples
• Pros and cons analysis
• Complexity ratings
• Interactive examples

Select an algorithm from the list to get started! 🚀
        """
        self.details_text.delete(1.0, tk.END)
        self.details_text.insert(1.0, welcome_text)
        
    def on_algorithm_select(self, event, category, algorithms, listbox):
        """Handle algorithm selection."""
        selection = listbox.curselection()
        if not selection:
            return
            
        algorithm_name = listbox.get(selection[0])[2:]  # Remove status icon
        self.show_algorithm_details(algorithm_name, category)
        
    def show_algorithm_details(self, algorithm_name, category):
        """Display detailed information about selected algorithm."""
        if category == 'supervised':
            algorithms = MLAlgorithmInfo.SUPERVISED_ALGORITHMS
        elif category == 'unsupervised':
            algorithms = MLAlgorithmInfo.UNSUPERVISED_ALGORITHMS
        elif category == 'semi_supervised':
            algorithms = MLAlgorithmInfo.SEMI_SUPERVISED_ALGORITHMS
        else:
            return
            
        if algorithm_name not in algorithms:
            return
            
        info = algorithms[algorithm_name]
        
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
🎯 {algorithm_name} ({category})

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

🔍 Want to see this algorithm in action? Click 'Run Example' below!
        """
        
        self.details_text.delete(1.0, tk.END)
        self.details_text.insert(1.0, details_text)
        
    def implement_algorithm(self):
        """Run an example of the selected algorithm."""
        selection = self.get_current_selection()
        if not selection:
            messagebox.showwarning("No Selection", "Please select an algorithm first!")
            return
            
        algorithm_name, category = selection
        messagebox.showinfo(
            "Implement Algorithm", 
            f"Implementing example for {algorithm_name}...\n\n"
            "This would launch an interactive example with sample data.\n"
            "Feature coming soon! 🚀"
        )
        
    def view_examples(self):
        """Open external resources for learning."""
        selection = self.get_current_selection()
        if not selection:
            messagebox.showwarning("No Selection", "Please select an algorithm first!")
            return
            
        algorithm_name, category = selection
        messagebox.showinfo(
            "View Examples", 
            f"Opening example resources for {algorithm_name}...\n\n"
            "This would open relevant documentation, tutorials, or research papers.\n"
            "Feature coming soon! 📚"
        )
        
    def compare_performance(self):
        """Visualize the algorithm."""
        selection = self.get_current_selection()
        if not selection:
            messagebox.showwarning("No Selection", "Please select an algorithm first!")
            return
            
        algorithm_name, category = selection
        messagebox.showinfo(
            "Compare Performance", 
            f"Comparing performance for {algorithm_name}...\n\n"
            "This would show interactive plots and decision boundaries.\n"
            "Feature coming soon! 📊"
        )
        
    def get_current_selection(self):
        """Get the currently selected algorithm and category."""
        for tab in self.notebook.tabs():
            widget = self.notebook.nametowidget(tab)
            selection = widget.curselection()
            if selection:
                algorithm_name = widget.get(selection[0])[2:]  # Remove status icon
                category = tab.split(' ')[-1][1:-1]  # Extract category from tab text
                return (algorithm_name, category)
                
        return None
        
    def run(self):
        """Start the GUI application."""
        self.root.mainloop()

def main():
    """Main entry point for the GUI application."""
    app = MainWindow()
    app.run()

if __name__ == "__main__":
    main()