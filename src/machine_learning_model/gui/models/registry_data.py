"""Static algorithm metadata separated from GUI logic.

This file was extracted from the legacy main_window_pyqt6 module to allow reuse and testing.
"""
from __future__ import annotations

SUPERVISED_ALGORITHMS_DATA = {
    # Minimal subset; could be fully copied if needed. Keeping all current entries for parity.
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
        "status": "✅ Complete - Prototype",
        "implementation_status": "complete",
        "examples": "Text document classification, face recognition, protein fold prediction, handwritten digit recognition"
    },
    "XGBoost": {
        "description": "Advanced gradient boosting framework optimized for speed and performance with regularization to prevent overfitting.",
        "use_cases": "Kaggle competitions, structured data prediction, feature selection, ranking problems, large-scale machine learning",
        "pros": "State-of-the-art performance, built-in regularization, handles missing values, parallel processing, cross-validation support",
        "cons": "Many hyperparameters to tune, computationally intensive, requires careful tuning, can overfit, black box model",
        "complexity": "High",
        "type": "Both",
        "status": "✅ Complete - Prototype",
        "implementation_status": "complete",
        "examples": "Kaggle competition winning models, customer lifetime value prediction, ad click prediction, risk modeling"
    },
    "Neural Networks": {
        "description": "Multi-layered networks of interconnected nodes mimicking brain neurons for complex pattern recognition and function approximation.",
        "use_cases": "Image recognition, natural language processing, speech recognition, game playing, time series forecasting",
        "pros": "Universal approximator, handles complex non-linear relationships, automatic feature learning, scalable to large datasets",
        "cons": "Requires large datasets, computationally expensive, black box, prone to overfitting, many hyperparameters",
        "complexity": "High",
        "type": "Both",
        "status": "✅ Complete - Prototype",
        "implementation_status": "complete",
        "examples": "Image classification with CNNs, language translation with transformers, game AI with deep reinforcement learning"
    },
}

UNSUPERVISED_ALGORITHMS_DATA = {
    "K-Means Clustering": {
        "description": "Partitions data into k clusters by iteratively minimizing within-cluster sum of squares using centroid-based approach.",
        "use_cases": "Customer segmentation, image compression, market research, data compression, anomaly detection preprocessing",
        "pros": "Simple and fast, works well with spherical clusters, scales to large datasets, guaranteed convergence",
        "cons": "Must specify k beforehand, sensitive to initialization and outliers, assumes spherical clusters, struggles with varying densities",
        "complexity": "Medium",
        "type": "Clustering",
        "status": "✅ Complete - Prototype",
        "implementation_status": "complete",
        "examples": "Customer segmentation for marketing, color quantization in images, organizing news articles by topic"
    },
    "DBSCAN": {
        "description": "Density-based clustering that groups together points in high-density areas and marks outliers in low-density regions.",
        "use_cases": "Anomaly detection, image processing, social network analysis, fraud detection, spatial data analysis",
        "pros": "Automatically determines number of clusters, handles noise and outliers well, finds arbitrary shaped clusters",
        "cons": "Sensitive to hyperparameters (eps, min_samples), struggles with varying densities, difficult to use in high dimensions",
        "complexity": "Medium",
        "type": "Clustering",
        "status": "✅ Complete - Prototype",
        "implementation_status": "complete",
        "examples": "Detecting fraudulent transactions, finding crime hotspots, identifying communities in social networks"
    },
    "Principal Component Analysis": {
        "description": "Dimensionality reduction technique that projects data onto principal components that capture maximum variance in the data.",
        "use_cases": "Data visualization, feature reduction, noise reduction, data compression, exploratory data analysis",
        "pros": "Reduces overfitting, speeds up training, removes multicollinearity, provides interpretable components, reduces storage",
        "cons": "Loses interpretability of original features, linear transformation only, sensitive to scaling, may lose important information",
        "complexity": "Medium",
        "type": "Dimensionality Reduction",
        "status": "✅ Complete - Prototype",
        "implementation_status": "complete",
        "examples": "Visualizing high-dimensional data in 2D/3D, reducing features before classification, image compression"
    },
    "Hierarchical Clustering": {
        "description": "Creates tree-like cluster hierarchy using linkage criteria, either agglomerative (bottom-up) or divisive (top-down) approach.",
        "use_cases": "Phylogenetic analysis, social network analysis, image segmentation, organizing product catalogs, taxonomy creation",
        "pros": "No need to specify number of clusters, creates interpretable hierarchy, deterministic results, handles any distance metric",
        "cons": "Computationally expensive O(n³), sensitive to noise and outliers, difficult to handle large datasets, sensitive to metric choice",
        "complexity": "High",
        "type": "Clustering",
        "status": "✅ Complete - Prototype",
        "implementation_status": "complete",
        "examples": "Building species evolution trees, organizing company departments, creating product category hierarchies"
    },
}

SEMI_SUPERVISED_ALGORITHMS_DATA = {
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
    },
}

__all__ = [
    "SUPERVISED_ALGORITHMS_DATA",
    "UNSUPERVISED_ALGORITHMS_DATA",
    "SEMI_SUPERVISED_ALGORITHMS_DATA",
]
