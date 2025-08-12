"""Algorithm execution utilities (Step 3 of MVC refactor).

Encapsulates the logic previously embedded in QuickRunPanel.run_algorithm.
"""
from __future__ import annotations

from dataclasses import dataclass
import numpy as np
from typing import Any

from .data_synthesizer import SyntheticDataSpec, generate_synthetic_data

HAVE_SVM = True
HAVE_XGBOOST = True
HAVE_NN = True
HAVE_UNSUPERVISED = True

try:  # Custom implementations
    from machine_learning_model.supervised.decision_tree import (
        DecisionTreeClassifier,
        DecisionTreeRegressor,
    )
    from machine_learning_model.supervised.random_forest import (
        RandomForestClassifier,
        RandomForestRegressor,
    )
except Exception:  # pragma: no cover

    class DecisionTreeClassifier:  # type: ignore
        def fit(self, X, y):
            pass

        def predict(self, X):
            return np.zeros(len(X))

    class DecisionTreeRegressor:  # type: ignore
        def fit(self, X, y):
            pass

        def predict(self, X):
            return np.zeros(len(X))

    class RandomForestClassifier:  # type: ignore
        def __init__(self, n_estimators=25):
            pass

        def fit(self, X, y):
            pass

        def predict(self, X):
            return np.zeros(len(X))

    class RandomForestRegressor:  # type: ignore
        def __init__(self, n_estimators=25):
            pass

        def fit(self, X, y):
            pass

        def predict(self, X):
            return np.zeros(len(X))

try:
    from sklearn.svm import SVC, SVR
    from sklearn.preprocessing import StandardScaler
except Exception:
    HAVE_SVM = False
try:
    from xgboost import XGBClassifier, XGBRegressor  # type: ignore
except Exception:
    HAVE_XGBOOST = False
try:
    from sklearn.neural_network import MLPClassifier, MLPRegressor
except Exception:
    HAVE_NN = False
try:
    from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering
    from sklearn.decomposition import PCA
    from sklearn.metrics import silhouette_score
except Exception:
    HAVE_UNSUPERVISED = False


@dataclass(slots=True)
class RunResult:
    algorithm: str
    task: str
    success: bool
    details: str
    metrics: dict[str, Any]
    warnings: list[str]


def _linear_regression_metrics(X: np.ndarray, y: np.ndarray) -> RunResult:
    coef, *_ = np.linalg.lstsq(X, y, rcond=None)
    preds = X @ coef
    mse = float(np.mean((preds - y) ** 2))
    ss_res = float(np.sum((y - preds) ** 2))
    ss_tot = float(np.sum((y - np.mean(y)) ** 2))
    r2 = 1.0 - (ss_res / ss_tot if ss_tot != 0 else 0.0)
    details = (
        "✅ Linear Regression Results:\n"
        f"MSE: {mse:.4f} | R²: {r2:.4f}\n"
        f"Samples: {X.shape[0]} | Features: {X.shape[1]}"
    )
    return RunResult(
        algorithm="Linear Regression",
        task="regression",
        success=True,
        details=details,
        metrics={"mse": mse, "r2": r2},
        warnings=[],
    )


def run_algorithm(algorithm: str, task: str, spec: SyntheticDataSpec | None = None) -> RunResult:
    if spec is None:
        spec = SyntheticDataSpec(task=task)
    try:
        X, y = generate_synthetic_data(spec)
        if algorithm in {
            "K-Means Clustering",
            "DBSCAN",
            "Principal Component Analysis",
            "Hierarchical Clustering",
        }:
            if not HAVE_UNSUPERVISED:
                return RunResult(algorithm, task, False, "❌ scikit-learn not available for Unsupervised algorithms", {}, [])
            scaler = StandardScaler() if HAVE_SVM else None
            X_scaled = scaler.fit_transform(X) if scaler else X
            if algorithm == "K-Means Clustering":
                k = 3
                model = KMeans(n_clusters=k, random_state=42, n_init=10)
                labels = model.fit_predict(X_scaled)
                inertia = float(model.inertia_)
                sil = float(silhouette_score(X_scaled, labels)) if len(set(labels)) > 1 else -1.0
                counts = {c: int(np.sum(labels == c)) for c in sorted(set(labels))}
                details = (
                    "✅ K-Means Results:\n"
                    f"Clusters: {k} | Inertia: {inertia:.2f} | Silhouette: {sil:.3f}\n"
                    f"Counts: {counts}\nSamples: {spec.n_samples} | Features: {spec.n_features}"
                )
                return RunResult(algorithm, task, True, details, {"inertia": inertia, "silhouette": sil}, [])
            if algorithm == "DBSCAN":
                model = DBSCAN(eps=0.9, min_samples=5)
                labels = model.fit_predict(X_scaled)
                unique = [c for c in set(labels) if c != -1]
                n_clusters = len(unique)
                noise = int(np.sum(labels == -1))
                sil = float(silhouette_score(X_scaled, labels)) if n_clusters > 1 else -1.0
                details = (
                    "✅ DBSCAN Results:\n"
                    f"Clusters: {n_clusters} | Noise points: {noise} | Silhouette: {sil:.3f}\n"
                    f"Samples: {spec.n_samples} | Features: {spec.n_features}"
                )
                return RunResult(algorithm, task, True, details, {"clusters": n_clusters, "noise": noise, "silhouette": sil}, [])
            if algorithm == "Principal Component Analysis":
                n_components = min(3, spec.n_features)
                pca = PCA(n_components=n_components, random_state=42)
                pca.fit(X_scaled)
                explained = pca.explained_variance_ratio_
                cumulative = float(np.cumsum(explained)[-1])
                explained_fmt = ", ".join(f"{v:.2f}" for v in explained)
                details = (
                    "✅ PCA Results:\n"
                    f"Components: {n_components} | Var ratios: [{explained_fmt}] | Cumulative: {cumulative:.2f}\n"
                    f"Samples: {spec.n_samples} | Original Features: {spec.n_features}"
                )
                return RunResult(algorithm, task, True, details, {"cumulative_variance": cumulative}, [])
            k = 3
            model = AgglomerativeClustering(n_clusters=k)
            labels = model.fit_predict(X_scaled)
            sil = float(silhouette_score(X_scaled, labels)) if k > 1 else -1.0
            counts = {c: int(np.sum(labels == c)) for c in sorted(set(labels))}
            details = (
                "✅ Hierarchical Clustering Results:\n"
                f"Clusters: {k} | Silhouette: {sil:.3f} | Counts: {counts}\n"
                f"Samples: {spec.n_samples} | Features: {spec.n_features}"
            )
            return RunResult(algorithm, task, True, details, {"silhouette": sil}, [])
        if algorithm == "Linear Regression" and task == "regression":
            assert y is not None
            return _linear_regression_metrics(X, y)
        if algorithm == "Support Vector Machine":
            if not HAVE_SVM:
                return RunResult(algorithm, task, False, "❌ scikit-learn not available for SVM", {}, [])
            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(X)
            if task == "classification":
                assert y is not None
                model = SVC(kernel="rbf", gamma="scale", C=1.0, random_state=42)
                model.fit(X_scaled, y)
                preds = model.predict(X_scaled)
                acc = float(np.mean(preds == y))
                details = (
                    "✅ Support Vector Machine Classification Results:\n"
                    f"Accuracy: {acc:.3f} | Classes: {len(np.unique(y))}\n"
                    f"Samples: {spec.n_samples} | Features: {spec.n_features} (scaled)"
                )
                return RunResult(algorithm, task, True, details, {"accuracy": acc}, [])
            assert y is not None
            model = SVR(kernel="rbf", C=1.0)
            model.fit(X_scaled, y)
            preds = model.predict(X_scaled)
            mse = float(np.mean((preds - y) ** 2))
            mae = float(np.mean(np.abs(preds - y)))
            details = (
                "✅ Support Vector Machine Regression Results:\n"
                f"MSE: {mse:.4f} | MAE: {mae:.4f}\n"
                f"Samples: {spec.n_samples} | Features: {spec.n_features} (scaled)"
            )
            return RunResult(algorithm, task, True, details, {"mse": mse, "mae": mae}, [])
        if algorithm == "XGBoost":
            if not HAVE_XGBOOST:
                return RunResult(algorithm, task, False, "❌ XGBoost library not available", {}, [])
            assert y is not None
            if task == "classification":
                model = XGBClassifier(
                    n_estimators=100,
                    max_depth=3,
                    learning_rate=0.1,
                    subsample=0.9,
                    colsample_bytree=0.9,
                    reg_lambda=1,
                    eval_metric="logloss",
                    verbosity=0,
                )
            else:
                model = XGBRegressor(
                    n_estimators=120,
                    max_depth=3,
                    learning_rate=0.1,
                    subsample=0.9,
                    colsample_bytree=0.9,
                    reg_lambda=1,
                    objective="reg:squarederror",
                    verbosity=0,
                )
            model.fit(X, y)
            preds = model.predict(X)
            if task == "classification":
                acc = float(np.mean(preds == y))
                details = (
                    "✅ XGBoost Classification Results:\n"
                    f"Accuracy: {acc:.3f} | Trees: 100\n"
                    f"Samples: {spec.n_samples} | Features: {spec.n_features}"
                )
                return RunResult(algorithm, task, True, details, {"accuracy": acc}, [])
            mse = float(np.mean((preds - y) ** 2))
            mae = float(np.mean(np.abs(preds - y)))
            details = (
                "✅ XGBoost Regression Results:\n"
                f"MSE: {mse:.4f} | MAE: {mae:.4f}\n"
                f"Samples: {spec.n_samples} | Features: {spec.n_features}"
            )
            return RunResult(algorithm, task, True, details, {"mse": mse, "mae": mae}, [])
        if algorithm == "Neural Networks":
            if not HAVE_NN:
                return RunResult(algorithm, task, False, "❌ scikit-learn not available for Neural Networks", {}, [])
            assert y is not None
            if task == "classification":
                model = MLPClassifier(hidden_layer_sizes=(32, 16), activation="relu", max_iter=300, random_state=42)
                model.fit(X, y)
                preds = model.predict(X)
                acc = float(np.mean(preds == y))
                details = (
                    "✅ Neural Network Classification Results:\n"
                    f"Accuracy: {acc:.3f} | Layers: 2 hidden\n"
                    f"Samples: {spec.n_samples} | Features: {spec.n_features}"
                )
                return RunResult(algorithm, task, True, details, {"accuracy": acc}, [])
            model = MLPRegressor(hidden_layer_sizes=(64, 32), activation="relu", max_iter=300, random_state=42)
            model.fit(X, y)
            preds = model.predict(X)
            mse = float(np.mean((preds - y) ** 2))
            mae = float(np.mean(np.abs(preds - y)))
            details = (
                "✅ Neural Network Regression Results:\n"
                f"MSE: {mse:.4f} | MAE: {mae:.4f}\n"
                f"Samples: {spec.n_samples} | Features: {spec.n_features}"
            )
            return RunResult(algorithm, task, True, details, {"mse": mse, "mae": mae}, [])
        if algorithm in {"Decision Trees", "Random Forest"}:
            assert y is not None
            if algorithm == "Decision Trees":
                model = DecisionTreeClassifier() if task == "classification" else DecisionTreeRegressor()
            else:
                model = (
                    RandomForestClassifier(n_estimators=50)
                    if task == "classification"
                    else RandomForestRegressor(n_estimators=50)
                )
            model.fit(X, y)
            preds = model.predict(X)
            if task == "classification":
                acc = float(np.mean(preds == y))
                details = (
                    f"✅ {algorithm} Classification Results:\n"
                    f"Accuracy: {acc:.3f} | Classes: {len(np.unique(y))}\n"
                    f"Samples: {spec.n_samples} | Features: {spec.n_features}"
                )
                return RunResult(algorithm, task, True, details, {"accuracy": acc}, [])
            mse = float(np.mean((preds - y) ** 2))
            mae = float(np.mean(np.abs(preds - y)))
            details = (
                f"✅ {algorithm} Regression Results:\n"
                f"MSE: {mse:.4f} | MAE: {mae:.4f}\n"
                f"Samples: {spec.n_samples} | Features: {spec.n_features}"
            )
            return RunResult(algorithm, task, True, details, {"mse": mse, "mae": mae}, [])
        return RunResult(algorithm, task, False, f"❌ {algorithm} not implemented for {task} task", {}, [])
    except Exception as exc:  # pragma: no cover
        return RunResult(algorithm, task, False, f"❌ Error running {algorithm}: {exc}", {}, [repr(exc)])


__all__ = ["run_algorithm", "RunResult"]
