"""Algorithm execution utilities (Step 3 of MVC refactor).

Encapsulates the logic previously embedded in QuickRunPanel.run_algorithm.
"""
from __future__ import annotations

import time
from dataclasses import asdict, dataclass, field
from typing import Any

import numpy as np

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
    from sklearn.preprocessing import StandardScaler
    from sklearn.svm import SVC, SVR
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
    from sklearn.cluster import DBSCAN, AgglomerativeClustering, KMeans
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
    warning_levels: list[tuple[str, str]] = field(default_factory=list)
    execution_time: float = 0.0
    model_info: dict[str, Any] = field(default_factory=dict)
    performance_summary: str = ""
    recommendations: list[str] = field(default_factory=list)
    data_characteristics: dict[str, Any] = field(default_factory=dict)
    model_diagnostics: dict[str, Any] = field(default_factory=dict)
    reasoning: str = ""  # Narrative explanation of what happened and why

    def to_json(self) -> dict[str, Any]:  # Simple serializer
        return asdict(self)


def _create_enhanced_result(
    algorithm: str,
    task: str,
    success: bool,
    details: str,
    metrics: dict[str, Any],
    warnings: list[str],
    start_time: float,
    model_params: dict[str, Any] | None = None,
    additional_info: dict[str, Any] | None = None,
) -> RunResult:
    """Create a RunResult with enhanced fields for richer output."""
    execution_time = time.time() - start_time

    # Generate model info & data characteristics
    data_info = additional_info or {}
    model_params = model_params or {}
    model_info = {
        "algorithm_type": algorithm,
        "task_type": task,
        "parameters": model_params,
        "data_info": data_info,
    }

    # Generate performance summary & performance category
    if success:
        if task == "classification" and "accuracy" in metrics:
            acc = metrics["accuracy"]
            base_acc = additional_info.get("baseline_accuracy") if additional_info else None
            acc_improve = None
            if base_acc is not None and base_acc > 0:
                acc_improve = acc - base_acc
                metrics.setdefault("baseline_accuracy", base_acc)
                metrics.setdefault("accuracy_improvement", acc_improve)
            category = "Excellent" if acc > 0.9 else "Good" if acc > 0.75 else "Fair" if acc > 0.6 else "Poor"
            performance_summary = f"Accuracy: {acc:.1%} ({category}); " + (
                f"Baseline: {base_acc:.1%}; Δ: {acc_improve:.1%}" if acc_improve is not None else "Baseline: n/a"
            )
            metrics.setdefault("performance_category", category)
        elif task == "regression" and "r2" in metrics:
            r2 = metrics["r2"]
            category = (
                "Excellent" if r2 > 0.9 else "Good" if r2 > 0.75 else "Fair" if r2 > 0.5 else "Poor"
            )
            performance_summary = f"R²: {r2:.3f} ({category})"
            metrics.setdefault("performance_category", category)
        elif task == "regression" and "mse" in metrics:
            mse = metrics["mse"]
            category = "Low error" if mse < 1.0 else "Moderate error" if mse < 10.0 else "High error"
            performance_summary = f"MSE: {mse:.4f} ({category})"
            metrics.setdefault("performance_category", category)
        elif "silhouette" in metrics:
            sil = metrics["silhouette"]
            category = (
                "Excellent" if sil > 0.7 else "Good" if sil > 0.5 else "Fair" if sil > 0.25 else "Poor"
            )
            performance_summary = f"Silhouette: {sil:.3f} ({category})"
            metrics.setdefault("performance_category", category)
        else:
            performance_summary = "Algorithm completed successfully"
    else:
        performance_summary = "Algorithm execution failed"

    # Generate recommendations
    recommendations = []
    if success:
        if task == "classification" and "accuracy" in metrics:
            acc = metrics["accuracy"]
            if acc < 0.7:
                recommendations.extend([
                    "Consider feature engineering or selection",
                    "Try different hyperparameters",
                    "Collect more training data"
                ])
            elif acc > 0.95:
                recommendations.append("Check for overfitting - consider cross-validation")
        elif task == "regression" and "r2" in metrics:
            r2 = metrics["r2"]
            if r2 < 0.5:
                recommendations.extend([
                    "Model may be underfitting - try more complex models",
                    "Feature engineering might improve performance",
                    "Check for outliers in the data"
                ])
        elif "silhouette" in metrics:
            sil = metrics["silhouette"]
            if sil < 0.25:
                recommendations.extend([
                    "Try different number of clusters",
                    "Consider different distance metrics",
                    "Data preprocessing might help"
                ])

        # Timing-based recommendations
        if execution_time > 1.0:
            recommendations.append("Consider optimization for faster execution")

        # General recommendations
        recommendations.append("Validate results with cross-validation")
        if task in ["classification", "regression"]:
            recommendations.append("Try ensemble methods for better performance")

    # Basic data characteristics extraction (lightweight to avoid heavy recomputation)
    data_characteristics: dict[str, Any] = {}
    try:  # Guard in case of missing entries
        if isinstance(data_info.get("n_samples"), int) and isinstance(data_info.get("n_features"), int):
            ns = int(data_info["n_samples"]) or 0
            nf = int(data_info["n_features"]) or 0
            density = None
            if ns > 0 and nf > 0:
                density = f"samples_per_feature={ns / max(nf, 1):.2f}"  # crude ratio
            data_characteristics.update({
                "n_samples": ns,
                "n_features": nf,
                "samples_per_feature_ratio": density,
            })
        if "n_classes" in data_info:
            data_characteristics["n_classes"] = data_info["n_classes"]
        if "cluster_counts" in data_info:
            counts = data_info["cluster_counts"]
            if isinstance(counts, dict) and counts:
                data_characteristics["cluster_balance"] = {
                    "min_cluster_size": min(counts.values()),
                    "max_cluster_size": max(counts.values()),
                }
    except Exception:  # pragma: no cover
        pass

    # Model diagnostics extraction
    model_diagnostics: dict[str, Any] = {}
    if task == "regression" and {"mse", "r2"}.intersection(metrics):
        mse = metrics.get("mse")
        r2 = metrics.get("r2")
        if mse is not None:
            model_diagnostics["error_scale"] = (
                "low" if mse < 1 else "moderate" if mse < 10 else "high"
            )
        if r2 is not None:
            model_diagnostics["fit_quality"] = (
                "excellent" if r2 > 0.9 else "good" if r2 > 0.7 else "fair" if r2 > 0.5 else "poor"
            )
    if task == "classification" and "accuracy" in metrics:
        acc = metrics["accuracy"]
        model_diagnostics["confidence_assessment"] = (
            "strong" if acc > 0.85 else "moderate" if acc > 0.7 else "weak"
        )
    if "silhouette" in metrics:
        sil = metrics["silhouette"]
        model_diagnostics["cluster_separation"] = (
            "excellent" if sil > 0.7 else "good" if sil > 0.5 else "fair" if sil > 0.25 else "poor"
        )

    # Narrative reasoning synthesis
    reasoning_parts: list[str] = []
    reasoning_parts.append(f"Executed {algorithm} for {task} task in {execution_time:.3f}s.")
    if task == "classification" and "accuracy" in metrics:
        reasoning_parts.append(
            f"Observed accuracy={metrics['accuracy']:.3f}, categorized as {model_diagnostics.get('confidence_assessment', 'n/a')} confidence."  # noqa: E501
        )
        if data_characteristics.get("n_classes"):
            reasoning_parts.append(f"Problem involves {data_characteristics['n_classes']} classes.")
    if task == "regression" and ("r2" in metrics or "mse" in metrics):
        if "r2" in metrics:
            reasoning_parts.append(
                f"R²={metrics['r2']:.3f} indicating {model_diagnostics.get('fit_quality', 'unknown')} fit quality."  # noqa: E501
            )
        if "mse" in metrics:
            reasoning_parts.append(
                f"MSE={metrics['mse']:.4f} suggesting {model_diagnostics.get('error_scale', 'unknown')} error magnitude."  # noqa: E501
            )
    if "silhouette" in metrics:
        reasoning_parts.append(
            f"Silhouette={metrics['silhouette']:.3f} implies {model_diagnostics.get('cluster_separation', 'unknown')} cluster separation."  # noqa: E501
        )
    if recommendations:
        reasoning_parts.append("Key next steps: " + "; ".join(recommendations[:3]))
    reasoning = " " .join(reasoning_parts)
    # Derive structured warning levels
    warning_levels: list[tuple[str, str]] = []
    for w in warnings:
        wl = w.lower()
        level = "info"
        if any(k in wl for k in ["error", "fail", "missing"]):
            level = "critical"
        elif any(k in wl for k in ["warn", "slow", "overfit", "high"]):
            level = "warn"
        warning_levels.append((level, w))

    # Append runtime & summary to details for richer display
    if success:
        details = details.rstrip() + f"\nRuntime: {execution_time:.3f}s\nSummary: {performance_summary}"
    return RunResult(
        algorithm=algorithm,
        task=task,
        success=success,
        details=details,
        metrics=metrics,
        warnings=warnings,
        warning_levels=warning_levels,
        execution_time=execution_time,
        model_info=model_info,
        performance_summary=performance_summary,
        recommendations=recommendations,
        data_characteristics=data_characteristics,
        model_diagnostics=model_diagnostics,
        reasoning=reasoning,
    )


def _linear_regression_metrics(X: np.ndarray, y: np.ndarray, start_time: float, spec: SyntheticDataSpec) -> RunResult:
    coef, *_ = np.linalg.lstsq(X, y, rcond=None)
    preds = X @ coef
    mse = float(np.mean((preds - y) ** 2))
    ss_res = float(np.sum((y - preds) ** 2))
    ss_tot = float(np.sum((y - np.mean(y)) ** 2))
    r2 = 1.0 - (ss_res / ss_tot if ss_tot != 0 else 0.0)
    rmse = float(np.sqrt(mse))
    mae = float(np.mean(np.abs(preds - y)))
    resid_std = float(np.std(preds - y))
    details = (
        "✅ Linear Regression Results:\n"
        f"MSE: {mse:.4f} | RMSE: {rmse:.4f} | MAE: {mae:.4f} | ResidualStd: {resid_std:.4f} | R²: {r2:.4f}\n"
        f"Samples: {X.shape[0]} | Features: {X.shape[1]}\n"
        "Interpretation: MSE is mean squared error; RMSE gives typical error magnitude in original units; MAE is average absolute deviation; R² shows variance explained (1=perfect, 0=baseline)."
    )

    model_params = {
        "coefficients": coef.tolist() if hasattr(coef, 'tolist') else str(coef),
        "solver": "least_squares",
    }

    additional_info = {
        "n_samples": spec.n_samples,
        "n_features": spec.n_features,
        "feature_names": [f"feature_{i}" for i in range(X.shape[1])],
    "target_mean": float(np.mean(y)),
    "target_std": float(np.std(y)),
    }

    return _create_enhanced_result(
        algorithm="Linear Regression",
        task="regression",
        success=True,
        details=details,
        metrics={"mse": mse, "r2": r2, "rmse": rmse, "mae": mae, "residual_std": resid_std},
        warnings=[],
        start_time=start_time,
        model_params=model_params,
        additional_info=additional_info,
    )


def run_algorithm(algorithm: str, task: str, spec: SyntheticDataSpec | None = None) -> RunResult:
    if spec is None:
        spec = SyntheticDataSpec(task=task)

    start_time = time.time()
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
                    f"Counts: {counts}\nSamples: {spec.n_samples} | Features: {spec.n_features}\n"
                    "Interpretation: Inertia is within-cluster sum of squares (lower=denser clusters); Silhouette measures separation (-1 to 1). Review cluster size balance for stability."
                )

                model_params = {
                    "n_clusters": k,
                    "random_state": 42,
                    "n_init": 10,
                    "algorithm": "lloyd",
                }

                additional_info = {
                    "n_samples": spec.n_samples,
                    "n_features": spec.n_features,
                    "scaled_data": True,
                    "cluster_counts": counts,
                }

                return _create_enhanced_result(
                    algorithm=algorithm,
                    task=task,
                    success=True,
                    details=details,
                    metrics={"inertia": inertia, "silhouette": sil},
                    warnings=[],
                    start_time=start_time,
                    model_params=model_params,
                    additional_info=additional_info,
                )
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
                    f"Samples: {spec.n_samples} | Features: {spec.n_features}\n"
                    "Interpretation: DBSCAN discovers dense regions; noise points labeled -1. Adjust eps/min_samples to merge or split clusters; Silhouette contextualizes cohesion/separation."
                )

                model_params = {
                    "eps": 0.9,
                    "min_samples": 5,
                    "algorithm": "auto",
                    "leaf_size": 30,
                }

                additional_info = {
                    "n_samples": spec.n_samples,
                    "n_features": spec.n_features,
                    "scaled_data": True,
                    "n_clusters_found": n_clusters,
                    "noise_points": noise,
                }

                return _create_enhanced_result(
                    algorithm=algorithm,
                    task=task,
                    success=True,
                    details=details,
                    metrics={"clusters": n_clusters, "noise": noise, "silhouette": sil},
                    warnings=[],
                    start_time=start_time,
                    model_params=model_params,
                    additional_info=additional_info,
                )
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
                    f"Samples: {spec.n_samples} | Original Features: {spec.n_features}\n"
                    "Interpretation: Variance ratios show proportion of original variability captured per component; cumulative indicates retained information after dimensionality reduction."
                )

                model_params = {
                    "n_components": n_components,
                    "random_state": 42,
                    "svd_solver": "auto",
                }

                additional_info = {
                    "n_samples": spec.n_samples,
                    "n_features": spec.n_features,
                    "explained_variance_ratio": explained.tolist(),
                    "cumulative_variance": cumulative,
                    "scaled_data": True,
                }

                return _create_enhanced_result(
                    algorithm=algorithm,
                    task=task,
                    success=True,
                    details=details,
                    metrics={"cumulative_variance": cumulative},
                    warnings=[],
                    start_time=start_time,
                    model_params=model_params,
                    additional_info=additional_info,
                )
            # Default case: Hierarchical Clustering
            k = 3
            model = AgglomerativeClustering(n_clusters=k)
            labels = model.fit_predict(X_scaled)
            sil = float(silhouette_score(X_scaled, labels)) if k > 1 else -1.0
            counts = {c: int(np.sum(labels == c)) for c in sorted(set(labels))}
            details = (
                "✅ Hierarchical Clustering Results:\n"
                f"Clusters: {k} | Silhouette: {sil:.3f} | Counts: {counts}\n"
                f"Samples: {spec.n_samples} | Features: {spec.n_features}\n"
                "Interpretation: Agglomerative clustering builds a merge tree; Silhouette reflects separation; consider different linkage or distance metrics for alternative structures."
            )

            model_params = {
                "n_clusters": k,
                "linkage": "ward",
                "affinity": "euclidean",
            }

            additional_info = {
                "n_samples": spec.n_samples,
                "n_features": spec.n_features,
                "scaled_data": True,
                "cluster_counts": counts,
            }

            return _create_enhanced_result(
                algorithm="Hierarchical Clustering",
                task=task,
                success=True,
                details=details,
                metrics={"silhouette": sil},
                warnings=[],
                start_time=start_time,
                model_params=model_params,
                additional_info=additional_info,
            )
        if algorithm == "Linear Regression":
            assert y is not None
            if task == "regression":
                return _linear_regression_metrics(X, y, start_time, spec)
            if task == "classification":
                # Treat 'Linear Regression' in classification context as a Logistic Regression model
                try:
                    from sklearn.linear_model import LogisticRegression  # type: ignore
                    from sklearn.metrics import log_loss as sk_log_loss
                    from sklearn.metrics import roc_auc_score as sk_roc_auc
                    scaler = StandardScaler()
                    X_scaled = scaler.fit_transform(X)
                    model = LogisticRegression(max_iter=300, random_state=42)
                    model.fit(X_scaled, y)
                    preds = model.predict(X_scaled)
                    acc = float(np.mean(preds == y))
                    proba = None
                    logloss = None
                    roc_auc = None
                    try:
                        proba = model.predict_proba(X_scaled)
                        if proba is not None:
                            logloss = float(sk_log_loss(y, proba))
                            if proba.shape[1] == 2:  # binary
                                roc_auc = float(sk_roc_auc(y, proba[:, 1]))
                    except Exception:  # pragma: no cover
                        pass
                    details = (
                        "✅ Linear (Logistic) Regression Classification Results:\n"
                        f"Accuracy: {acc:.3f} | Classes: {len(np.unique(y))}\n"
                        f"Samples: {spec.n_samples} | Features: {spec.n_features} (scaled)\n"
                        "Interpretation: Accuracy measures overall correctness; Log Loss penalizes confident wrong predictions; ROC AUC (binary) reflects ranking quality (0.5=random, 1.0=perfect)."
                    )
                    model_params = {
                        "penalty": getattr(model, "penalty", None),
                        "C": getattr(model, "C", None),
                        "solver": getattr(model, "solver", None),
                        "max_iter": getattr(model, "max_iter", None),
                        "multi_class": getattr(model, "multi_class", None),
                    }
                    class_counts = {int(c): int(np.sum(y == c)) for c in np.unique(y)}
                    baseline_acc = max(class_counts.values()) / spec.n_samples
                    additional_info = {
                        "n_samples": spec.n_samples,
                        "n_features": spec.n_features,
                        "n_classes": len(np.unique(y)),
                        "scaled_data": True,
                        "model_kind": "logistic_regression_adapter",
                        "class_distribution": class_counts,
                        "baseline_accuracy": baseline_acc,
                    }
                    metrics_extra = {"accuracy": acc, "baseline_accuracy": baseline_acc}
                    if logloss is not None:
                        metrics_extra["log_loss"] = logloss
                    if roc_auc is not None:
                        metrics_extra["roc_auc"] = roc_auc
                    return _create_enhanced_result(
                        algorithm=algorithm,
                        task=task,
                        success=True,
                        details=details,
                        metrics=metrics_extra,
                        warnings=[],
                        start_time=start_time,
                        model_params=model_params,
                        additional_info=additional_info,
                    )
                except Exception as exc:  # pragma: no cover
                    return RunResult(
                        algorithm, task, False,
                        f"❌ Failed classification adaptation for Linear Regression: {exc}", {}, []
                    )
        if algorithm == "Logistic Regression":
            assert y is not None
            if task == "classification":
                # Standard logistic regression path (reuse logic akin to linear->classification above)
                try:
                    from sklearn.linear_model import LogisticRegression  # type: ignore
                    from sklearn.metrics import log_loss as sk_log_loss
                    from sklearn.metrics import roc_auc_score as sk_roc_auc
                    scaler = StandardScaler()
                    X_scaled = scaler.fit_transform(X)
                    model = LogisticRegression(max_iter=300, random_state=42)
                    model.fit(X_scaled, y)
                    preds = model.predict(X_scaled)
                    acc = float(np.mean(preds == y))
                    proba = None
                    logloss = None
                    roc_auc = None
                    try:
                        proba = model.predict_proba(X_scaled)
                        if proba is not None:
                            logloss = float(sk_log_loss(y, proba))
                            if proba.shape[1] == 2:
                                roc_auc = float(sk_roc_auc(y, proba[:, 1]))
                    except Exception:  # pragma: no cover
                        pass
                    details = (
                        "✅ Logistic Regression Classification Results:\n"
                        f"Accuracy: {acc:.3f} | Classes: {len(np.unique(y))}\n"
                        f"Samples: {spec.n_samples} | Features: {spec.n_features} (scaled)\n"
                        "Interpretation: Compare accuracy to baseline (majority class). Lower Log Loss suggests better probability calibration; higher ROC AUC indicates superior class separability."
                    )
                    model_params = {
                        "penalty": getattr(model, "penalty", None),
                        "C": getattr(model, "C", None),
                        "solver": getattr(model, "solver", None),
                        "max_iter": getattr(model, "max_iter", None),
                        "multi_class": getattr(model, "multi_class", None),
                    }
                    class_counts = {int(c): int(np.sum(y == c)) for c in np.unique(y)}
                    baseline_acc = max(class_counts.values()) / spec.n_samples
                    additional_info = {
                        "n_samples": spec.n_samples,
                        "n_features": spec.n_features,
                        "n_classes": len(np.unique(y)),
                        "scaled_data": True,
                        "model_kind": "logistic_regression",
                        "class_distribution": class_counts,
                        "baseline_accuracy": baseline_acc,
                    }
                    metrics_extra = {"accuracy": acc, "baseline_accuracy": baseline_acc}
                    if logloss is not None:
                        metrics_extra["log_loss"] = logloss
                    if roc_auc is not None:
                        metrics_extra["roc_auc"] = roc_auc
                    return _create_enhanced_result(
                        algorithm=algorithm,
                        task=task,
                        success=True,
                        details=details,
                        metrics=metrics_extra,
                        warnings=[],
                        start_time=start_time,
                        model_params=model_params,
                        additional_info=additional_info,
                    )
                except Exception as exc:  # pragma: no cover
                    return RunResult(
                        algorithm, task, False,
                        f"❌ Failed Logistic Regression classification run: {exc}", {}, []
                    )
            if task == "regression":
                # Provide a pseudo regression mode: treat as linear regression (using least squares)
                try:
                    coef, *_ = np.linalg.lstsq(X, y, rcond=None)
                    preds = X @ coef
                    mse = float(np.mean((preds - y) ** 2))
                    ss_res = float(np.sum((y - preds) ** 2))
                    ss_tot = float(np.sum((y - np.mean(y)) ** 2))
                    r2 = 1.0 - (ss_res / ss_tot if ss_tot != 0 else 0.0)
                    details = (
                        "✅ Logistic Regression (Linear Mode) Regression Results:\n"
                        f"MSE: {mse:.4f} | R²: {r2:.4f}\n"
                        f"Samples: {spec.n_samples} | Features: {spec.n_features}\n"
                        "Interpretation: Using linear least squares surrogate; R² indicates variance explained; investigate residual spread (std) for noise level."
                    )
                    model_params = {
                        "coefficients": coef.tolist() if hasattr(coef, 'tolist') else str(coef),
                        "solver": "least_squares_adapter",
                    }
                    additional_info = {
                        "n_samples": spec.n_samples,
                        "n_features": spec.n_features,
                        "feature_names": [f"feature_{i}" for i in range(X.shape[1])],
                        "model_kind": "logistic_regression_linear_adapter",
                        "target_mean": float(np.mean(y)),
                        "target_std": float(np.std(y)),
                    }
                    # Extend with additional residual metrics
                    rmse = float(np.sqrt(mse))
                    mae = float(np.mean(np.abs(preds - y)))
                    resid_std = float(np.std(preds - y))
                    return _create_enhanced_result(
                        algorithm=algorithm,
                        task=task,
                        success=True,
                        details=details,
                        metrics={"mse": mse, "r2": r2, "rmse": rmse, "mae": mae, "residual_std": resid_std},
                        warnings=[],
                        start_time=start_time,
                        model_params=model_params,
                        additional_info=additional_info,
                    )
                except Exception as exc:  # pragma: no cover
                    return RunResult(
                        algorithm, task, False,
                        f"❌ Failed Logistic Regression regression adaptation: {exc}", {}, []
                    )
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
                    f"Samples: {spec.n_samples} | Features: {spec.n_features} (scaled)\n"
                    "Interpretation: Accuracy shows proportion correct; RBF kernel projects data to higher-dimensional space; enable probability calibration for probabilistic outputs if needed."
                )

                model_params = {
                    "kernel": "rbf",
                    "gamma": "scale",
                    "C": 1.0,
                    "random_state": 42,
                }

                class_counts = {int(c): int(np.sum(y == c)) for c in np.unique(y)}
                baseline_acc = max(class_counts.values()) / spec.n_samples
                additional_info = {
                    "n_samples": spec.n_samples,
                    "n_features": spec.n_features,
                    "n_classes": len(np.unique(y)),
                    "scaled_data": True,
                    "class_distribution": class_counts,
                    "baseline_accuracy": baseline_acc,
                }

                return _create_enhanced_result(
                    algorithm=algorithm,
                    task=task,
                    success=True,
                    details=details,
                    metrics={"accuracy": acc, "baseline_accuracy": baseline_acc},
                    warnings=[],
                    start_time=start_time,
                    model_params=model_params,
                    additional_info=additional_info,
                )
            # Default case: SVM Regression
            assert y is not None
            model = SVR(kernel="rbf", C=1.0)
            model.fit(X_scaled, y)
            preds = model.predict(X_scaled)
            mse = float(np.mean((preds - y) ** 2))
            mae = float(np.mean(np.abs(preds - y)))
            rmse = float(np.sqrt(mse))
            resid_std = float(np.std(preds - y))
            details = (
                "✅ Support Vector Machine Regression Results:\n"
                f"MSE: {mse:.4f} | RMSE: {rmse:.4f} | MAE: {mae:.4f} | ResidualStd: {resid_std:.4f}\n"
                f"Samples: {spec.n_samples} | Features: {spec.n_features} (scaled)\n"
                "Interpretation: MSE/MAE capture squared vs absolute error; tuning C, epsilon and kernel parameters can adjust bias-variance and margin sensitivity."
            )

            model_params = {
                "kernel": "rbf",
                "C": 1.0,
                "gamma": "scale",
                "epsilon": 0.1,
            }

            additional_info = {
                "n_samples": spec.n_samples,
                "n_features": spec.n_features,
                "scaled_data": True,
                "target_mean": float(np.mean(y)),
                "target_std": float(np.std(y)),
            }

            return _create_enhanced_result(
                algorithm=algorithm,
                task=task,
                success=True,
                details=details,
                metrics={"mse": mse, "mae": mae, "rmse": rmse, "residual_std": resid_std},
                warnings=[],
                start_time=start_time,
                model_params=model_params,
                additional_info=additional_info,
            )
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
                    f"Samples: {spec.n_samples} | Features: {spec.n_features}\n"
                    "Interpretation: Gradient boosting reduces bias iteratively; monitor log loss for calibration and adjust learning_rate / n_estimators to manage bias-variance tradeoff."
                )

                model_params = {
                    "n_estimators": 100,
                    "max_depth": 3,
                    "learning_rate": 0.1,
                    "subsample": 0.9,
                    "colsample_bytree": 0.9,
                    "reg_lambda": 1,
                }

                class_counts = {int(c): int(np.sum(y == c)) for c in np.unique(y)}
                baseline_acc = max(class_counts.values()) / spec.n_samples
                additional_info = {
                    "n_samples": spec.n_samples,
                    "n_features": spec.n_features,
                    "n_classes": len(np.unique(y)),
                    "class_distribution": class_counts,
                    "baseline_accuracy": baseline_acc,
                }

                return _create_enhanced_result(
                    algorithm=algorithm,
                    task=task,
                    success=True,
                    details=details,
                    metrics={"accuracy": acc, "baseline_accuracy": baseline_acc},
                    warnings=[],
                    start_time=start_time,
                    model_params=model_params,
                    additional_info=additional_info,
                )

            # Default case: XGBoost Regression
            mse = float(np.mean((preds - y) ** 2))
            mae = float(np.mean(np.abs(preds - y)))
            rmse = float(np.sqrt(mse))
            resid_std = float(np.std(preds - y))
            details = (
                "✅ XGBoost Regression Results:\n"
                f"MSE: {mse:.4f} | RMSE: {rmse:.4f} | MAE: {mae:.4f} | ResidualStd: {resid_std:.4f}\n"
                f"Samples: {spec.n_samples} | Features: {spec.n_features}\n"
                "Interpretation: Boosting sequentially fits residuals; watch for overfitting (training error << validation). Use subsample/colsample_bytree to reduce variance."
            )

            model_params = {
                "n_estimators": 120,
                "max_depth": 3,
                "learning_rate": 0.1,
                "subsample": 0.9,
                "colsample_bytree": 0.9,
                "reg_lambda": 1,
            }

            additional_info = {
                "n_samples": spec.n_samples,
                "n_features": spec.n_features,
                "target_mean": float(np.mean(y)),
                "target_std": float(np.std(y)),
            }

            return _create_enhanced_result(
                algorithm=algorithm,
                task=task,
                success=True,
                details=details,
                metrics={"mse": mse, "mae": mae, "rmse": rmse, "residual_std": resid_std},
                warnings=[],
                start_time=start_time,
                model_params=model_params,
                additional_info=additional_info,
            )
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
                    f"Samples: {spec.n_samples} | Features: {spec.n_features}\n"
                    "Interpretation: MLP optimizes cross-entropy; monitor for overfitting (training >> validation). Adjust hidden sizes, regularization, or learning rate as needed."
                )

                model_params = {
                    "hidden_layer_sizes": (32, 16),
                    "activation": "relu",
                    "max_iter": 300,
                    "random_state": 42,
                }

                class_counts = {int(c): int(np.sum(y == c)) for c in np.unique(y)}
                baseline_acc = max(class_counts.values()) / spec.n_samples
                additional_info = {
                    "n_samples": spec.n_samples,
                    "n_features": spec.n_features,
                    "n_classes": len(np.unique(y)),
                    "n_hidden_layers": 2,
                    "class_distribution": class_counts,
                    "baseline_accuracy": baseline_acc,
                }

                return _create_enhanced_result(
                    algorithm=algorithm,
                    task=task,
                    success=True,
                    details=details,
                    metrics={"accuracy": acc, "baseline_accuracy": baseline_acc},
                    warnings=[],
                    start_time=start_time,
                    model_params=model_params,
                    additional_info=additional_info,
                )

            # Default case: Neural Network Regression
            model = MLPRegressor(hidden_layer_sizes=(64, 32), activation="relu", max_iter=300, random_state=42)
            model.fit(X, y)
            preds = model.predict(X)
            mse = float(np.mean((preds - y) ** 2))
            mae = float(np.mean(np.abs(preds - y)))
            rmse = float(np.sqrt(mse))
            resid_std = float(np.std(preds - y))
            details = (
                "✅ Neural Network Regression Results:\n"
                f"MSE: {mse:.4f} | RMSE: {rmse:.4f} | MAE: {mae:.4f} | ResidualStd: {resid_std:.4f}\n"
                f"Samples: {spec.n_samples} | Features: {spec.n_features}\n"
                "Interpretation: Neural network approximates nonlinear function; examine learning curves and consider regularization or architecture changes for bias/variance issues."
            )

            model_params = {
                "hidden_layer_sizes": (64, 32),
                "activation": "relu",
                "max_iter": 300,
                "random_state": 42,
            }

            additional_info = {
                "n_samples": spec.n_samples,
                "n_features": spec.n_features,
                "n_hidden_layers": 2,
                "target_mean": float(np.mean(y)),
                "target_std": float(np.std(y)),
            }

            return _create_enhanced_result(
                algorithm=algorithm,
                task=task,
                success=True,
                details=details,
                metrics={"mse": mse, "mae": mae, "rmse": rmse, "residual_std": resid_std},
                warnings=[],
                start_time=start_time,
                model_params=model_params,
                additional_info=additional_info,
            )
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
                    f"Samples: {spec.n_samples} | Features: {spec.n_features}\n"
                    "Interpretation: Tree-based models partition feature space; review feature importances for dominant predictors; consider pruning or more estimators for stability/generalization."
                )
                # Feature importances if available
                try:
                    importances = getattr(model, "feature_importances_", None)
                    if importances is not None:
                        top_imp = sorted(
                            enumerate(importances), key=lambda t: t[1], reverse=True
                        )[:5]
                        details += "\nTop Feature Importances: " + ", ".join(
                            f"f{idx}:{val:.3f}" for idx, val in top_imp
                        )
                except Exception:  # pragma: no cover
                    pass

                if algorithm == "Decision Trees":
                    model_params = {
                        "criterion": "gini",
                        "max_depth": None,
                        "min_samples_split": 2,
                        "min_samples_leaf": 1,
                    }
                else:  # Random Forest
                    model_params = {
                        "n_estimators": 50,
                        "criterion": "gini",
                        "max_depth": None,
                        "min_samples_split": 2,
                        "min_samples_leaf": 1,
                    }

                class_counts = {int(c): int(np.sum(y == c)) for c in np.unique(y)}
                baseline_acc = max(class_counts.values()) / spec.n_samples
                additional_info = {
                    "n_samples": spec.n_samples,
                    "n_features": spec.n_features,
                    "n_classes": len(np.unique(y)),
                    "class_distribution": class_counts,
                    "baseline_accuracy": baseline_acc,
                }

                return _create_enhanced_result(
                    algorithm=algorithm,
                    task=task,
                    success=True,
                    details=details,
                    metrics={"accuracy": acc, "baseline_accuracy": baseline_acc},
                    warnings=[],
                    start_time=start_time,
                    model_params=model_params,
                    additional_info=additional_info,
                )

            # Default case: Regression
            mse = float(np.mean((preds - y) ** 2))
            mae = float(np.mean(np.abs(preds - y)))
            rmse = float(np.sqrt(mse))
            residuals = preds - y
            resid_std = float(np.std(residuals))
            details = (
                f"✅ {algorithm} Regression Results:\n"
                f"MSE: {mse:.4f} | RMSE: {rmse:.4f} | MAE: {mae:.4f} | ResidualStd: {resid_std:.4f}\n"
                f"Samples: {spec.n_samples} | Features: {spec.n_features}\n"
                "Interpretation: ResidualStd approximates noise level; feature importances highlight influential variables; tune tree depth / number of estimators for bias-variance tradeoffs."
            )
            # Feature importances if available
            try:
                importances = getattr(model, "feature_importances_", None)
                if importances is not None:
                    top_imp = sorted(
                        enumerate(importances), key=lambda t: t[1], reverse=True
                    )[:5]
                    details += "\nTop Feature Importances: " + ", ".join(
                        f"f{idx}:{val:.3f}" for idx, val in top_imp
                    )
            except Exception:  # pragma: no cover
                pass

            if algorithm == "Decision Trees":
                model_params = {
                    "criterion": "squared_error",
                    "max_depth": None,
                    "min_samples_split": 2,
                    "min_samples_leaf": 1,
                }
            else:  # Random Forest
                model_params = {
                    "n_estimators": 50,
                    "criterion": "squared_error",
                    "max_depth": None,
                    "min_samples_split": 2,
                    "min_samples_leaf": 1,
                }

            additional_info = {
                "n_samples": spec.n_samples,
                "n_features": spec.n_features,
                "target_mean": float(np.mean(y)),
                "target_std": float(np.std(y)),
            }

            return _create_enhanced_result(
                algorithm=algorithm,
                task=task,
                success=True,
                details=details,
                metrics={"mse": mse, "mae": mae, "rmse": rmse, "residual_std": resid_std},
                warnings=[],
                start_time=start_time,
                model_params=model_params,
                additional_info=additional_info,
            )
        return RunResult(algorithm, task, False, f"❌ {algorithm} not implemented for {task} task", {}, [])
    except Exception as exc:  # pragma: no cover
        return RunResult(algorithm, task, False, f"❌ Error running {algorithm}: {exc}", {}, [repr(exc)])


__all__ = ["run_algorithm", "RunResult"]
