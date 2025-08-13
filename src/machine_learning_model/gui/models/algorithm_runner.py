"""Algorithm execution utilities (Step 3 of MVC refactor).

Encapsulates the logic previously embedded in QuickRunPanel.run_algorithm.
"""
from __future__ import annotations

import time
from dataclasses import asdict, dataclass, field
from typing import Any
from typing import Any as _Any

import numpy as np

from .data_synthesizer import SyntheticDataSpec, generate_synthetic_data

# Custom implementations (fallback to simple stubs if module missing)
DecisionTreeClassifier: _Any
DecisionTreeRegressor: _Any
RandomForestClassifier: _Any
RandomForestRegressor: _Any
try:  # Custom implementations
    from machine_learning_model.supervised.decision_tree import (
        DecisionTreeClassifier as _DTClf,
    )
    from machine_learning_model.supervised.decision_tree import (
        DecisionTreeRegressor as _DTReg,
    )
    from machine_learning_model.supervised.random_forest import (
        RandomForestClassifier as _RFClf,
    )
    from machine_learning_model.supervised.random_forest import (
        RandomForestRegressor as _RFReg,
    )
    DecisionTreeClassifier = _DTClf
    DecisionTreeRegressor = _DTReg
    RandomForestClassifier = _RFClf
    RandomForestRegressor = _RFReg
except ImportError:  # pragma: no cover
    class _DTClfStub:
        def fit(self, X, y) -> None:
            return None

        def predict(self, X):
            return np.zeros(len(X))

    class _DTRegStub:
        def fit(self, X, y) -> None:
            return None

        def predict(self, X):
            return np.zeros(len(X))

    class _RFClfStub:
        def __init__(self, n_estimators: int = 25) -> None:
            self.n_estimators = n_estimators

        def fit(self, X, y) -> None:
            return None

        def predict(self, X):
            return np.zeros(len(X))

    class _RFRegStub:
        def __init__(self, n_estimators: int = 25) -> None:
            self.n_estimators = n_estimators

        def fit(self, X, y) -> None:
            return None

        def predict(self, X):
            return np.zeros(len(X))

    DecisionTreeClassifier = _DTClfStub  # type: ignore[assignment]
    DecisionTreeRegressor = _DTRegStub  # type: ignore[assignment]
    RandomForestClassifier = _RFClfStub  # type: ignore[assignment]
    RandomForestRegressor = _RFRegStub  # type: ignore[assignment]

# Note: All optional third-party imports (sklearn, xgboost) are imported lazily within
# the branches that need them. This keeps static analyzers quiet when deps are absent.


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
    except (KeyError, TypeError, ValueError):  # pragma: no cover
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
        # ---------------------- Semi-Supervised Algorithms -----------------------------
        if algorithm in {"Label Propagation", "Self-Training", "Co-Training", "Semi-Supervised SVM"}:
            if task != "classification":
                return RunResult(algorithm, task, False, f"❌ {algorithm} currently supports classification only", {}, [])
            assert y is not None

            rng = np.random.default_rng(42)
            n_samples = X.shape[0]
            labeled_fraction = 0.2 if n_samples > 10 else 0.5
            n_initial = max(2, int(labeled_fraction * n_samples))
            initial_indices = rng.choice(n_samples, size=n_initial, replace=False)
            unlabeled_indices = np.setdiff1d(np.arange(n_samples), initial_indices)
            y_masked = np.full_like(y, fill_value=-1)
            y_masked[initial_indices] = y[initial_indices]

            def _finish(preds: np.ndarray, pseudo_added: int, iterations: int, desc: str) -> RunResult:
                acc = float(np.mean(preds == y))
                acc_labeled = float(np.mean(preds[initial_indices] == y[initial_indices])) if len(initial_indices) else 0.0
                acc_unlabeled = float(np.mean(preds[unlabeled_indices] == y[unlabeled_indices])) if len(unlabeled_indices) else 0.0
                majority = int(np.bincount(y[initial_indices]).argmax()) if len(initial_indices) else int(np.bincount(y).argmax())
                baseline_acc = float(np.mean(y == majority))
                details = (
                    f"✅ {algorithm} Semi-Supervised Results:\n"
                    f"Overall Acc: {acc:.3f} | Labeled Acc: {acc_labeled:.3f} | Unlabeled Acc: {acc_unlabeled:.3f}\n"
                    f"Initial labeled: {len(initial_indices)} | Unlabeled: {len(unlabeled_indices)} | Pseudo-labels: {pseudo_added} | Iterations: {iterations}\n"
                    + desc
                )
                model_params = {"initial_labeled_fraction": labeled_fraction, "iterations": iterations}
                add_info = {
                    "n_samples": n_samples,
                    "n_features": X.shape[1],
                    "n_initial_labeled": len(initial_indices),
                    "n_unlabeled": len(unlabeled_indices),
                    "pseudo_labels_added": pseudo_added,
                    "baseline_accuracy": baseline_acc,
                }
                return _create_enhanced_result(
                    algorithm=algorithm,
                    task=task,
                    success=True,
                    details=details,
                    metrics={
                        "accuracy": acc,
                        "accuracy_labeled": acc_labeled,
                        "accuracy_unlabeled": acc_unlabeled,
                        "baseline_accuracy": baseline_acc,
                        "pseudo_labels": pseudo_added,
                    },
                    warnings=[],
                    start_time=start_time,
                    model_params=model_params,
                    additional_info=add_info,
                )

            if algorithm == "Label Propagation":
                try:
                    from sklearn.semi_supervised import LabelPropagation  # type: ignore
                except ImportError as e:
                    return RunResult(algorithm, task, False, "❌ scikit-learn not available for Label Propagation", {}, [repr(e)])
                model = LabelPropagation(kernel="rbf", gamma=20)
                model.fit(X, y_masked)
                preds = model.transduction_
                pseudo_added = int(np.sum(preds[unlabeled_indices] != -1))
                return _finish(preds, pseudo_added, 1, "Graph-based diffusion with RBF kernel.")

            if algorithm == "Self-Training":
                try:
                    from sklearn.linear_model import LogisticRegression  # type: ignore
                    from sklearn.semi_supervised import (
                        SelfTrainingClassifier,  # type: ignore
                    )
                except ImportError as e:
                    return RunResult(algorithm, task, False, "❌ scikit-learn not available for Self-Training", {}, [repr(e)])
                wrapper = SelfTrainingClassifier(LogisticRegression(max_iter=300), threshold=0.8, verbose=False)
                wrapper.fit(X, y_masked)
                preds = wrapper.predict(X)
                iterations = int(getattr(wrapper, "n_iter_", 1))
                pseudo_added = int(np.sum(preds[unlabeled_indices] != -1))
                return _finish(preds, pseudo_added, iterations, "Iterative self-labeling using LogisticRegression base.")

            if algorithm == "Co-Training":
                try:
                    from sklearn.linear_model import LogisticRegression  # type: ignore
                    from sklearn.neighbors import KNeighborsClassifier  # type: ignore
                except ImportError as e:
                    return RunResult(algorithm, task, False, "❌ scikit-learn not available for Co-Training", {}, [repr(e)])
                if X.shape[1] < 2:
                    return RunResult(algorithm, task, False, "❌ Co-Training requires >=2 features", {}, [])
                mid = X.shape[1] // 2
                view1, view2 = X[:, :mid], X[:, mid:]
                clf1 = LogisticRegression(max_iter=300)
                clf2 = KNeighborsClassifier(n_neighbors=5)
                labeled_set = set(initial_indices.tolist())
                y_work = y_masked.copy()
                added_total = 0
                max_iter = 5
                last_iter = 0
                for it in range(1, max_iter + 1):
                    mask = np.array([i in labeled_set for i in range(n_samples)])
                    clf1.fit(view1[mask], y_work[mask])
                    clf2.fit(view2[mask], y_work[mask])
                    unlabeled = [i for i in range(n_samples) if i not in labeled_set]
                    if not unlabeled:
                        break
                    probs1 = np.asarray(clf1.predict_proba(view1[unlabeled]))
                    probs2 = np.asarray(clf2.predict_proba(view2[unlabeled]))
                    conf1 = probs1.max(axis=1)
                    conf2 = probs2.max(axis=1)
                    k = min(5, len(unlabeled))
                    pick1 = np.argsort(-conf1)[:k]
                    pick2 = np.argsort(-conf2)[:k]
                    threshold = 0.9
                    newly_added = 0
                    for picks, probs in [(pick1, probs1), (pick2, probs2)]:
                        for local_i in picks:
                            global_i = unlabeled[local_i]
                            if global_i in labeled_set:
                                continue
                            if probs[local_i].max() >= threshold:
                                y_work[global_i] = int(np.argmax(probs[local_i]))
                                labeled_set.add(global_i)
                                newly_added += 1
                    added_total += newly_added
                    last_iter = it
                    if newly_added == 0:
                        break
                final_mask = np.array([i in labeled_set for i in range(n_samples)])
                from sklearn.linear_model import (
                    LogisticRegression as LRFinal,  # type: ignore
                )
                final_clf = LRFinal(max_iter=300)
                final_clf.fit(X[final_mask], y_work[final_mask])
                final_preds = final_clf.predict(X)
                return _finish(final_preds, added_total, last_iter, "Dual-view exchange between LogisticRegression and KNN (prototype).")

            if algorithm == "Semi-Supervised SVM":
                try:
                    from sklearn.svm import SVC  # type: ignore
                    y_work = y_masked.copy()
                    labeled_mask = y_work != -1
                    max_iter = 5
                    added_total = 0
                    last_iter = 0
                    for it in range(1, max_iter + 1):
                        clf = SVC(probability=True, kernel="rbf", gamma="scale")
                        clf.fit(X[labeled_mask], y_work[labeled_mask])
                        unlabeled_mask = ~labeled_mask
                        if not np.any(unlabeled_mask):
                            break
                        probs = clf.predict_proba(X[unlabeled_mask])
                        conf = probs.max(axis=1)
                        threshold = 0.95
                        high_conf = np.where(conf >= threshold)[0]
                        if len(high_conf) == 0:
                            break
                        unlabeled_idx_full = np.where(unlabeled_mask)[0]
                        for local_i in high_conf:
                            global_i = unlabeled_idx_full[local_i]
                            y_work[global_i] = int(np.argmax(probs[local_i]))
                            labeled_mask[global_i] = True
                        added_total += len(high_conf)
                        last_iter = it
                    final_clf = SVC(kernel="rbf", gamma="scale")
                    final_clf.fit(X[labeled_mask], y_work[labeled_mask])
                    final_preds = final_clf.predict(X)
                    return _finish(final_preds, added_total, last_iter, "Iterative high-confidence pseudo-labeling with SVC (TSVM placeholder).")
                except Exception as e:  # pragma: no cover
                    return RunResult(algorithm, task, False, f"❌ Error or missing dependency: {e}", {}, [repr(e)])

        # ---------------------- Unsupervised Algorithms -------------------------------
        if algorithm in {
            "K-Means Clustering",
            "DBSCAN",
            "Principal Component Analysis",
            "Hierarchical Clustering",
        }:
            try:
                from sklearn.cluster import (  # type: ignore
                    DBSCAN,
                    AgglomerativeClustering,
                    KMeans,
                )
                from sklearn.decomposition import PCA  # type: ignore
                from sklearn.metrics import silhouette_score  # type: ignore
                from sklearn.preprocessing import StandardScaler  # type: ignore
            except ImportError as e:
                return RunResult(
                    algorithm,
                    task,
                    False,
                    "❌ scikit-learn not available for Unsupervised algorithms",
                    {},
                    [repr(e)],
                )
            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(X)
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
                    from sklearn.metrics import log_loss as sk_log_loss  # type: ignore
                    from sklearn.metrics import (
                        roc_auc_score as sk_roc_auc,  # type: ignore
                    )
                    from sklearn.preprocessing import StandardScaler  # type: ignore
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
                    except (AttributeError, ValueError):  # pragma: no cover
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
                    from sklearn.metrics import log_loss as sk_log_loss  # type: ignore
                    from sklearn.metrics import (
                        roc_auc_score as sk_roc_auc,  # type: ignore
                    )
                    from sklearn.preprocessing import StandardScaler  # type: ignore
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
                    except (AttributeError, ValueError):  # pragma: no cover
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
            try:
                from sklearn.preprocessing import StandardScaler  # type: ignore
                from sklearn.svm import SVC, SVR  # type: ignore
            except ImportError as e:
                return RunResult(algorithm, task, False, "❌ scikit-learn not available for SVM", {}, [repr(e)])
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
            assert y is not None
            try:
                from xgboost import XGBClassifier, XGBRegressor  # type: ignore
            except ImportError as e:
                return RunResult(algorithm, task, False, "❌ XGBoost library not available", {}, [repr(e)])
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
            assert y is not None
            try:
                from sklearn.neural_network import (  # type: ignore
                    MLPClassifier,
                    MLPRegressor,
                )
            except ImportError as e:
                return RunResult(algorithm, task, False, "❌ scikit-learn not available for Neural Networks", {}, [repr(e)])
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
