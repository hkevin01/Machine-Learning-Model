"""MLflow tracking utilities.

Optional, safe-to-import helpers wrapping MLflow. If MLflow isn't installed
or disabled via MLFLOW_DISABLE=1, all functions become no-ops.
"""
from __future__ import annotations

import json
import os
import time
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Any, Mapping

_mlflow = None  # Lazy import cache (False = unavailable)


def _load_mlflow():
    global _mlflow
    if _mlflow is None:
        if os.getenv("MLFLOW_DISABLE") == "1":
            _mlflow = False
            return _mlflow
        try:  # pragma: no cover - best effort import
            import mlflow  # type: ignore

            _mlflow = mlflow
        except ImportError:  # mlflow not installed
            _mlflow = False
    return _mlflow


def tracking_enabled() -> bool:
    return bool(_load_mlflow())


@dataclass
class RunConfig:
    experiment_name: str = "default"
    run_name: str | None = None
    tags: Mapping[str, str] | None = None


@contextmanager
def start_run(config: RunConfig | None = None):
    mlflow = _load_mlflow()
    if not mlflow:
        yield None
        return
    exp_name = (config.experiment_name if config else None) or os.getenv(
        "MLFLOW_EXPERIMENT_NAME", "default"
    )
    mlflow.set_experiment(exp_name)
    with mlflow.start_run(run_name=config.run_name if config else None) as run:
        if config and config.tags:
            mlflow.set_tags(dict(config.tags))
        yield run


def log_params(params: Mapping[str, Any]):
    mlflow = _load_mlflow()
    if not mlflow:
        return
    safe = {k: (v if isinstance(v, (int, float, str, bool)) else str(v)) for k, v in params.items()}
    mlflow.log_params(safe)


def log_metrics(metrics: Mapping[str, float], step: int | None = None):
    mlflow = _load_mlflow()
    if not mlflow:
        return
    mlflow.log_metrics({k: float(v) for k, v in metrics.items()}, step=step)


def log_artifact(path: str):
    mlflow = _load_mlflow()
    if not mlflow:
        return
    if os.path.exists(path):
        mlflow.log_artifact(path)


def log_dict(data: dict[str, Any], artifact_file: str):
    mlflow = _load_mlflow()
    if not mlflow:
        return
    with open(artifact_file, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)
    log_artifact(artifact_file)


def log_feature_importances(importances, feature_names=None):
    mlflow = _load_mlflow()
    if not mlflow or importances is None:
        return
    payload = {
        "features": list(map(str, feature_names)) if feature_names is not None else list(range(len(importances))),
        "importances": [float(v) for v in importances],
        "timestamp": int(time.time()),
    }
    log_dict(payload, "feature_importances.json")


__all__ = [
    "RunConfig",
    "start_run",
    "log_params",
    "log_metrics",
    "log_artifact",
    "log_dict",
    "log_feature_importances",
    "tracking_enabled",
]
