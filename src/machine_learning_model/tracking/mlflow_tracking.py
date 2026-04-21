"""
Module: tracking.mlflow_tracking
Purpose: Optional MLflow integration layer for experiment tracking.
         All public functions are safe no-ops when MLflow is not installed
         or when the MLFLOW_DISABLE=1 environment variable is set. This
         allows algorithm code to call tracking functions unconditionally
         without coupling to MLflow's presence.
Rationale: Lazy-importing MLflow avoids import-time failures in environments
           where only the core ML stack is installed.
Assumptions: When MLflow is available, a tracking server reachable at
             MLFLOW_TRACKING_URI is expected. Local file-based tracking
             is used by default if no URI is set.
Failure Modes: ImportError → _mlflow set to False; all functions become
               no-ops. Connection errors during log_* calls are the
               caller's responsibility to handle.
"""
from __future__ import annotations

import json
import os
import time
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Any, Mapping

# Lazy-loaded MLflow module reference.
# None  = not yet attempted; False = unavailable; module = loaded.
_mlflow = None


def _load_mlflow():
    """
    Purpose:    Lazily import MLflow on first call and cache the result.
                Returns False if MLflow is unavailable or explicitly disabled.
    Returns:    mlflow module object, or False.
    Side-effect: Mutates module-level _mlflow cache.
    """
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
    """
    Purpose:  Report whether MLflow tracking is active for this session.
    Returns:  True only when the mlflow package is importable and not disabled.
    """
    return bool(_load_mlflow())


@dataclass
class RunConfig:
    """
    Purpose:   Typed configuration for a single MLflow run.
    Fields:    experiment_name — MLflow experiment to log under.
               run_name        — Optional human-readable run identifier.
               tags            — Key-value metadata attached to the run.
    """

    experiment_name: str = "default"
    run_name: str | None = None
    tags: Mapping[str, str] | None = None


@contextmanager
def start_run(config: RunConfig | None = None):
    """
    Purpose:    Context manager that wraps mlflow.start_run.
                Yields the active Run object, or None when tracking is off.
    Inputs:     config — RunConfig; falls back to MLFLOW_EXPERIMENT_NAME env
                         var or "default" when None.
    Side-effect: Creates or resumes an MLflow run on the tracking server.
    Error:      If the tracking server is unreachable, MLflow raises an
                exception that propagates to the caller.
    """
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


def log_params(params: Mapping[str, Any]) -> None:
    """
    Purpose:    Log a dictionary of hyperparameters to the active MLflow run.
                Values are coerced to str when not natively serialisable by
                MLflow (int, float, str, bool).
    Inputs:     params — mapping of parameter name → value.
    Precond:    An active MLflow run must exist (called within start_run context).
    Side-effect: Writes param entries to the MLflow tracking store.
    """
    mlflow = _load_mlflow()
    if not mlflow:
        return
    safe = {k: (v if isinstance(v, (int, float, str, bool)) else str(v)) for k, v in params.items()}
    mlflow.log_params(safe)


def log_metrics(metrics: Mapping[str, float], step: int | None = None) -> None:
    """
    Purpose:    Log scalar evaluation metrics to the active MLflow run.
    Inputs:     metrics — mapping of metric name → float value.
                step    — optional training step/epoch index for time-series
                          visualisation in the MLflow UI.
    Precond:    An active MLflow run must exist.
    Side-effect: Writes metric entries to the MLflow tracking store.
    """
    mlflow = _load_mlflow()
    if not mlflow:
        return
    mlflow.log_metrics({k: float(v) for k, v in metrics.items()}, step=step)


def log_artifact(path: str) -> None:
    """
    Purpose:    Upload a local file as an artifact of the active MLflow run.
    Inputs:     path — absolute or relative path to the file.
    Precond:    File must exist at path; active run must exist.
    Failure:    Silently skips when path does not exist to avoid breaking
                the pipeline on optional artifact generation failures.
    """
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
