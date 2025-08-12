"""Train a simple model for DVC pipeline demonstration.

Adds structured logging and optional MLflow metric logging (accuracy).
"""
import joblib
from pathlib import Path
import pandas as pd
import numpy as np
from machine_learning_model.supervised.decision_tree import DecisionTreeClassifier

try:  # structured logging
    from machine_learning_model.logging_utils import get_logger
    _logger = get_logger(__name__)
except Exception:  # pragma: no cover
    import logging as _stdlib_logging
    _logger = _stdlib_logging.getLogger(__name__)

try:  # optional tracking
    from machine_learning_model.tracking import mlflow_tracking as _tracking  # type: ignore
except Exception:  # pragma: no cover
    _tracking = None


def main():
    df = pd.read_csv("data/processed/sample_prepared.csv")
    X = df.drop(columns=["target"]).values
    y = df["target"].values
    model = DecisionTreeClassifier(max_depth=4)
    model.fit(X, y)
    preds = model.predict(X)
    acc = float(np.mean(preds == y))
    _logger.info("sample_model_trained accuracy=%0.4f samples=%d", acc, len(y))
    if _tracking and _tracking.tracking_enabled():
        try:
            _tracking.log_metrics({"train_accuracy": acc})
        except Exception:  # pragma: no cover
            pass
    Path("models").mkdir(exist_ok=True)
    joblib.dump(model, "models/sample_decision_tree.pkl")


if __name__ == "__main__":
    main()
