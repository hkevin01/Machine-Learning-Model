# Runbook: MLflow Experiment Tracking

## Purpose

Consistent, reproducible experiment logging (params, metrics, artifacts) for built‑in algorithms.

## Prerequisites

- Install dev deps: `pip install -r requirements-dev.txt`
- (Optional) Start local UI: `make mlflow-ui` (<http://localhost:5000>)

## Configuration

Set in `.env` (copy `.env.example`):

```bash
MLFLOW_TRACKING_URI=http://localhost:5000
MLFLOW_EXPERIMENT_NAME=default
```

Disable tracking with `MLFLOW_DISABLE=1`.

## What Gets Logged

- Parameters: core hyperparameters for decision tree & random forest
- Feature importances (artifact: `feature_importances.json`)

## Usage Pattern

Just call `.fit()` on supported algorithms; logging occurs automatically when MLflow available.

## Common Operations

```bash
mlflow ui --port 5000
# View runs, compare metrics, download artifacts
```

## Troubleshooting

| Symptom | Action |
|---------|-------|
| No runs appear | Verify MLflow installed & env vars; check `pip show mlflow` |
| Connection refused | Ensure tracking server/ UI is running |
| Permission errors (remote) | Validate credentials / tokens for remote backend |

## Cleanup

Delete `.mlruns/` (local backend) or prune remote store per retention policy.
