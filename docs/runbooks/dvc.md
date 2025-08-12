# Runbook: DVC Data & Model Versioning

## Purpose

Track large data & derived model artifacts reproducibly alongside Git metadata.

## Initialization

```bash
make dvc-init   # creates .dvc if not present
```

## Pipeline

Defined in `dvc.yaml`:

- Stage `prepare`: generates processed sample dataset
- Stage `train`: trains sample decision tree model

Run:

```bash
dvc repro
```

## Adding Data

```bash
dvc add data/raw/my_dataset.csv
git add data/raw/my_dataset.csv.dvc .gitignore
git commit -m "Track dataset with DVC"
```

## Remotes

```bash
dvc remote add -d origin s3://my-bucket/dvc-store
dvc push
```

## Reproduce Environment

```bash
git clone <repo>
pip install -r requirements-dev.txt
make dvc-init
dvc pull
dvc repro
```

## Troubleshooting

| Issue | Resolution |
|-------|------------|
| Missing file after clone | Run `dvc pull` |
| Auth failure | Configure cloud credentials (AWS CLI, etc.) |
| Cache bloat | Use `dvc gc --workspace` |
