# Runbook: Model & Data Monitoring

## Purpose

Detect dataset drift & quality issues using Evidently-generated reports.

## Script

`scripts/monitoring/generate_reports.py` produces HTML / JSON artifacts.

## Manual Run

```bash
make monitor
```

Artifacts appear under `test-outputs/` (or configured output directory).

## Scheduling Ideas

- GitHub Action nightly job calling `make monitor` and uploading artifacts
- Cron within container (sidecar) writing to object storage

## Extending

- Capture baseline stats post‑deployment; compare to rolling window
- Add alerting (e.g., threshold on PSI) via CI workflow step

## Troubleshooting

| Symptom | Action |
|---------|-------|
| Empty report | Ensure sample/baseline data present |
| Import error | Reinstall dev requirements |
