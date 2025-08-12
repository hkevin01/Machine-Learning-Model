---
title: Agent Mode Kickoff Prompt Template
---

## Agent Mode Kickoff Prompt Template

Copy, fill the bracketed fields, and paste into Agent Mode after launching `./run_agent.sh`.

```text
Goal
- Problem type: [classification | regression | clustering | semi-supervised]
- Business context: [brief description]
- Success metric(s): [ROC-AUC, F1, MAPE, silhouette, etc.]
- Constraints: [training time, model size, interpretability, fairness, etc.]
- Deployment target: [local artifact | API | batch job]
- Monitoring needs: [data drift | performance degradation | retraining cadence]

Data
- Dataset location: [./data/raw/my_dataset.csv]
- Target column: [label]
- Feature notes: [categorical/ordinal fields, leakage concerns, missing values]
- Class balance: [balanced | imbalanced; handling]

Required Workflow
1) Data validation & schema inference (types, missingness summary)
2) EDA with automated plots (save to docs/reports & notebooks/)
3) Preprocessing plan (encoding, scaling, imputation) with justification
4) Feature engineering & importance / selection evaluation
5) Intelligent splits (train/val/test, stratify if needed)
6) Baselines (simple models for floor)
7) Model selection (≥3 suitable families)
8) Hyperparameter tuning w/ early stopping; report best config
9) Final evaluation (calibrated metrics + error analysis)
10) Persist artifacts (models/, preprocessing pipeline)
11) Export reproducible pipeline (code + params) & one-page summary
12) Propose monitoring strategy & drift checks
13) Produce runbook in docs/ with exact commands & next steps

Deliverables
- Reports: docs/reports/*.md + figures in docs/figures/
- Artifacts: models/best_model.(pkl|joblib) + preprocessing pipeline
- Repro: env (pip freeze), RNG seeds, CLI steps
- Risks: assumptions, data issues, mitigations
- Roadmap: next actions for +10% metric improvement

Run
- Begin at Data Collection; proceed step-by-step, request clarification if blocked.
- Use State Persistence so session can pause/resume.
```
