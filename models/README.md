# Models

Trained models and model artifacts.

## Structure

- `trained/` - Production-ready models (*.pkl, *.joblib, *.pt, *.h5)
- `checkpoints/` - Training checkpoints and intermediate states
- `experiments/` - Experimental and research models
- `metadata/` - Model documentation, configs, and performance metrics
- `legacy/` - Backed up files from cleaned duplicate folders

## Naming Convention

- Include version: `model_v1.0.0.pkl`
- Include date: `model_20231201.pkl`
- Include metrics: `rf_acc_0.95_20231201.pkl`

## Model Storage Guidelines

1. **Production Models** → `trained/`
2. **Development Models** → `experiments/`
3. **Training States** → `checkpoints/`
4. **Documentation** → `metadata/`
5. **Legacy Files** → `legacy/` (from cleanup)

Never commit large model files to Git. Use Git LFS or cloud storage for models >100MB.
