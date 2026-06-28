# quantum-ai — Claude Context

## Project
Churn prediction on Telco customer churn dataset (7043 rows, 23 features after cleaning). Compares classical ML models against a Variational Quantum Classifier (VQC). Python 3.12, managed with uv.

## Running
```bash
uv run python src/main.py        # full run
uv run python src/main.py -qo   # quantum only (loads outputs/top_features.json)
```

## Key design decisions

**Polars not pandas** — all data manipulation uses Polars. `infer_schema_length=0` reads everything as strings on load; casting happens explicitly in the pipeline.

**Shared cleaning pipeline** — `src/data/pipeline.py` `CleaningPipeline` is used by all models. Outputs 23 features. The quantum path receives a pre-selected subset of `n_qubits` features chosen by aggregated classical feature importances.

**Config-driven** — all tuneable params in `config/pipeline.yaml`. No hardcoded values in model files. Quantum optimizer, n_qubits, n_layers, encoding strategy all set via YAML.

**Abstract base class** — `src/models/base.py` `BaseChurnModel` enforces `fit`/`predict`/`predict_proba` on all models. `evaluate()` is concrete and shared.

**Class imbalance** — handled differently per model: `class_weight='balanced'` for sklearn models, `scale_pos_weight` computed at fit time for XGBoost, decision threshold = class prior for MLP.

**VQC feature selection** — classical models run first, their feature importances are averaged, top `n_qubits` features saved to `outputs/top_features.json` and reused for quantum-only reruns.

## Quantum pipeline notes

- Uses PennyLane `lightning.qubit` with `diff_method='adjoint'` (much faster than parameter-shift on simulator)
- `pennylane.numpy` is imported as `np` for gradient tracking during training; standard `numpy` imported as `_np` for inference in `predict_proba`/`predict` to avoid pennylane array tracking issues that caused duplicate outputs
- Adam optimizer recommended — gradient descent hits barren plateaus easily
- Angle encoding uses `RY(feature × π)` per qubit; amplitude encoding pads to 2^n_qubits with zeros
- `max_train_samples` limits VQC training data for speed; evaluation still uses full test set

## File layout
```
src/
  config.py, schema.py, main.py
  data/loader.py, data/pipeline.py
  models/base.py
  classical/{logistic,random_forest,xgboost_model,mlp,ensemble}.py
  quantum/encoding.py, quantum/vqc.py
  viz/eda.py, viz/evaluation.py
config/pipeline.yaml
```

## Known gotchas
- `TotalCharges` has 11 whitespace rows — cast with `strict=False`, imputed with median
- XGBoost needs `brew install libomp` on macOS
- VQC with `lightning.qubit` requires `pennylane-lightning` (`uv add pennylane-lightning`)
- `parameter-shift` diff_method is incompatible with `lightning.qubit` — use `adjoint`
- EDA plots run on raw df (before cleaning) so labels are human-readable
- `outputs/top_features.json` must exist before using `-qo` flag
