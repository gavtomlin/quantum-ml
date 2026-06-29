# quantum-ml: Churn Prediction with Classical and Quantum ML

A churn prediction system trained on the Telco customer churn dataset, comparing classical ML models against a Variational Quantum Classifier (VQC). Built to explore whether quantum methods can offer practical value on a real tabular classification problem.

---

## Dataset

**Telco Customer Churn** — 7043 rows × 21 columns. Target: `Churn` (Yes/No).

Features cover customer demographics, service subscriptions (phone, internet, streaming), contract type, billing method, and charges. The dataset has a ~75/25 class imbalance (non-churner majority).

---

## Setup

**Requirements:** Python 3.12, [uv](https://github.com/astral-sh/uv)

```bash
# Install dependencies
uv sync

# Install OpenMP (macOS only — required by XGBoost)
brew install libomp

# Install PennyLane Lightning backend (faster quantum simulation)
uv add pennylane-lightning
```

---

## Running

```bash
# Full run — classical models + VQC + all visualisations
uv run python src/main.py

# Quantum only — skip classical models, reuse saved feature indices
uv run python src/main.py -qo

# Classical only — skip quantum models
uv run python src/main.py -co
```

All plots saved to `outputs/` as interactive HTML files.

The `-qo` flag requires a prior full run to have generated `outputs/top_features.json`.

---

## Configuration

All pipeline parameters are in `config/pipeline.yaml`:

```yaml
data:
  path: src/data/customer-churn.csv
  output_dir: outputs

split:
  test_size: 0.2
  random_state: 10

classical:
  max_iter: 1000        # LogisticRegression
  n_estimators: 100     # RandomForest
  random_state: 10

quantum:
  n_qubits: 12
  n_layers: 6
  encoding: angle       # angle | amplitude
  n_steps: 100
  step_size: 0.5
  max_train_samples: 600
  optimizer: adam       # adam | gradient_descent | adagrad | momentum
  max_kernel_samples: 200  # quantum SVM kernel matrix is O(n²) circuits
```

---

## Repo Structure

```
src/
  config.py             # YAML loader, typed dataclasses
  schema.py             # Pydantic schema for raw CSV validation
  main.py               # Entry point, CLI arg parser
  data/
    loader.py           # CSV loader with smoke-test validation
    pipeline.py         # Shared cleaning pipeline (Polars)
  models/
    base.py             # Abstract BaseChurnModel
  classical/
    logistic.py         # Logistic Regression
    random_forest.py    # Random Forest
    xgboost_model.py    # XGBoost
    mlp.py              # MLP (Neural Network)
    ensemble.py         # Soft-voting ensemble of all classical models
    tabicl.py           # TabICL (Tabular In-Context Learning)
  quantum/
    encoding.py         # Angle and amplitude encoding layers (PennyLane)
    vqc.py              # Variational Quantum Classifier
    kernel.py           # Quantum kernel circuit
    svm.py              # Quantum Kernel SVM
  viz/
    eda.py              # EDA plots (Plotly)
    evaluation.py       # Model evaluation plots (ROC, PR, confusion matrix)
config/
  pipeline.yaml         # All tuneable parameters
```

---

## Data Pipeline

All models share the same cleaning pipeline (`src/data/pipeline.py`):

1. Drop `customerID`
2. Cast native numerics (`SeniorCitizen`, `tenure`, `MonthlyCharges`)
3. Fix `TotalCharges` — 11 rows have whitespace values, imputed with median
4. Binary encode Yes/No columns → 0/1
5. Collapse service columns (`No internet service`, `No phone service` → 0)
6. Ordinal encode `Contract` (Month-to-month=0, One year=1, Two year=2)
7. One-hot encode `InternetService` (keep all 3 categories)
8. One-hot encode `PaymentMethod` (drop-first, Bank transfer as reference)
9. Binary encode `gender`
10. Extract target: `Churn` → 0/1
11. `StandardScaler` on `tenure`, `MonthlyCharges`, `TotalCharges`

Output: 23 features, 7043 rows, zero nulls.

---

## Models

### Classical

All classical models inherit `BaseChurnModel` which enforces a consistent `fit` / `predict` / `predict_proba` / `evaluate` interface.

Class imbalance is handled per-model:
- **Logistic Regression** — `class_weight='balanced'`
- **Random Forest** — `class_weight='balanced'`
- **XGBoost** — `scale_pos_weight = n_negative / n_positive` computed at fit time
- **MLP** — decision threshold adjusted to class prior (`n_churners / n_total`) rather than default 0.5
- **Ensemble** — soft-voting across all classical models
- **TabICL** — pre-trained transformer for tabular in-context learning; no fine-tuning, passes training data as context at inference time. Trained on 200 samples (CPU only — MPS deadlocks due to internal PyTorch multiprocessing conflict)

### Quantum

Built with [PennyLane](https://pennylane.ai/) using the `lightning.qubit` simulator.

**Feature selection:** before VQC training, feature importances from the classical models are averaged and the top `n_qubits` features are selected. This gives the VQC the most informative features rather than arbitrary first-N.

**Circuit architecture:**
1. **Angle encoding** — each of the top N features is encoded as an RY rotation on its own qubit (`RY(feature × π)`)
2. **Entanglement** — CNOT gates in a ring topology (0→1→...→N→0) create qubit interactions, capturing feature relationships
3. **Variational layers** — trainable RY rotations per qubit per layer, optimised during training
4. **Measurement** — expectation value of PauliZ on qubit 0, converted from [-1,1] to [0,1] probability

**Gradient computation:** adjoint differentiation (single forward+backward pass, much faster than parameter-shift on a simulator).

**Optimisers available:** Adam (recommended), gradient descent, AdaGrad, momentum — configurable via YAML.

**Amplitude encoding** is also implemented — pads features to 2^n_qubits with zeros and L2-normalises. Switch via `encoding: amplitude` in config.

### Quantum Kernel SVM

An alternative quantum approach using the quantum kernel as a similarity function for a classical SVM, rather than training a variational circuit.

**How it works:**
1. For every pair of training points `(x_i, x_j)`, run both through the angle encoding circuit and compute their state overlap: `K(x_i, x_j) = |⟨φ(x_j)|φ(x_i)⟩|²`
2. This produces a precomputed `n×n` kernel matrix
3. sklearn's `SVC(kernel='precomputed')` uses this matrix to find the optimal hyperplane

The SVM itself has no quantum components — the quantum part is entirely in computing the similarity scores. The kernel matrix computation requires O(n²) circuit evaluations, making it expensive on a simulator (200 training samples = 40,000 circuits for training, 281,800 for test inference).

---

## Results

All models evaluated on the same 20% held-out test set (1409 samples). Classical models trained on all 23 features (~5634 training samples). Quantum models trained on top 12 features selected by averaged classical feature importance.

| Model | Accuracy | ROC-AUC | Churn Recall | Training data |
|-------|----------|---------|--------------|---------------|
| TabICL | 0.798 | 0.828 | 0.53 | 200 samples, 23 features |
| Logistic Regression | 0.736 | 0.843 | 0.78 | ~5634 samples, 23 features |
| Random Forest | 0.787 | 0.834 | 0.68 | ~5634 samples, 23 features |
| XGBoost | 0.765 | 0.832 | 0.66 | ~5634 samples, 23 features |
| MLP | 0.745 | 0.841 | 0.75 | ~5634 samples, 23 features |
| Ensemble | 0.793 | 0.846 | 0.68 | ~5634 samples, 23 features |
| VQC | 0.632 | 0.730 | 0.75 | 600 samples, 12 features |
| Quantum Kernel SVM | 0.764 | 0.699 | 0.46 | 200 samples, 12 features |

**Key observations:**

- **TabICL** achieves 0.828 ROC-AUC on only 200 training samples — the strongest accuracy of any model and close to the best ROC-AUC. This demonstrates the power of pre-trained meta-learning on tabular data: it has learned a general prior over tabular classification tasks that transfers directly without fine-tuning.
- **Ensemble** achieves the best ROC-AUC (0.846) by combining the diverse decision boundaries of all classical models.
- **VQC** improved significantly over the previous run — ROC-AUC 0.730 (up from 0.639) and churn recall 0.75, competitive with the best classical models. The class-weighted cost function is working.
- **Quantum Kernel SVM** achieves respectable accuracy (0.764) but lower ROC-AUC (0.699) than the VQC. The kernel approach captures a different similarity structure in Hilbert space but is heavily constrained by the 200-sample kernel matrix limit. The test kernel matrix (1409×200 = 281,800 circuit evaluations) ran in ~19 minutes.
- Quantum advantage on classical tabular data does not exist at this scale or on a simulator. The interesting result is how well both quantum methods perform given their severe data constraints relative to classical models.

---

## Visualisations

EDA plots (run on raw data, human-readable labels):
- `churn_rate_by_category.html` — churn rate per category across all categorical features
- `numeric_distributions.html` — tenure, MonthlyCharges, TotalCharges split by churn
- `correlation_heatmap.html` — Pearson correlation between numeric features
- `tenure_vs_charges.html` — scatter plot coloured by churn

Evaluation plots (generated after model training):
- `roc_curves.html` — ROC curves for all models on one chart
- `pr_curves.html` — Precision-Recall curves
- `model_comparison.html` — accuracy and ROC-AUC grouped bar chart
- `confusion_matrix_{model}.html` — per-model confusion matrix
- `feature_importance_{model}.html` — top 15 features per classical model

---

## Next Steps

### Follow-up project: quantum chemistry / molecular property prediction

The most credible near-term use case for quantum advantage is problems where the data is already quantum in nature. Molecular simulation is the canonical example — representing a molecule's electronic structure as a quantum state is natural, whereas encoding it classically requires exponential approximations.

**Variational Quantum Eigensolver (VQE) for molecular energy prediction**

- **Dataset:** QM9 (134k small organic molecules with DFT-computed properties) or PubChemQC
- **Task:** predict ground state energy or HOMO-LUMO gap for small molecules
- **Quantum approach:** represent the molecular Hamiltonian as a sum of Pauli operators; use VQE to find the minimum eigenvalue (ground state energy)
- **Classical baseline:** graph neural network (e.g. SchNet or DimeNet) operating on molecular graphs
- **Tools:** PennyLane's `qml.qchem` module has built-in Hamiltonian construction from molecular geometry; `pyscf` handles classical chemistry preprocessing

**Why this is a better test of quantum advantage:**

Classical methods (Hartree-Fock, DFT) use approximations that break down for strongly correlated electron systems. A quantum circuit can represent the full wavefunction exactly in Hilbert space — the same space the problem actually lives in. This means:

- State preparation is natural — molecules are quantum systems, not tabular rows
- The Hilbert space grows exponentially with electron count, which is exactly where classical simulation becomes intractable
- Results are verifiable against known experimental values
- Real quantum hardware (IBM Quantum, IonQ) is already being used for this class of problem at small scale

This would be a step from "quantum-inspired ML on classical data" — what this project explores — toward "quantum simulation of quantum systems", which is where the field's credible near-term advantage actually lies.
