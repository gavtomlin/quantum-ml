import sys
import json
import argparse
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).parent))

TOP_FEATURES_PATH = Path("outputs/top_features.json")

from viz.eda import run_eda
from viz.evaluation import (
    plot_roc_curve, plot_pr_curve,
    plot_confusion_matrix, plot_feature_importance,
    plot_model_comparison,
)

from config import load_config
from data.loader import load_raw
from data.pipeline import CleaningPipeline
from classical.logistic import LogisticChurnModel
from classical.random_forest import RandomForestChurnModel
from classical.xgboost_model import XGBoostChurnModel
from classical.mlp import MLPChurnModel
from classical.ensemble import EnsembleChurnModel
from classical.tabicl import TabICLChurnModel
from quantum.vqc import VQCChurnModel
from quantum.svm import QuantumSVMModel
from sklearn.model_selection import train_test_split


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-qo', '--quantum-only', action='store_true',
                        help='Skip classical models, load saved feature indices')
    parser.add_argument('-co', '--classical-only', action='store_true',
                        help='Skip quantum model, run classical models only')
    args = parser.parse_args()

    config = load_config("config/pipeline.yaml")

    df = load_raw(config.data.path)
    if not args.classical_only:
        run_eda(df)
    X, y = CleaningPipeline(config).fit_transform(df)

    X_np = X.to_numpy()
    y_np = y.to_numpy()

    X_train, X_test, y_train, y_test = train_test_split(
            X_np, y_np,
            test_size=config.split.test_size, 
            random_state=config.split.random_state, 
    )

    eval_results = {}

    if args.quantum_only:
        if not TOP_FEATURES_PATH.exists():
            raise FileNotFoundError("Run without -qo first to generate feature indices")
        with open(TOP_FEATURES_PATH) as f:
            saved = json.load(f)
        top_indices = np.array(saved['indices'])
        top_features = saved['features']
        print(f"Loaded top features: {top_features}")
    else:
        classical_models = [
                TabICLChurnModel(config),
                LogisticChurnModel(config),
                RandomForestChurnModel(config),
                XGBoostChurnModel(config),
                MLPChurnModel(config),
                EnsembleChurnModel(config),
            ]

        importances = []

        for model in classical_models:
            print(f"\nfitting {model.name}...")
            model.fit(X_train, y_train)
            print(f"predicting {model.name}...")
            y_pred = model.predict(X_test)
            y_score = model.predict_proba(X_test)[:, 1]
            results = model.evaluate(X_test, y_test)
            eval_results[model.name] = {
                'y_true': y_test,
                'y_score': y_score,
                'accuracy': results['accuracy'],
                'roc_auc': results['roc_auc'],
            }
            print(f"\n{model.name}")
            print(f"  Accuracy: {results['accuracy']:.4f}")
            print(f"  ROC-AUC:  {results['roc_auc']:.4f}")
            print(results['report'])
            plot_confusion_matrix(y_test, y_pred, model.name)
            if hasattr(model, 'feature_importances_'):
                importances.append(model.feature_importances_)
                plot_feature_importance(X.columns, model.feature_importances_, model.name)

        # select top features for qubits
        avg_importance = np.mean(importances, axis=0)
        top_indices = np.argsort(avg_importance)[-config.quantum.n_qubits:]
        top_features = [X.columns[i] for i in top_indices]
        print(f"\nTop {config.quantum.n_qubits} features for VQC: {top_features}")

        # save for quantum-only reruns
        TOP_FEATURES_PATH.parent.mkdir(exist_ok=True)
        with open(TOP_FEATURES_PATH, 'w') as f:
            json.dump({'indices': top_indices.tolist(), 'features': top_features}, f)

    if not args.classical_only:
        X_train_q = X_train[:, top_indices]
        X_test_q = X_test[:, top_indices]

        for qmodel in [VQCChurnModel(config), QuantumSVMModel(config)]:
            qmodel.fit(X_train_q, y_train)
            y_pred_qm = qmodel.predict(X_test_q)
            y_score_qm = qmodel.predict_proba(X_test_q)[:, 1]
            results_qm = qmodel.evaluate(X_test_q, y_test)
            eval_results[qmodel.name] = {
                'y_true': y_test,
                'y_score': y_score_qm,
                'accuracy': results_qm['accuracy'],
                'roc_auc': results_qm['roc_auc'],
            }
            print(f"\n{qmodel.name}")
            print(f"  Accuracy: {results_qm['accuracy']:.4f}")
            print(f"  ROC-AUC:  {results_qm['roc_auc']:.4f}")
            print(results_qm['report'])
            plot_confusion_matrix(y_test, y_pred_qm, qmodel.name)
   

    plot_roc_curve(eval_results)
    plot_pr_curve(eval_results)
    plot_model_comparison(eval_results)
    print("\nEvaluation plots saved to outputs/")

if __name__ == "__main__":
    main()
