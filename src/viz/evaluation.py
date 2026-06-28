from pathlib import Path
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from sklearn.metrics import (
    roc_curve, auc, precision_recall_curve,
    average_precision_score, confusion_matrix,
)

OUTPUT_DIR = Path("outputs")


def save(fig, filename: str) -> None:
    OUTPUT_DIR.mkdir(exist_ok=True)
    fig.write_html(OUTPUT_DIR / filename)


def plot_roc_curve(results: dict) -> None:
    fig = go.Figure()
    for model_name, data in results.items():
        fpr, tpr, _ = roc_curve(data['y_true'], data['y_score'])
        roc_auc = auc(fpr, tpr)
        fig.add_trace(go.Scatter(
            x=fpr.tolist(), y=tpr.tolist(),
            name=f'{model_name} (AUC={roc_auc:.3f})',
            mode='lines',
        ))
    fig.add_trace(go.Scatter(
        x=[0, 1], y=[0, 1],
        name='Random',
        mode='lines',
        line=dict(dash='dash', color='grey'),
    ))
    fig.update_layout(
        title='ROC Curves',
        xaxis_title='False Positive Rate',
        yaxis_title='True Positive Rate',
        template='plotly_dark',
        height=500,
    )
    save(fig, 'roc_curves.html')


def plot_pr_curve(results: dict) -> None:
    fig = go.Figure()
    for model_name, data in results.items():
        precision, recall, _ = precision_recall_curve(data['y_true'], data['y_score'])
        ap = average_precision_score(data['y_true'], data['y_score'])
        fig.add_trace(go.Scatter(
            x=recall.tolist(), y=precision.tolist(),
            name=f'{model_name} (AP={ap:.3f})',
            mode='lines',
        ))
    fig.update_layout(
        title='Precision-Recall Curves',
        xaxis_title='Recall',
        yaxis_title='Precision',
        template='plotly_dark',
        height=500,
    )
    save(fig, 'pr_curves.html')


def plot_confusion_matrix(y_true: np.ndarray, y_pred: np.ndarray, model_name: str) -> None:
    cm = confusion_matrix(y_true, y_pred)
    fig = px.imshow(
        cm,
        text_auto=True,
        x=['Predicted No', 'Predicted Yes'],
        y=['Actual No', 'Actual Yes'],
        color_continuous_scale='Blues',
        title=f'Confusion Matrix — {model_name}',
        template='plotly_dark',
    )
    save(fig, f'confusion_matrix_{model_name}.html')


def plot_feature_importance(feature_names: list, importances: np.ndarray, model_name: str, top_n: int = 15) -> None:
    indices = np.argsort(importances)[-top_n:]
    fig = go.Figure(go.Bar(
        x=importances[indices].tolist(),
        y=[feature_names[i] for i in indices],
        orientation='h',
    ))
    fig.update_layout(
        title=f'Feature Importance — {model_name}',
        xaxis_title='Importance',
        template='plotly_dark',
        height=500,
    )
    save(fig, f'feature_importance_{model_name}.html')


def plot_model_comparison(results: dict) -> None:
    model_names = list(results.keys())
    accuracies = [results[m]['accuracy'] for m in model_names]
    roc_aucs = [results[m]['roc_auc'] for m in model_names]

    fig = go.Figure(data=[
        go.Bar(name='Accuracy', x=model_names, y=accuracies),
        go.Bar(name='ROC-AUC', x=model_names, y=roc_aucs),
    ])
    fig.update_layout(
        barmode='group',
        title='Model Comparison',
        yaxis_title='Score',
        template='plotly_dark',
        height=400,
    )
    save(fig, 'model_comparison.html')
