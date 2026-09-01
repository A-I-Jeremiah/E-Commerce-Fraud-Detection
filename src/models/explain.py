"""
SHAP-based model explainability for the XGBoost fraud model.
"""

from pathlib import Path
from typing import List
import shap
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import xgboost as xgb


def _resolve_output_path(output_path: Path, filename: str) -> Path:
    """Accept either a directory or a full file path."""
    if output_path.suffix:
        return output_path
    return output_path / filename


def compute_shap_values(model: xgb.Booster, X: np.ndarray, feature_names: List[str], max_samples: 2000) -> shap.Explanation:
    """
    Compute SHAP values using TreeExplainer.
    Subsamples for speed if the dataset is large.
    """
    if len(X) > max_samples:
        idx = np.random.choice(len(X), max_samples, replace=False)
        X_samples = X[idx]
    else:
        X_samples = X

    explainer = shap.TreeExplainer(model)
    raw_values = explainer.shap_values(X_samples)

    # shap 0.46+ can return a numpy array for binary classification; older versions may return a list.
    if isinstance(raw_values, list):
        values = np.asarray(raw_values[1] if len(raw_values) > 1 else raw_values[0])
    else:
        values = np.asarray(raw_values)

    if values.ndim == 3 and values.shape[-1] == 2:
        values = values[..., 1]
    elif values.ndim == 1:
        values = values.reshape(len(X_samples), -1)

    base_values = explainer.expected_value
    if isinstance(base_values, list):
        base_values = np.asarray(base_values)
        if base_values.ndim > 0 and base_values.shape[-1] == 2:
            base_values = base_values[1]

    return shap.Explanation(
        values=values,
        base_values=base_values,
        data=X_samples,
        feature_names=feature_names,
    )


def plot_shap_summary(shap_values: shap.Explanation, output_path: Path, max_display: int = 20):
    """
    Generate and save a Beeswarm summary plot.
    """
    target_path = _resolve_output_path(output_path, "shap_summary_plot.png")
    plt.figure(figsize=(10, 6))
    shap.plots.beeswarm(shap_values, max_display=max_display, show=False)
    plt.title("SHAP Beeswarm Summary Plot")
    plt.tight_layout()
    plt.savefig(target_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"SHAP summary plot saved to {target_path}")


def plot_shap_bar(shap_values: shap.Explanation, output_path: Path, max_display: int = 20):
    """Mean |SHAP| bar plot."""
    target_path = _resolve_output_path(output_path, "shap_bar_plot.png")
    plt.figure(figsize=(10, 6))
    shap.plots.bar(shap_values, max_display=max_display)
    plt.title("SHAP Bar Plot")
    plt.tight_layout()
    plt.savefig(target_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"SHAP bar plot saved to {target_path}")


def get_top_features_by_shap(shap_values: shap.Explanation, top_n: int = 20) -> pd.DataFrame:
    """Return a DataFrame of features ranked by mean |SHAP|."""
    mean_abs = np.abs(shap_values.values).mean(axis=0)
    df = pd.DataFrame({
        "feature": shap_values.feature_names,
        "mean_abs_shap": mean_abs
    }).sort_values(by="mean_abs_shap", ascending=False).head(top_n)
    return df