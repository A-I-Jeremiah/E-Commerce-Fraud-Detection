"""
Performance monitoring once ground-truth labels become available.
"""

from typing import Dict, Any
import numpy as np
from sklearn.metrics import (
    average_precision_score,
    roc_auc_score,
    f1_score,
    precision_score,
    recall_score,
    brier_score_loss,
)


def evaluate_live_performance(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    threshold: float,
) -> Dict[str, float]:
    """Compute key metrics on a labelled production window."""
    y_pred = (y_prob >= threshold).astype(int)

    return {
        "pr_auc": float(average_precision_score(y_true, y_prob)),
        "roc_auc": float(roc_auc_score(y_true, y_prob)),
        "f1": float(f1_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "brier_score": float(brier_score_loss(y_true, y_prob)),
        "n_samples": int(len(y_true)),
        "fraud_rate": float(y_true.mean()),
    }


def check_performance_degradation(
    current_metrics: Dict[str, float],
    baseline_metrics: Dict[str, float],
    pr_auc_drop_threshold: float = 0.05,
) -> Dict[str, Any]:
    """
    Simple rule: if PR-AUC drops more than `pr_auc_drop_threshold`
    relative to the training/validation baseline → flag degradation.
    """
    drop = baseline_metrics.get("pr_auc", 0) - current_metrics.get("pr_auc", 0)
    degraded = drop >= pr_auc_drop_threshold

    return {
        "degraded": degraded,
        "pr_auc_drop": round(drop, 4),
        "current_pr_auc": current_metrics.get("pr_auc"),
        "baseline_pr_auc": baseline_metrics.get("pr_auc"),
        "message": (
            f"PR-AUC dropped by {drop:.4f}" if degraded
            else "Performance within acceptable range"
        ),
    }