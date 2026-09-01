"""
Evaluation metrics and reporting for fraud detection.
Focus on Precision-Recall metrics and cost-sensitive views.
"""

import joblib
from pathlib import Path
from typing import Dict, Any
import numpy as np
from sklearn.metrics import (
    average_precision_score,
    roc_auc_score,
    auc,
    f1_score,
    precision_score,
    recall_score,
    confusion_matrix,
    precision_recall_curve
)


def compute_metrics(y_true: np.ndarray, y_prob: np.ndarray, threshold: float = 0.5, label_names: list = ["Legit", "Fraud"]) -> Dict[str, float]:
    """
    Compute evaluation metrics for binary classification.

    Args:
        y_true (np.ndarray): True binary labels.
        y_prob (np.ndarray): Predicted probabilities for the positive class.
        threshold (float): Threshold to convert probabilities to binary predictions.
        label_names (list): Names for the two classes.

    Returns:
        Dict[str, float]: Dictionary containing evaluation metrics.
    """
    # Convert probabilities to binary predictions based on the threshold
    y_pred = (y_prob >= threshold).astype(int)

    # Calculate Precision-Recall AUC
    precision, recall, _ = precision_recall_curve(y_true, y_prob)
    pr_auc = auc(recall, precision)

    # Compute metrics
    metrics = {
        "average_precision": average_precision_score(y_true, y_prob),
        "roc_auc": roc_auc_score(y_true, y_prob),
        "pr_auc": pr_auc,
        "f1_score": f1_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred),
        "recall": recall_score(y_true, y_pred),
        "confusion_matrix": confusion_matrix(y_true, y_pred)
    }

    return metrics

def cost_sensitive_score(y_true: np.ndarray, y_prob: np.ndarray, threshold: float, cost_fp: float = 5.0, cost_fn: float = 100.0) -> float:
    """"
    Expected cost = FP * cost_fp + FN * cost_fn (Lower is better)
    """
    y_pred = (y_prob >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    expected_cost = (fp * cost_fp) + (fn * cost_fn) 
    return float(expected_cost)

def find_best_threshold(y_true: np.ndarray, y_prob: np.ndarray, cost_fp: float=5.0, cost_fn: float=100.0, 
                        n_thresholds: int=200, label_names: list = ["Legit", "Fraud"]) -> Dict[str, Any]:
    """Search for the best threshold that minimizes expected cost"""
    thresholds = np.linspace(0, 1, n_thresholds)
    best = {"threshold": 0.5, "cost": float("inf"), "metrics": {}}

    for t in thresholds:
        cost = cost_sensitive_score(y_true, y_prob, t, cost_fp, cost_fn)
        if cost < best["cost"]:
            metrics = compute_metrics(y_true, y_prob, threshold=t, label_names=label_names)
            best = {"threshold": float(t), "cost": cost, "metrics": metrics}

    return best

def print_evaluation_report(y_true: np.ndarray, y_prob: np.ndarray, threshold: float = 0.5, split_name: str = "Validation", 
                            cost_fp: float = 5.0, cost_fn: float = 100.0, label_names: list = ["Legit", "Fraud"]) -> Dict[str, Any]:
    """"
    Print evaluation report and return metrics dictionary.
    """
    metrics = compute_metrics(y_true, y_prob, threshold, label_names)
    tn, fp, fn, tp = confusion_matrix(y_true, (y_prob >= threshold).astype(int)).ravel()
    cost = cost_sensitive_score(y_true, y_prob, threshold, cost_fp, cost_fn)


    print(f"\n{'='*60}")
    print(f"Evaluation Report - {split_name} Set, Threshold: {threshold:.2f}")
    print(f"\n{'='*60}")
    print(f" PR-AUC (Average Precision): {metrics['pr_auc']:.4f}")
    print(f" ROC-AUC                   : {metrics['roc_auc']:.4f}")
    print(f"Precision                  : {metrics['precision']:.4f}")
    print(f"Recall                     : {metrics['recall']:.4f}")
    print(f"F1-Score                   : {metrics['f1_score']:.4f}")
    print(f"Expected Cost (FP: {cost_fp}, FN: {cost_fn}): {cost:.2f}")
    print(f"\nConfusion Matrix:")
    print(f"                   Pred {label_names[0]}    Pred {label_names[1]}")
    print(f"  Actual {label_names[0]}     {tn:6d}           {fp:6d}")
    print(f"  Actual {label_names[1]}     {fn:6d}           {tp:6d}")
    print(f"{'='*60}\n")

    return {
        **metrics,
        "cost": cost,
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "tp": int(tp),
    }