"""Unit tests for evaluation metrics."""

import numpy as np
from src.models.evaluate import compute_metrics, cost_sensitive_score, find_best_threshold


def test_compute_metrics_perfect():
    y_true = np.array([0, 0, 1, 1])
    y_prob = np.array([0.1, 0.2, 0.8, 0.9])
    m = compute_metrics(y_true, y_prob, threshold=0.5)
    assert m["precision"] == 1.0
    assert m["recall"] == 1.0
    assert m["f1"] == 1.0
    assert 0.9 < m["pr_auc"] <= 1.0


def test_cost_sensitive_score():
    y_true = np.array([0, 1])
    y_prob = np.array([0.6, 0.4])  # 1 FP, 1 FN at threshold 0.5
    cost = cost_sensitive_score(y_true, y_prob, threshold=0.5, cost_fp=5.0, cost_fn=100.0)
    assert cost == 105.0


def test_find_best_threshold_returns_valid():
    y_true = np.array([0, 0, 1, 1, 0, 1])
    y_prob = np.array([0.1, 0.4, 0.6, 0.9, 0.3, 0.7])
    best = find_best_threshold(y_true, y_prob, cost_fp=5.0, cost_fn=100.0)
    assert 0.01 <= best["threshold"] <= 0.99
    assert "metrics" in best