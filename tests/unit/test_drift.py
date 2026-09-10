"""Unit tests for drift detection."""

import numpy as np
import pandas as pd
from src.monitoring.drift import calculate_psi, detect_feature_drift, summarize_drift


def test_psi_identical_distributions():
    x = np.random.randn(1000)
    psi = calculate_psi(x, x.copy())
    assert psi < 0.05  # nearly zero


def test_psi_shifted_distribution():
    ref = np.random.randn(1000)
    cur = np.random.randn(1000) + 2.0  # clear shift
    psi = calculate_psi(ref, cur)
    assert psi > 0.25


def test_detect_feature_drift():
    ref = pd.DataFrame({"a": np.random.randn(500), "b": np.random.randn(500)})
    cur = pd.DataFrame({"a": np.random.randn(500) + 3, "b": np.random.randn(500)})
    drift_df = detect_feature_drift(ref, cur, ["a", "b"], psi_threshold=0.25)
    assert "a" in drift_df["feature"].values
    summary = summarize_drift(drift_df)
    assert "status" in summary