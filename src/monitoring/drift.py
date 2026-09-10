"""
Data drift detection using Population Stability Index (PSI) and KS test.
"""

from typing import Dict, List, Tuple, Optional
import numpy as np
import pandas as pd
from scipy.stats import ks_2samp


def calculate_psi(
    expected: np.ndarray,
    actual: np.ndarray,
    bins: int = 10,
    eps: float = 1e-6,
) -> float:
    """
    Population Stability Index between two distributions.
    PSI < 0.1  → no significant drift
    0.1–0.25  → moderate drift
    > 0.25    → significant drift
    """
    # Remove NaNs
    expected = expected[~np.isnan(expected)]
    actual = actual[~np.isnan(actual)]

    if len(expected) == 0 or len(actual) == 0:
        return np.nan

    # Shared breakpoints
    breakpoints = np.histogram_bin_edges(expected, bins=bins)

    expected_counts, _ = np.histogram(expected, bins=breakpoints)
    actual_counts, _ = np.histogram(actual, bins=breakpoints)

    expected_perc = expected_counts / len(expected) + eps
    actual_perc = actual_counts / len(actual) + eps

    psi = np.sum((actual_perc - expected_perc) * np.log(actual_perc / expected_perc))
    return float(psi)


def ks_test(expected: np.ndarray, actual: np.ndarray) -> Tuple[float, float]:
    """Kolmogorov-Smirnov test. Returns (statistic, p-value)."""
    expected = expected[~np.isnan(expected)]
    actual = actual[~np.isnan(actual)]
    if len(expected) == 0 or len(actual) == 0:
        return np.nan, np.nan
    stat, pvalue = ks_2samp(expected, actual)
    return float(stat), float(pvalue)


def detect_feature_drift(
    reference_df: pd.DataFrame,
    current_df: pd.DataFrame,
    features: List[str],
    psi_threshold: float = 0.25,
) -> pd.DataFrame:
    """
    Compare reference (e.g. training) vs current (recent production) distributions.
    Returns a DataFrame with PSI, KS statistic and drift flag per feature.
    """
    rows = []
    for col in features:
        if col not in reference_df.columns or col not in current_df.columns:
            continue

        ref = reference_df[col].values.astype(float)
        cur = current_df[col].values.astype(float)

        psi = calculate_psi(ref, cur)
        ks_stat, ks_p = ks_test(ref, cur)

        rows.append({
            "feature": col,
            "psi": round(psi, 4) if not np.isnan(psi) else None,
            "ks_statistic": round(ks_stat, 4) if not np.isnan(ks_stat) else None,
            "ks_pvalue": round(ks_p, 4) if not np.isnan(ks_p) else None,
            "drift_detected": bool(psi >= psi_threshold) if not np.isnan(psi) else False,
        })

    return pd.DataFrame(rows).sort_values("psi", ascending=False)


def summarize_drift(drift_df: pd.DataFrame) -> Dict:
    """High-level drift summary."""
    if drift_df.empty:
        return {"n_features": 0, "n_drifted": 0, "max_psi": None, "status": "unknown"}

    n_drifted = int(drift_df["drift_detected"].sum())
    max_psi = float(drift_df["psi"].max())
    status = "significant" if n_drifted > 0 else "stable"

    return {
        "n_features": len(drift_df),
        "n_drifted": n_drifted,
        "max_psi": max_psi,
        "status": status,
        "drifted_features": drift_df.loc[drift_df["drift_detected"], "feature"].tolist(),
    }