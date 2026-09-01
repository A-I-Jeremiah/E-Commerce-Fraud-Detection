"""
Probability calibration and advanced evaluation utilities.
"""

from typing import Tuple, Dict, Any
import numpy as np
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss, log_loss


class ProbabilityCalibrator:
    """
    Calibrates raw model scores using Isotonic Regression (non-parametric)
    or Platt scaling (logistic).
    Fit only on validation predictions to avoid leakage.
    """

    def __init__(self, method: str = "isotonic"):
        assert method in {"isotonic", "platt"}
        self.method = method
        self.calibrator = None

    def fit(self, y_true: np.ndarray, y_prob: np.ndarray):
        if self.method == "isotonic":
            self.calibrator = IsotonicRegression(out_of_bounds="clip")
            self.calibrator.fit(y_prob, y_true)
        else:  # platt
            self.calibrator = LogisticRegression(solver="lbfgs")
            self.calibrator.fit(y_prob.reshape(-1, 1), y_true)
        return self

    def transform(self, y_prob: np.ndarray) -> np.ndarray:
        if self.calibrator is None:
            raise RuntimeError("Calibrator has not been fitted yet.")
        if self.method == "isotonic":
            return self.calibrator.predict(y_prob)
        return self.calibrator.predict_proba(y_prob.reshape(-1, 1))[:, 1]

    def fit_transform(self, y_true: np.ndarray, y_prob: np.ndarray) -> np.ndarray:
        self.fit(y_true, y_prob)
        return self.transform(y_prob)


def calibration_metrics(y_true: np.ndarray, y_prob: np.ndarray) -> Dict[str, float]:
    """Brier score and log-loss (lower is better)."""
    return {
        "brier_score": float(brier_score_loss(y_true, y_prob)),
        "log_loss": float(log_loss(y_true, y_prob)),
    }