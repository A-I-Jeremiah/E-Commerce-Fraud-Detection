"""
Model training utilities for fraud detection.
Baseline Model: Logistic Regression
Primary model: XGBoost
"""

from typing import Dict, Any, Tuple, Optional
import numpy as np
import xgboost as xgb
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from src.config import RANDOM_SEED
from src.pipeline.fraud_pipeline import build_preprocessing_pipeline

def get_scale_pos_weight(y:np.ndarray) -> float:
    """
    Calculate the scale_pos_weight for XGBoost based on class imbalance.

    Args:
        y (np.ndarray): Array of true binary labels.

    Returns:
        float: The scale_pos_weight value.
    """
    n_pos = y.sum()
    n_neg = len(y) - n_pos
    return float(n_neg / n_pos) if n_pos > 0 else 1.0

# Train Baseline Model: Logistic Regression
def train_logistic_regression(X_train: np.ndarray, y_train: np.ndarray, random_state: int = RANDOM_SEED) -> LogisticRegression:
    """Logistic regression baseline wrapped in a preprocessing pipeline.

    Uses StandardScaler to help with convergence and ensures y is a 1D array.
    """
    clf = LogisticRegression(
        class_weight="balanced",
        max_iter=2000,
        random_state=random_state,
        solver="lbfgs",
    )
    pipeline = Pipeline([
        ("scaler", StandardScaler()),
        ("clf", clf),
    ])
    # Ensure y is 1D for scikit-learn
    y_train_1d = y_train.ravel() if hasattr(y_train, "ravel") else y_train
    pipeline.fit(X_train, y_train_1d)
    return pipeline

# Train Primary Model: XGBoost
def train_xgboost(X_train: np.ndarray, y_train: np.ndarray, X_val: np.ndarray, y_val: np.ndarray, params: Optional[Dict[str, Any]], random_state: int = RANDOM_SEED,
                  num_boost_round: int = 1000, early_stopping_rounds: int = 50) -> Tuple[xgb.Booster, Dict[str, Any]]:
    """
    Train an XGBoost model with scale_pos_weight for class imbalance.

    Args:
        X_train (np.ndarray): Training features.
        y_train (np.ndarray): Training labels.
        X_val (np.ndarray): Validation features.
        y_val (np.ndarray): Validation labels.
        params (Dict[str, Any]): Hyperparameters for the XGBoost model.
        random_state (int): Random state for reproducibility.
        num_boost_round (int): Number of boosting rounds.
        early_stopping_rounds (int): Number of rounds to wait for improvement.

    Returns:
        Tuple[xgb.Booster, Dict[str, Any]]: Trained XGBoost model and evaluation metrics.
    """

    # Hyperparameters for XGBoost (from hyperparameter tuning in the notebook)
    if params is None:
        params = {
            "random_state": RANDOM_SEED,
            "objective": "binary:logistic",
            "eval_metric": "aucpr",
            "subsample": 0.7,
            "reg_lambda": 10,
            "reg_alpha": 0,
            "n_estimators": 800,
            "min_child_weight": 7,
            "max_depth": 3,
            "learning_rate": 0.01,
            "gamma": 0.5,
            "colsample_bytree": 0.85,
            "tree_method": "hist"
        }

    params = params.copy()
    # xgb.train uses num_boost_round rather than sklearn-style n_estimators; remove if present
    params.pop("n_estimators", None)
    params["scale_pos_weight"] = get_scale_pos_weight(y_train)

    dtrain = xgb.DMatrix(X_train, label=y_train)
    dval = xgb.DMatrix(X_val, label=y_val)

    evals = [(dtrain, "train"), (dval, "eval")]
    evals_result = {}

    model = xgb.train(
        params,
        dtrain,
        num_boost_round=num_boost_round,
        evals=evals,
        early_stopping_rounds=early_stopping_rounds,
        evals_result=evals_result,
        verbose_eval=100
    )

    return model, evals_result