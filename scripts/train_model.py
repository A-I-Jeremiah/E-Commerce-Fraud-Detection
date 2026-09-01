"""
Phase 3 – Train baseline + primary XGBoost model.
Saves versioned artifacts to models/.
"""

import sys 
from pathlib import Path
from datetime import datetime
import json

from itsdangerous import TimestampSigner
from pandas import Timestamp

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

import joblib
import numpy as np
import pandas as pd
import xgboost as xgb

from src.config import PROCESSED_DATA_DIR, MODELS_DIR, TARGET_COL, COST_FALSE_NEGATIVE, COST_FALSE_POSITIVE, RANDOM_SEED
from src.models.train import train_logistic_regression, train_xgboost
from src.models.evaluate import print_evaluation_report, find_best_threshold, compute_metrics

def load_processed_data():
    """Load the saved parquet artifacts"""
    X_train = pd.read_parquet(PROCESSED_DATA_DIR / "X_train.parquet").values
    X_test = pd.read_parquet(PROCESSED_DATA_DIR / "X_test.parquet").values
    X_val = pd.read_parquet(PROCESSED_DATA_DIR / "X_val.parquet").values

    # Ensure y arrays are 1D to satisfy scikit-learn expectations (avoid column-vector warnings)
    y_train = pd.read_parquet(PROCESSED_DATA_DIR / "y_train.parquet").to_numpy().ravel()
    y_test = pd.read_parquet(PROCESSED_DATA_DIR / "y_test.parquet").to_numpy().ravel()
    y_val = pd.read_parquet(PROCESSED_DATA_DIR / "y_val.parquet").to_numpy().ravel()

    meta = joblib.load(PROCESSED_DATA_DIR / "meta.joblib")
    feature_names = meta["feature_names"]

    
    return X_train, X_test, X_val, y_train, y_test, y_val, feature_names

def main():
    print("=" * 70)
    print("PHASE 3 - Model Training: Primary Model (XGBoost) & Baseline Model (Logistic Regression)")

    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # 1. Load data
    print("\nLoading proceesed data...")
    X_train, X_test, X_val, y_train, y_test, y_val, feature_names = load_processed_data()
    print(f"Train = X_train Shape: {X_train.shape}, y_train Shape: {y_train.shape}")
    print(f"Test = X_test Shape:{X_test.shape}, y_test Shape: {y_test.shape}")
    print(f"Val = X_val Shape: {X_val.shape}, y_val Shape: {y_val.shape}")

    results = {}

    # 2. Train Baseline Model (Logistic Regression)
    print("=" * 50)
    print("Training Baseline Model - Logistic Regression...")
    print("=" * 50)
    lr_model = train_logistic_regression(X_train, y_train)
    lr_val_prob = lr_model.predict_proba(X_val)[:, 1]
    lr_test_prob = lr_model.predict_proba(X_test)[:, 1]

    print_evaluation_report(y_val, lr_val_prob, split_name="Val - Logistic @0.5", threshold=0.5, label_names=["Legit", "Fraud"])
    lr_best = find_best_threshold(y_val, lr_val_prob, cost_fp=COST_FALSE_POSITIVE, cost_fn=COST_FALSE_NEGATIVE, label_names=["Legit", "Fraud"])
    print(f"\nLogistic Regression Best Cost-Sensitive Threshold (val): {lr_best['threshold']:.3f}"
          f"| Cost = {lr_best['cost']:.3f}") 

    results["Logistic Regression"] = {
        "val": compute_metrics(y_val, lr_val_prob, threshold=lr_best["threshold"], label_names=["Legit", "Fraud"]),
        "test": compute_metrics(y_test, lr_test_prob, threshold=lr_best["threshold"], label_names=["Legit", "Fraud"]),
        "best_threshold": lr_best["threshold"]
    }

    # Train Primary Model (XGBoost Model)
    print("\n=" * 50)
    print("Train Primary Model - XGBoost...")
    print("=" * 50)
    xgb_model, history = train_xgboost(X_train, y_train, X_val, y_val, params=None)

    dval = xgb.DMatrix(X_val)
    dtest = xgb.DMatrix(X_test)

    xgb_val_prob = xgb_model.predict(dval)
    xgb_test_prob = xgb_model.predict(dtest)

    print_evaluation_report(y_val, xgb_val_prob, split_name="Val - XGBoost @0.5", threshold=0.5, label_names=["Legit", "Fraud"])

    # Find XGBoost Best Threshold
    xgb_best = find_best_threshold(y_val, xgb_val_prob, cost_fp=COST_FALSE_POSITIVE, cost_fn=COST_FALSE_NEGATIVE, label_names=["Legit", "Fraud"])
    print(f"\nXGBoost Best Cost-Sensitive Threshold (val): {xgb_best['threshold']:.3f}"
          f"| Cost = {xgb_best['cost']:.3f}")

    # Val Evaluation using Best Threshold
    print_evaluation_report(y_val, xgb_val_prob, split_name="Val - XGBoost @Best Threshold", threshold=xgb_best["threshold"], label_names=["Legit", "Fraud"])

    # Test Evaluation using Best Threshold
    print_evaluation_report(y_test, xgb_test_prob, split_name="Test - XGBoost @Best Threshold", threshold=xgb_best["threshold"], label_names=["Legit", "Fraud"])

    # Save to results
    results["XGBoost"] = {
        "val": compute_metrics(y_val, xgb_val_prob, threshold=xgb_best["threshold"], label_names=["Legit", "Fraud"]),
        "test": compute_metrics(y_test, xgb_test_prob, threshold=xgb_best["threshold"], label_names=["Legit", "Fraud"]),
        "best_threshold": xgb_best["threshold"]
    }

    # 4. Feature Importance (Top 20)
    print("Feature Importance by gain - XGBoost Model (Top 20):")
    score = xgb_model.get_score(importance_type="gain")
    # Map f0, f1, ... back to real feature names
    importance = pd.DataFrame([
        {"feature": feature_names[int(k[1:])], "gain": v}
        for k, v in score.items()
    ]).sort_values("gain", ascending=False)

    print(importance.head(20).to_string(index=False))

    # 5. Save Artifacts
    print("\nSaving model artifacts...")
    # XGBoost model (JSON format – portable & readable)
    model_path = MODELS_DIR / f"xgboost_{timestamp}.json"
    xgb_model.save_model(str(model_path))   

    # Latest Pointer
    latest_path = MODELS_DIR / "xgboost_latest.json"
    xgb_model.save_model(str(latest_path))

    # MetaData
    meta = {
        "timestamp": timestamp,
        "model_type": "xgboost",
        "best_threshold": float(xgb_best["threshold"]),
        "feature_names": feature_names,
        "metrics": results,
        "cost_fp": COST_FALSE_POSITIVE,
        "cost_fn": COST_FALSE_NEGATIVE,
        "random_seed": RANDOM_SEED
    }

    def make_json_safe(obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, (np.integer, np.floating)):
            return obj.item()
        if isinstance(obj, dict):
            return {k: make_json_safe(v) for k, v in obj.items()}
        if isinstance(obj, list):
            return [make_json_safe(v) for v in obj]
        return obj

    meta_safe = make_json_safe(meta)

    meta_path = MODELS_DIR / f"xgboost_{timestamp}_meta.json"
    with open(meta_path, "w") as f:
        json.dump(meta_safe, f, indent=2)

    with open(MODELS_DIR / "xgboost_latest_meta.json", "w") as f:
        json.dump(meta_safe, f, indent=2)

    # Logistic Baseline
    joblib.dump(lr_model, MODELS_DIR / f"logistic_{timestamp}.joblib")

    print(F"\nArtifacts saved to: {MODELS_DIR}...")
    print(f"      - {model_path.name}")
    print(f"      - {meta_path.name}")
    print(f"      - xgboost_latest.json / xgboost_latest_meta.json")
    print("\nPHASE 3 Complete.")

if __name__ == "__main__":
    main()