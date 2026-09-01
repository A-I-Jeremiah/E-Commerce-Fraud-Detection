"""
Phase 4 – Calibration, SHAP, curves, and final threshold recommendation.
"""

import sys
from pathlib import Path
from datetime import datetime
import json

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

import joblib
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import xgboost as xgb
from sklearn.metrics import precision_recall_curve, roc_curve, auc

from src.config import PROCESSED_DATA_DIR, MODELS_DIR, TARGET_COL, COST_FALSE_POSITIVE, COST_FALSE_NEGATIVE
from src.models.evaluate import print_evaluation_report, find_best_threshold, compute_metrics, cost_sensitive_score
from src.models.calibration import ProbabilityCalibrator, calibration_metrics
from src.models.explain import compute_shap_values, plot_shap_summary, plot_shap_bar, get_top_features_by_shap

# Output Directories
REPORTS_DIR = PROJECT_ROOT / "reports"
PLOTS_DIR = REPORTS_DIR / "plots"
REPORTS_DIR.mkdir(parents=True, exist_ok=True)
PLOTS_DIR.mkdir(parents=True, exist_ok=True)

def load_data_and_model():
    """Load processed data + latest XGBoost model + metadata."""
    X_val = pd.read_parquet(PROCESSED_DATA_DIR / "X_val.parquet").values
    X_test = pd.read_parquet(PROCESSED_DATA_DIR / "X_test.parquet").values
    y_val = pd.read_parquet(PROCESSED_DATA_DIR / "y_val.parquet")[TARGET_COL].values
    y_test = pd.read_parquet(PROCESSED_DATA_DIR / "y_test.parquet")[TARGET_COL].values

    meta = joblib.load(PROCESSED_DATA_DIR / "meta.joblib")
    feature_names = meta["feature_names"]

    model = xgb.Booster()
    model.load_model(str(MODELS_DIR / "xgboost_latest.json"))

    with open(MODELS_DIR / "xgboost_latest_meta.json") as f:
        model_meta = json.load(f)

    return X_val, X_test, y_val, y_test, feature_names, model, model_meta


def make_json_safe(obj):
    """Recursively convert NumPy/scalar values to JSON-serializable Python objects."""
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, (np.integer, np.floating)):
        return obj.item()
    if isinstance(obj, dict):
        return {str(k): make_json_safe(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [make_json_safe(v) for v in obj]
    if isinstance(obj, tuple):
        return [make_json_safe(v) for v in obj]
    return obj


def plot_pr_curve(y_true, y_prob, title, path):
    precision, recall, _ = precision_recall_curve(y_true, y_prob)
    pr_auc = auc(recall, precision)
    plt.figure(figsize=(10, 6))
    plt.plot(recall, precision, label=f"PR Curve (AUC = {pr_auc:.4f})", color="blue")
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title(title)
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Precision-Recall curve saved to {path}")

def plot_cost_curve(y_true, y_prob, path, cost_fp=5.0, cost_fn=100):
    thresholds = np.linspace(0, 1, 200)
    cost = [cost_sensitive_score(y_true, y_prob, t, cost_fp, cost_fn) for t in thresholds]
    best_idx = np.argmin(cost)

    plt.figure(figsize=(10, 6))
    plt.plot(thresholds, cost, label="Expected Cost", color="red")
    plt.axvline(thresholds[best_idx], color="red", linestyle="--", label=f"Best Threshold = {thresholds[best_idx]:.2f}")
    plt.xlabel("Threshold")
    plt.ylabel("Expected Cost")
    plt.title("Cost vs Threshold Curve")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Cost curve saved to {path} with best threshold at {thresholds[best_idx]:.2f} and cost {cost[best_idx]:.2f}")

def main():
    print("=" * 70)
    print("PHASE 4 – CALIBRATION, SHAP & FINAL THRESHOLD")
    print("=" * 70)

    X_val, X_test, y_val, y_test, feature_names, model, model_meta = load_data_and_model()

    dval = xgb.DMatrix(X_val)
    dtest = xgb.DMatrix(X_test)

    # Raw Probabilities
    val_prob_raw = model.predict(dval)
    test_prob_raw = model.predict(dtest)

    # 1. Calibration - Fit on validation only
    print("\n--- Probability Calibration (Isotonic) ---")
    calibrator = ProbabilityCalibrator(method="isotonic")
    val_prob_cal = calibrator.fit_transform(y_val, val_prob_raw)
    test_prob_cal = calibrator.transform(test_prob_raw)

    print("Validation Calibratoin Metrics")
    print(f"Raw Probabilities:        {calibration_metrics(y_val, val_prob_raw)}")
    print(f"Calibrated Probabilities: {calibration_metrics(y_val, val_prob_cal)}")

    print("\nTest Calibration Metrics")
    print(f"Raw Probabilities:        {calibration_metrics(y_test, test_prob_raw)}")
    print(f"Calibrated Probabilities: {calibration_metrics(y_test, test_prob_cal)}")

    # 2. Threshold Optimization on calibrated validation scores
    print("\nCost-Sensitive Search {calibrated}")
    best = find_best_threshold(y_val, val_prob_cal, cost_fp=COST_FALSE_POSITIVE, cost_fn=COST_FALSE_NEGATIVE)
    final_threshold = best["threshold"]
    print(f"Recommended Threshold: {final_threshold:.4f} with Expected Cost: {best['cost']:.2f}")

    # 3. Final Evaluation Report
    print_evaluation_report(y_val, val_prob_cal, split_name="Validation (Calibrated)", threshold=final_threshold, cost_fp=COST_FALSE_POSITIVE, cost_fn=COST_FALSE_NEGATIVE)
    test_metrics = print_evaluation_report(y_test, test_prob_cal, split_name="Test (Calibrated + Final Threshold)", threshold=final_threshold, 
                                           cost_fp=COST_FALSE_POSITIVE, cost_fn=COST_FALSE_NEGATIVE)

    # 4. Curves
    plot_pr_curve(y_test, test_prob_cal, "Precision-Recall Curve (Test, calibrated)", PLOTS_DIR / "pr_curve_test.png")
    plot_cost_curve(y_test, test_prob_cal, PLOTS_DIR / "cost_curve_test.png", cost_fp=COST_FALSE_POSITIVE, cost_fn=COST_FALSE_NEGATIVE) 

    # 5. SHAP Explainability
    print("\n--- SHAP Explainability ---")
    shap_values = compute_shap_values(model, X_test, feature_names, max_samples=1500)

    plot_shap_summary(shap_values, PLOTS_DIR / "shap_summary_plot.png")
    plot_shap_bar(shap_values, PLOTS_DIR / "shap_bar_plot.png")

    top_shap = get_top_features_by_shap(shap_values, top_n=20)
    print("\nTop 20 Features by Mean |SHAP|:")
    print(top_shap.to_string(index=False))
    top_shap.to_csv(REPORTS_DIR / "top_20_features_by_shap.csv", index=False)

    # 6. Save Final Production Artifacts
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # calibrator
    joblib.dump(calibrator, MODELS_DIR / f"calibrator_latest.joblib")

    # Final Decision Config
    decision_config = {
        "timestamp": timestamp,
        "model_path": "xgboost_latest.json",
        "calibrator_path": "calibrator_latest.joblib",
        "final_threshold": float(final_threshold),
        "cost_fp": COST_FALSE_POSITIVE,
        "cost_fn": COST_FALSE_NEGATIVE,
        "test_metrics": test_metrics,
        "calibration_method": "isotonic",
        "feature_names": feature_names,
    }

    decision_config_json = make_json_safe(decision_config)

    with open(MODELS_DIR / "final_decision_config.json", "w") as f:
        json.dump(decision_config_json, f, indent=2)

    with open(REPORTS_DIR / f"phase4_report_{timestamp}.json", "w") as f:
        json.dump(decision_config_json, f, indent=2)

    print("\n" + "=" * 70)
    print("PHASE 4 COMPLETE")
    print("=" * 70)
    print(f"Final threshold          : {final_threshold:.4f}")
    print(f"Calibrator saved         : models/calibrator_latest.joblib")
    print(f"Decision config saved    : models/final_decision_config.json")
    print(f"Plots & SHAP reports     : reports/")
    print("=" * 70)

if __name__ == "__main__":
    main()