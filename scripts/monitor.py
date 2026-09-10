"""
Phase 6 – Run drift detection and (optionally) performance checks.
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

from src.config import PROCESSED_DATA_DIR, MODELS_DIR, PROJECT_ROOT
from src.monitoring.drift import detect_feature_drift, summarize_drift
from src.monitoring.performance import (
    evaluate_live_performance,
    check_performance_degradation,
)
from src.monitoring.retrain import should_retrain

# Features to monitor for drift (high-signal + engineered)
MONITOR_FEATURES = [
    "Transaction_Amount",
    "IP_Risk_Score",
    "Velocity_Score",
    "Merchant_Risk_Score",
    "Previous_Chargebacks",
    "Transactions_Last_24H",
    "Failed_Payment_Attempts",
    "Login_Anomalies",
    "Amount_per_Item",
    "Log_Transaction_Amount",
    "IP_x_Velocity",
    "NewDevice_x_VPN",
]


def load_reference_data():
    """Load training features as the reference distribution."""
    X_train = pd.read_parquet(PROCESSED_DATA_DIR / "X_train.parquet")
    meta = joblib.load(PROCESSED_DATA_DIR / "meta.joblib")
    # Map back to original-style names where possible; for simplicity we monitor
    # the processed matrix columns that exist.
    return X_train, meta.get("feature_names", list(X_train.columns))


def load_recent_predictions(days: int = 7) -> pd.DataFrame:
    """
    Load recent prediction logs (JSONL) and extract feature values.
    In a real system this would come from a feature store or database.
    """
    log_dir = PROJECT_ROOT / "logs" / "predictions"
    if not log_dir.exists():
        return pd.DataFrame()

    records = []
    for path in sorted(log_dir.glob("predictions_*.jsonl"))[-days:]:
        with open(path) as f:
            for line in f:
                rec = json.loads(line)
                feats = rec.get("features", {})
                feats["_prob"] = rec.get("fraud_probability")
                feats["_is_fraud"] = rec.get("is_fraud")
                records.append(feats)

    if not records:
        return pd.DataFrame()
    return pd.DataFrame(records)


def main():
    print("=" * 70)
    print("PHASE 6 – MONITORING & DRIFT DETECTION")
    print("=" * 70)

    # 1. Reference (training) distribution
    ref_df, feature_names = load_reference_data()
    print(f"\nReference data: {len(ref_df):,} rows")

    # 2. Current / recent production data
    current_df = load_recent_predictions(days=14)
    if current_df.empty:
        print("\nNo recent prediction logs found.")
        print("Make some predictions via the API first, then re-run this script.")
        print("Logs are written to logs/predictions/")
        return

    print(f"Recent production data: {len(current_df):,} rows")

    # Align columns (use intersection of available numeric features)
    common = [c for c in MONITOR_FEATURES if c in ref_df.columns and c in current_df.columns]
    if not common:
        # Fallback: use any overlapping numeric columns
        common = list(set(ref_df.select_dtypes(include=np.number).columns) &
                      set(current_df.select_dtypes(include=np.number).columns))
        common = [c for c in common if not c.startswith("_")][:20]

    print(f"Monitoring {len(common)} features for drift...")

    # 3. Drift detection
    drift_df = detect_feature_drift(ref_df, current_df, common, psi_threshold=0.25)
    summary = summarize_drift(drift_df)

    print("\n--- Drift Summary ---")
    print(f"Status          : {summary['status']}")
    print(f"Features checked: {summary['n_features']}")
    print(f"Drifted         : {summary['n_drifted']}")
    print(f"Max PSI         : {summary['max_psi']}")
    if summary["drifted_features"]:
        print(f"Drifted features: {summary['drifted_features']}")

    print("\nTop features by PSI:")
    print(drift_df.head(10).to_string(index=False))

    # Save drift report
    reports_dir = PROJECT_ROOT / "reports" / "monitoring"
    reports_dir.mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    drift_df.to_csv(reports_dir / f"drift_report_{ts}.csv", index=False)
    with open(reports_dir / f"drift_summary_{ts}.json", "w") as f:
        json.dump(summary, f, indent=2)

    # 4. Optional performance check (only if you have labels)
    # For demo we skip real labels; plug them in when available.
    performance_check = None

    # 5. Retrain decision
    decision = should_retrain(summary, performance_check, min_drifted_features=3)
    print("\n--- Retrain Decision ---")
    print(f"Should retrain: {decision['should_retrain']}")
    if decision["reasons"]:
        for r in decision["reasons"]:
            print(f"  • {r}")
    else:
        print("  No retrain needed at this time.")

    print(f"\nReports saved to: {reports_dir}")
    print("Phase 6 monitoring run complete.")


if __name__ == "__main__":
    main()