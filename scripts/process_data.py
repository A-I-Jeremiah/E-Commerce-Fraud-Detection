"""
Phase 2 – Apply feature engineering + preprocessing and save train/val/test sets.
"""

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

import joblib
import pandas as pd
from src.data.load import load_raw_data, train_test_split_time_aware, get_target
from src.pipeline.fraud_pipeline import build_preprocessing_pipeline, get_feature_names
from src.config import PROCESSED_DATA_DIR, TARGET_COL, RANDOM_SEED

PROCESSED_DATA_DIR.mkdir(parents=True, exist_ok=True)


def main():
    print("=" * 70)
    print("PHASE 2 – FEATURE ENGINEERING & PREPROCESSING")
    print("=" * 70)

    # 1. Load & split
    df = load_raw_data()
    train_df, val_df, test_df = train_test_split_time_aware(df)

    print(f"\nTrain : {train_df.shape[0]:,} rows  | Fraud rate: {train_df[TARGET_COL].mean():.2%}")
    print(f"Val   : {val_df.shape[0]:,} rows  | Fraud rate: {val_df[TARGET_COL].mean():.2%}")
    print(f"Test  : {test_df.shape[0]:,} rows  | Fraud rate: {test_df[TARGET_COL].mean():.2%}")

    # 2. Build pipeline (no scaling needed for tree models)
    pipeline = build_preprocessing_pipeline(scale_numeric=False)

    # 3. Fit on train only
    print("\nFitting preprocessing pipeline on train set...")
    X_train = train_df.drop(columns=[TARGET_COL, "Fraud_Status", "Transaction_ID", "Transaction_Date"], errors="ignore")
    y_train = get_target(train_df)

    pipeline.fit(X_train, y_train)

    # 4. Transform all splits
    print("Transforming train / val / test...")
    X_train_proc = pipeline.transform(X_train)
    X_val_proc = pipeline.transform(
        val_df.drop(columns=[TARGET_COL, "Fraud_Status", "Transaction_ID", "Transaction_Date"], errors="ignore")
    )
    X_test_proc = pipeline.transform(
        test_df.drop(columns=[TARGET_COL, "Fraud_Status", "Transaction_ID", "Transaction_Date"], errors="ignore")
    )

    feature_names = get_feature_names(pipeline)
    print(f"Final feature count: {len(feature_names)}")

    # 5. Save processed arrays + targets + pipeline
    print("\nSaving artifacts...")
    joblib.dump(pipeline, PROCESSED_DATA_DIR / "preprocessing_pipeline.joblib")

    pd.DataFrame(X_train_proc, columns=feature_names).to_parquet(
        PROCESSED_DATA_DIR / "X_train.parquet", index=False
    )
    pd.DataFrame(X_val_proc, columns=feature_names).to_parquet(
        PROCESSED_DATA_DIR / "X_val.parquet", index=False
    )
    pd.DataFrame(X_test_proc, columns=feature_names).to_parquet(
        PROCESSED_DATA_DIR / "X_test.parquet", index=False
    )

    y_train.to_frame(name=TARGET_COL).to_parquet(PROCESSED_DATA_DIR / "y_train.parquet", index=False)
    get_target(val_df).to_frame(name=TARGET_COL).to_parquet(PROCESSED_DATA_DIR / "y_val.parquet", index=False)
    get_target(test_df).to_frame(name=TARGET_COL).to_parquet(PROCESSED_DATA_DIR / "y_test.parquet", index=False)

    # Also save a small metadata file
    meta = {
        "n_features": len(feature_names),
        "feature_names": feature_names,
        "train_rows": len(train_df),
        "val_rows": len(val_df),
        "test_rows": len(test_df),
        "fraud_rate_train": float(y_train.mean()),
    }
    joblib.dump(meta, PROCESSED_DATA_DIR / "meta.joblib")

    print(f"\nArtifacts saved to: {PROCESSED_DATA_DIR}")
    print("Phase 2 complete.")


if __name__ == "__main__":
    main()