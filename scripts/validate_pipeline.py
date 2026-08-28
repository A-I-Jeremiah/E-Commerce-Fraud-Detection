"""Quick sanity check of the preprocessing pipeline."""

import sys
from pathlib import Path
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

import joblib
import pandas as pd
from src.config import PROCESSED_DATA_DIR

def main():
    pipe = joblib.load(PROCESSED_DATA_DIR / "preprocessing_pipeline.joblib")
    meta = joblib.load(PROCESSED_DATA_DIR / "meta.joblib")
    X_train = pd.read_parquet(PROCESSED_DATA_DIR / "X_train.parquet")
    y_train = pd.read_parquet(PROCESSED_DATA_DIR / "y_train.parquet")

    print("Pipeline loaded successfully")
    print(f"Features      : {meta['n_features']}")
    print(f"Train shape   : {X_train.shape}")
    print(f"Fraud rate    : {y_train.iloc[:,0].mean():.2%}")
    print(f"Any NaNs?     : {X_train.isna().sum().sum()}")
    print("Sample feature names:", meta["feature_names"][:8], "...")
    print("Validation OK")

if __name__ == "__main__":
    main()