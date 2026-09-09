"""
Inference engine: loads pipeline, model, calibrator and applies the final threshold.
"""

from pathlib import Path
from typing import List, Dict, Any
import json
import logging

import joblib
import numpy as np
import pandas as pd
import xgboost as xgb

from src.config import MODELS_DIR, PROCESSED_DATA_DIR

logger = logging.getLogger(__name__)


class FraudInferenceEngine:
    """
    Production inference wrapper.
    Loads:
      - preprocessing pipeline (FeatureEngineer + ColumnTransformer)
      - XGBoost model
      - Isotonic calibrator
      - final decision threshold
    """

    def __init__(self):
        self.pipeline = None
        self.model = None
        self.calibrator = None
        self.threshold = 0.5
        self.feature_names = None
        self.is_ready = False

    def load(self):
        """Load all artifacts. Call once at startup."""
        logger.info("Loading production artifacts...")

        # 1. Preprocessing pipeline
        pipeline_path = PROCESSED_DATA_DIR / "preprocessing_pipeline.joblib"
        if not pipeline_path.exists():
            raise FileNotFoundError(f"Pipeline not found: {pipeline_path}")
        self.pipeline = joblib.load(pipeline_path)

        # 2. XGBoost model
        model_path = MODELS_DIR / "xgboost_latest.json"
        if not model_path.exists():
            raise FileNotFoundError(f"Model not found: {model_path}")
        self.model = xgb.Booster()
        self.model.load_model(str(model_path))

        # 3. Calibrator
        calibrator_path = MODELS_DIR / "calibrator_latest.joblib"
        if not calibrator_path.exists():
            raise FileNotFoundError(f"Calibrator not found: {calibrator_path}")
        self.calibrator = joblib.load(calibrator_path)

        # 4. Decision config (threshold + metadata)
        config_path = MODELS_DIR / "final_decision_config.json"
        if not config_path.exists():
            raise FileNotFoundError(f"Decision config not found: {config_path}")
        with open(config_path) as f:
            config = json.load(f)

        self.threshold = float(config["final_threshold"])
        self.feature_names = config.get("feature_names")

        self.is_ready = True
        logger.info(
            f"Inference engine ready | threshold={self.threshold:.4f}"
        )

    def _to_dataframe(self, records: List[Dict[str, Any]]) -> pd.DataFrame:
        """Convert list of feature dicts to DataFrame."""
        return pd.DataFrame(records)

    def predict(self, records: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Run full inference pipeline on a list of raw feature dictionaries.
        Returns list of dicts with probability, decision and threshold.
        """
        if not self.is_ready:
            raise RuntimeError("Inference engine not loaded. Call load() first.")

        df = self._to_dataframe(records)

        # Preprocess
        X = self.pipeline.transform(df)

        # XGBoost prediction
        dmatrix = xgb.DMatrix(X)
        raw_prob = self.model.predict(dmatrix)

        # Calibrate
        cal_prob = self.calibrator.transform(raw_prob)

        # Decision
        results = []
        for p in cal_prob:
            results.append({
                "fraud_probability": float(round(p, 6)),
                "is_fraud": bool(p >= self.threshold),
                "threshold_used": self.threshold,
                "model_version": "xgboost_latest",
            })
        return results


# Global singleton (loaded at startup)
engine = FraudInferenceEngine()