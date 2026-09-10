"""
Prediction and feature logging for monitoring & future retraining.
"""

import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Any, List, Optional

from src.config import PROJECT_ROOT

logger = logging.getLogger(__name__)

# Default log location
LOG_DIR = PROJECT_ROOT / "logs" / "predictions"
LOG_DIR.mkdir(parents=True, exist_ok=True)


class PredictionLogger:
    """
    Appends each prediction (features + scores + decision) to a daily JSONL file.
    This becomes the source of truth for drift detection and retraining datasets.
    """

    def __init__(self, log_dir: Path = LOG_DIR):
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)

    def _today_file(self) -> Path:
        today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
        return self.log_dir / f"predictions_{today}.jsonl"

    def log(
        self,
        features: Dict[str, Any],
        fraud_probability: float,
        is_fraud: bool,
        threshold: float,
        model_version: str = "xgboost_latest",
        request_id: Optional[str] = None,
        extra: Optional[Dict[str, Any]] = None,
    ):
        """Log a single prediction event."""
        record = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "request_id": request_id,
            "model_version": model_version,
            "threshold": threshold,
            "fraud_probability": fraud_probability,
            "is_fraud": is_fraud,
            "features": features,
        }
        if extra:
            record["extra"] = extra

        path = self._today_file()
        with open(path, "a", encoding="utf-8") as f:
            f.write(json.dumps(record, default=str) + "\n")

    def log_batch(
        self,
        features_list: List[Dict[str, Any]],
        results: List[Dict[str, Any]],
        model_version: str = "xgboost_latest",
    ):
        """Log a batch of predictions."""
        for feats, res in zip(features_list, results):
            self.log(
                features=feats,
                fraud_probability=res["fraud_probability"],
                is_fraud=res["is_fraud"],
                threshold=res["threshold_used"],
                model_version=model_version,
            )


# Global instance
prediction_logger = PredictionLogger()