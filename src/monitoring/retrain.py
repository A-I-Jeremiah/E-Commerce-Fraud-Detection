"""
Simple retraining decision logic.
"""

from typing import Dict, Any
from datetime import datetime, timezone
import json
from pathlib import Path

from src.config import PROJECT_ROOT

RETRAIN_DIR = PROJECT_ROOT / "logs" / "retrain_triggers"
RETRAIN_DIR.mkdir(parents=True, exist_ok=True)


def should_retrain(
    drift_summary: Dict[str, Any],
    performance_check: Dict[str, Any] | None = None,
    min_drifted_features: int = 3,
) -> Dict[str, Any]:
    """
    Decide whether a retrain should be triggered.
    Rules (can be made more sophisticated later):
      - Significant drift on >= min_drifted_features
      - OR clear performance degradation
    """
    reasons = []

    if drift_summary.get("n_drifted", 0) >= min_drifted_features:
        reasons.append(
            f"Data drift detected on {drift_summary['n_drifted']} features "
            f"(max PSI={drift_summary.get('max_psi')})"
        )

    if performance_check and performance_check.get("degraded"):
        reasons.append(performance_check["message"])

    trigger = len(reasons) > 0

    decision = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "should_retrain": trigger,
        "reasons": reasons,
        "drift_summary": drift_summary,
        "performance_check": performance_check,
    }

    # Persist the decision
    fname = RETRAIN_DIR / f"trigger_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    with open(fname, "w") as f:
        json.dump(decision, f, indent=2)

    return decision