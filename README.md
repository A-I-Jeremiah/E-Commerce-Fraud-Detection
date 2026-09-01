# E-Commerce Fraud Detection System

Production-grade end-to-end machine learning pipeline for predicting and detecting fraudulent transactions.

## Project Status

**Current Phase: Phase 4 – Evaluation, Calibration, SHAP & Threshold Finalisation** ✅ Completed

Phases 1–4 are complete. The system now includes:

- Clean data foundation & exploratory analysis
- Feature engineering + preprocessing pipeline
- Trained **XGBoost** primary model + Logistic Regression baseline
- Probability calibration (Isotonic)
- Cost-sensitive final threshold
- SHAP explainability
- Production decision configuration ready for serving

**Next:** Phase 5 – Production Packaging & FastAPI Serving

---

## Quick Start

```bash
# From project root
pip install -r requirements.txt

# Phase 1 – Exploratory Data Analysis
python notebooks/01_eda.py

# Phase 2 – Feature engineering + preprocessing
python scripts/process_data.py
python scripts/validate_pipeline.py

# Phase 3 – Train models (XGBoost primary + Logistic baseline)
python scripts/train_model.py

# Phase 4 – Calibration, SHAP, curves & final threshold
python scripts/evaluate_model.py
```

---

## Dataset

| Item | Details |
|------|---------|
| Source | `data/raw/ecommerce_fraud_detection.csv` |
| Size | 15,000 transactions |
| Date range | 2022-01-01 → 2025-12-31 |
| Target | `Fraud_Flag` (0 = Legitimate, 1 = Fraudulent) |
| Fraud rate | ~24.9% (moderately imbalanced) |
| Quality | 0 missing values, unique `Transaction_ID`s |
| Split | Time-aware chronological train / val / test |

---

## Project Structure

```
fraud detection/
├── data/
│   ├── raw/                              # Immutable original CSV
│   ├── processed/                        # Train/val/test matrices + pipeline
│   └── external/
├── notebooks/
│   ├── 01_eda.py                         # Phase 1 EDA + plots
│   └── eda_plots/
├── src/
│   ├── config.py                         # Paths, features, costs, defaults
│   ├── data/
│   │   ├── load.py                       # Loading + time-aware split
│   │   └── features.py                   # FeatureEngineer transformer
│   ├── pipeline/
│   │   └── fraud_pipeline.py             # Full sklearn preprocessing pipeline
│   ├── models/
│   │   ├── train.py                      # XGBoost + Logistic training
│   │   ├── evaluate.py                   # Core metrics + cost-sensitive threshold
│   │   ├── calibration.py                # Isotonic / Platt calibrator (Phase 4)
│   │   └── explain.py                    # SHAP utilities (Phase 4)
│   └── utils/
├── models/                               # Versioned model artifacts
│   ├── xgboost_latest.json
│   ├── xgboost_latest_meta.json
│   ├── calibrator_latest.joblib          # Phase 4
│   ├── decision_config_latest.json       # Phase 4 – production decision config
│   └── logistic_*.joblib
├── reports/                              # Phase 4 evaluation outputs
│   ├── plots/
│   │   ├── pr_curve_test.png
│   │   ├── cost_curve_test.png
│   │   ├── shap_summary.png
│   │   └── shap_bar.png
│   ├── shap_top_features.csv
│   └── phase4_report_*.json
├── api/                                  # FastAPI (Phase 5)
├── configs/
│   └── model_config.yaml
├── scripts/
│   ├── process_data.py                   # Phase 2
│   ├── validate_pipeline.py              # Phase 2
│   ├── train_model.py                    # Phase 3
│   └── evaluate_model.py                 # Phase 4
├── tests/
├── docker/
└── requirements.txt
```

---

## Phase Summaries

### Phase 1 – Foundations & EDA ✅
- Project skeleton and central configuration (`src/config.py`)
- Immutable raw data + data loader with time-aware chronological split
- Comprehensive EDA script and plots
- Business metric decision: **PR-AUC** as primary metric + cost-sensitive operating point

**Key findings:**
- Strong signals: `Transaction_Amount`, `Velocity_Score`, `IP_Risk_Score`, `Login_Anomalies`
- High-lift binary flags: `Shipping_Billing_Mismatch`, `VPN_Proxy_Used`, `High_Risk_Country`, `New_Device`
- Clean data (no missing values or duplicates)

### Phase 2 – Feature Engineering & Preprocessing ✅
- `FeatureEngineer` transformer adding 17 derived features (logs, ratios, risk interactions, account flags, cyclic time)
- Full sklearn `Pipeline` (FeatureEngineer → ColumnTransformer with OneHotEncoder)
- Fitted only on train data (no leakage)
- Artifacts: `preprocessing_pipeline.joblib`, `X_*.parquet`, `y_*.parquet`, `meta.joblib`

### Phase 3 – Modeling (XGBoost Primary) ✅
- **Primary model:** XGBoost with `scale_pos_weight` + early stopping (monitors `aucpr`)
- **Baseline:** Logistic Regression (`class_weight="balanced"`)
- Cost-sensitive threshold search on validation
- Versioned artifacts: `xgboost_latest.json` + metadata JSON
- Gain-based feature importance

### Phase 4 – Evaluation, Calibration, SHAP & Threshold Finalisation ✅

#### Added Components

| File | Purpose |
|------|---------|
| `src/models/calibration.py` | `ProbabilityCalibrator` (Isotonic Regression / Platt scaling) + Brier score & log-loss |
| `src/models/explain.py` | SHAP TreeExplainer helpers, summary/bar plots, top-feature ranking |
| `scripts/evaluate_model.py` | End-to-end Phase 4 script |

#### What Phase 4 Produces

1. **Probability Calibration**
   - Isotonic Regression fitted on **validation** predictions only
   - Improves probability quality (lower Brier score / log-loss)

2. **Final Threshold**
   - Cost-sensitive search (`Cost_FP = 5`, `Cost_FN = 100`) on calibrated validation scores
   - Threshold frozen for test evaluation and future production use

3. **Full Evaluation Reports**
   - Validation & Test metrics at the final threshold
   - Confusion matrix + expected business cost

4. **Curves**
   - Precision-Recall curve (test, calibrated)
   - Expected Cost vs Threshold curve

5. **SHAP Explainability**
   - Beeswarm summary plot
   - Mean \|SHAP\| bar plot
   - Ranked feature list (`shap_top_features.csv`)

6. **Production Decision Artifacts**
   - `models/calibrator_latest.joblib`
   - `models/decision_config_latest.json` (model path, calibrator, final threshold, metrics, feature names)
   - `reports/phase4_report_*.json`

---

## Business Metrics & Decision Logic

| Metric | Role |
|--------|------|
| **PR-AUC (Average Precision)** | Primary ranking metric |
| ROC-AUC | Secondary |
| Precision / Recall / F1 | Operating-point metrics |
| Expected Cost | `FP × 5 + FN × 100` – drives threshold selection |
| Brier Score / Log-Loss | Calibration quality |

**Production decision flow:**
1. Raw features → preprocessing pipeline
2. XGBoost → raw probability
3. Isotonic calibrator → calibrated probability
4. Compare to `final_threshold` → Fraud / Legitimate

---

## Design Principles

- **No leakage**: time-aware splits; preprocessing, calibrator and threshold all fitted/selected on train/validation only
- **Reusable components**: FeatureEngineer, full pipeline, calibrator and decision config are inference-ready
- **Imbalance-aware**: `scale_pos_weight` (XGBoost) + cost-sensitive threshold
- **Explainable**: SHAP values available for every prediction
- **Versioned & reproducible**: timestamped models, metadata, and decision config
- **Production formats**: Parquet, joblib, XGBoost JSON

---

## Phase Roadmap

1. **Phase 1** – Foundations & EDA ✅  
2. **Phase 2** – Feature Engineering & Preprocessing Pipeline ✅  
3. **Phase 3** – Modeling (XGBoost primary + Logistic baseline) ✅  
4. **Phase 4** – Evaluation, Calibration, SHAP & Threshold Finalisation ✅  
5. **Phase 5** – Production Packaging & FastAPI Serving  
6. **Phase 6** – Monitoring, Drift Detection, Retraining  
7. **Phase 7** – CI/CD, Tests, Hardening  

---

## How to Reproduce Phases 1–4

```bash
# Phase 1
python notebooks/01_eda.py

# Phase 2
python scripts/process_data.py
python scripts/validate_pipeline.py

# Phase 3
python scripts/train_model.py

# Phase 4
python scripts/evaluate_model.py

# Inspect key outputs
ls models/
cat models/decision_config_latest.json
ls reports/plots/
```

---

## Next Steps (Phase 5)

- Package the full inference pipeline (preprocessing → XGBoost → calibrator → threshold)
- FastAPI service with `/predict` and `/predict/batch` endpoints
- Pydantic request/response schemas
- Docker containerisation
- Health checks and basic logging
