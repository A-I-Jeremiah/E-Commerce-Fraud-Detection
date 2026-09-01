# E-Commerce Fraud Detection System

Production-grade end-to-end machine learning pipeline for predicting and detecting fraudulent transactions.

## Project Status

**Current Phase: Phase 3 – Modeling (XGBoost Primary)** ✅ Completed

Phases 1–3 are complete. The system now has:
- Clean data foundation & EDA
- Feature engineering + preprocessing pipeline
- Trained **XGBoost** primary model + Logistic Regression baseline
- Cost-sensitive threshold selection
- Versioned model artifacts ready for evaluation and serving

**Next:** Phase 4 – Deeper Evaluation, Calibration, SHAP & Threshold Finalisation

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
│   ├── raw/                          # Immutable original CSV
│   ├── processed/                    # Train/val/test matrices + pipeline
│   └── external/
├── notebooks/
│   ├── 01_eda.py                     # Phase 1 EDA + plots
│   └── eda_plots/
├── src/
│   ├── config.py                     # Paths, features, costs, defaults
│   ├── data/
│   │   ├── load.py                   # Loading + time-aware split
│   │   └── features.py               # FeatureEngineer transformer
│   ├── pipeline/
│   │   └── fraud_pipeline.py         # Full sklearn preprocessing pipeline
│   ├── models/
│   │   ├── train.py                  # XGBoost + Logistic training
│   │   └── evaluate.py               # Metrics, cost-sensitive threshold
│   └── utils/
├── models/                           # Versioned model artifacts
│   ├── xgboost_latest.json
│   ├── xgboost_latest_meta.json
│   └── logistic_*.joblib
├── api/                              # FastAPI (Phase 5)
├── configs/
│   └── model_config.yaml
├── scripts/
│   ├── process_data.py               # Phase 2 processing
│   ├── validate_pipeline.py          # Phase 2 validation
│   └── train_model.py                # Phase 3 training
├── tests/
├── docker/
└── requirements.txt
```

---

## Phase Summary

### Phase 1 – Foundations & EDA ✅
- Project skeleton and configuration
- Immutable raw data placement
- Central `src/config.py` (paths, feature groups, business costs)
- Data loader with time-aware chronological split
- Comprehensive EDA (`notebooks/01_eda.py`) with plots
- Business metric decision: **PR-AUC** primary + cost-sensitive operating point

**Key EDA findings:**
- Strong signals: `Transaction_Amount`, `Velocity_Score`, `IP_Risk_Score`, `Login_Anomalies`
- Binary flags with high lift: `Shipping_Billing_Mismatch`, `VPN_Proxy_Used`, `High_Risk_Country`, `New_Device`
- Fraudulent transactions show significantly higher average amounts and risk scores
- Clean data (no missing values / duplicates)

### Phase 2 – Feature Engineering & Preprocessing ✅
- **`src/data/features.py`**: Custom `FeatureEngineer` transformer adding 17 derived features:
  - Log transforms (`Log_Transaction_Amount`, `Log_Amount_per_Item`)
  - Ratios (`Amount_per_Order`, `Txn_Velocity_24H_vs_7D`, `Failed_Payment_Rate`, `Orders_per_Month`)
  - Risk interactions (`IP_x_Velocity`, `IP_x_Merchant`, `NewDevice_x_VPN`, …)
  - Account flags (`Is_New_Account`, `Is_Low_Tenure`)
  - Cyclic time features (`Hour_Sin/Cos`, `Dow_Sin/Cos`)
- **`src/pipeline/fraud_pipeline.py`**: Full sklearn `Pipeline`
  - FeatureEngineer → ColumnTransformer (numeric / binary / categorical)
  - OneHotEncoder with `handle_unknown="ignore"`
  - Fitted only on train data (no leakage)
- Artifacts saved to `data/processed/`:
  - `preprocessing_pipeline.joblib`
  - `X_train/val/test.parquet`, `y_train/val/test.parquet`
  - `meta.joblib`

### Phase 3 – Modeling (XGBoost Primary) ✅

**Primary model:** XGBoost  
**Baseline:** Logistic Regression (`class_weight="balanced"`)

#### Added / Modified Components

| File | Purpose |
|------|---------|
| `src/models/train.py` | `train_xgboost()` with `scale_pos_weight` + early stopping; `train_logistic_baseline()` |
| `src/models/evaluate.py` | PR-AUC, ROC-AUC, F1, Precision, Recall, confusion matrix, **cost-sensitive threshold search** |
| `scripts/train_model.py` | End-to-end training script: load processed data → train both models → evaluate → save artifacts |

#### Training Details
- Imbalance handled via `scale_pos_weight = n_neg / n_pos`
- Early stopping on validation set (monitors `aucpr`)
- Default hyperparameters: `max_depth=6`, `learning_rate=0.05`, `subsample=0.8`, `colsample_bytree=0.8`, etc.
- Cost assumptions: **Cost_FP = 5**, **Cost_FN = 100** (tunable)

#### Evaluation Focus
- **Primary metric:** Average Precision (PR-AUC)
- Secondary: ROC-AUC, Precision, Recall, F1
- Cost-sensitive threshold selected on **validation** only, then applied to test
- Full confusion matrix + expected business cost reported

#### Artifacts Produced (`models/`)
- `xgboost_YYYYMMDD_HHMMSS.json` – versioned model
- `xgboost_latest.json` – latest pointer
- `xgboost_*_meta.json` – metrics, best threshold, feature names, hyperparameters
- `logistic_*.joblib` – baseline model

#### Feature Importance
Gain-based importance is extracted and mapped back to real feature names (top-20 printed during training).

---

## Business Metrics

| Metric | Role |
|--------|------|
| **PR-AUC (Average Precision)** | Primary ranking metric (best for imbalance) |
| ROC-AUC | Secondary |
| Precision / Recall / F1 | Operating-point metrics |
| Expected Cost | `FP × 5 + FN × 100` – used to choose threshold |

Threshold is optimised on the validation set to minimise expected cost, then frozen for test and future production use.

---

## Design Principles

- **No leakage**: time-aware splits + preprocessing fitted only on train
- **Reusable transformers**: FeatureEngineer and the full pipeline can be used at inference time
- **Imbalance-aware**: `scale_pos_weight` (XGBoost) and `class_weight="balanced"` (Logistic)
- **Cost-sensitive**: explicit business costs drive threshold selection
- **Versioned artifacts**: timestamped models + metadata for reproducibility
- **Production-ready formats**: Parquet, joblib, XGBoost JSON

---

## Phase Roadmap

1. **Phase 1** – Foundations & EDA ✅  
2. **Phase 2** – Feature Engineering & Preprocessing Pipeline ✅  
3. **Phase 3** – Modeling (XGBoost primary + Logistic baseline) ✅  
4. **Phase 4** – Evaluation, Calibration, SHAP, Threshold Finalisation  
5. **Phase 5** – Production Packaging & FastAPI Serving  
6. **Phase 6** – Monitoring, Drift Detection, Retraining  
7. **Phase 7** – CI/CD, Tests, Hardening  

---

## How to Reproduce Phase 3

```bash
# 1. Ensure Phase 2 artifacts exist
ls data/processed/

# 2. Train
python scripts/train_model.py

# 3. Inspect results
ls models/
cat models/xgboost_latest_meta.json
```

---

## Next Steps (Phase 4)

- Probability calibration (Platt / Isotonic)
- SHAP explanations for model transparency
- Precision-Recall & cost curves
- Final threshold recommendation and documentation
- Model card
