# E-Commerce Fraud Detection System

Production-grade end-to-end machine learning pipeline for predicting and detecting fraudulent transactions.

## Project Status

**Current Phase: Phase 2 – Feature Engineering & Preprocessing Pipeline** (completed)

Phase 1 (Foundations & EDA) and Phase 2 are complete. Ready for Phase 3 (Modeling).

## Quick Start

```bash
# From project root
pip install -r requirements.txt

# Phase 1 – Exploratory Data Analysis
python notebooks/01_eda.py

# Phase 2 – Feature engineering + preprocessing (generates train/val/test artifacts)
python scripts/process_data.py

# Validate processed artifacts
python scripts/validate_pipeline.py
```

## Dataset

- **Source**: `data/raw/ecommerce_fraud_detection.csv`
- **Size**: 15,000 transactions (2022-01-01 → 2025-12-31)
- **Target**: `Fraud_Flag` (0 = Legitimate, 1 = Fraudulent)
- **Fraud rate**: ~24.9%
- **No missing values**, unique `Transaction_ID`s
- **Split**: Time-aware chronological train / val / test

## Project Structure

```
fraud detection/
├── data/
│   ├── raw/                      # Immutable original data
│   ├── processed/                # Train/val/test matrices + pipeline artifact
│   └── external/
├── notebooks/
│   ├── 01_eda.py                 # Phase 1 EDA script + plots
│   └── eda_plots/
├── src/
│   ├── config.py                 # Paths, feature lists, defaults, costs
│   ├── data/
│   │   ├── load.py               # Data loading & time-aware split
│   │   └── features.py           # FeatureEngineer transformer (Phase 2)
│   ├── pipeline/
│   │   └── fraud_pipeline.py     # Full sklearn preprocessing pipeline (Phase 2)
│   ├── models/
│   └── utils/
├── models/                       # Versioned model artifacts (Phase 3+)
├── api/                          # FastAPI serving (later phases)
├── configs/
│   └── model_config.yaml
├── scripts/
│   ├── process_data.py           # Run feature engineering + save artifacts
│   └── validate_pipeline.py      # Sanity-check processed data
├── tests/
├── docker/
└── requirements.txt
```

## What Was Added in Phase 2

### 1. Feature Engineering (`src/data/features.py`)
Custom `FeatureEngineer` sklearn transformer that creates:

| Category | New Features |
|----------|--------------|
| Log transforms | `Log_Transaction_Amount`, `Log_Amount_per_Item` |
| Ratios | `Amount_per_Order`, `Txn_Velocity_24H_vs_7D`, `Failed_Payment_Rate`, `Orders_per_Month` |
| Risk interactions | `IP_x_Velocity`, `IP_x_Merchant`, `NewDevice_x_VPN`, `Mismatch_x_HighRiskCountry`, `Chargeback_x_FailedPayments` |
| Account flags | `Is_New_Account`, `Is_Low_Tenure` |
| Cyclic time | `Hour_Sin`, `Hour_Cos`, `Dow_Sin`, `Dow_Cos` |

### 2. Preprocessing Pipeline (`src/pipeline/fraud_pipeline.py`)
- Full `sklearn.Pipeline`: FeatureEngineer → ColumnTransformer
- Handles numeric (impute), binary (impute), categorical (impute + OneHotEncoder)
- `handle_unknown="ignore"` for safe production inference
- Scaling optional (disabled by default for tree-based models)
- Fit only on train set → no leakage

### 3. Processing Scripts
- `scripts/process_data.py` – loads data, applies time-aware split, fits pipeline, transforms all splits, saves artifacts
- `scripts/validate_pipeline.py` – quick integrity check

### 4. Artifacts Produced (in `data/processed/`)
- `preprocessing_pipeline.joblib` – fitted pipeline (reusable at inference)
- `X_train.parquet`, `X_val.parquet`, `X_test.parquet`
- `y_train.parquet`, `y_val.parquet`, `y_test.parquet`
- `meta.joblib` – feature names, row counts, fraud rates

## Business Metrics

- **Primary**: Average Precision (PR-AUC)
- **Secondary**: F1, Precision, Recall, ROC-AUC
- **Operating point**: Cost-sensitive threshold  
  - Assumed Cost_FP = 5, Cost_FN = 100 (refine with real business input)

## Phase Roadmap

1. **Phase 1** – Foundations & EDA ✅
2. **Phase 2** – Feature Engineering & Preprocessing Pipeline ✅
3. **Phase 3** – Modeling (LightGBM / XGBoost baselines + tuning)
4. **Phase 4** – Evaluation, Calibration, Threshold Optimization
5. **Phase 5** – Production Packaging & FastAPI Serving
6. **Phase 6** – Monitoring, Drift Detection, Retraining
7. **Phase 7** – CI/CD, Tests, Hardening

## Key EDA Findings (Phase 1)

- Strong signals: `Transaction_Amount`, `Velocity_Score`, `IP_Risk_Score`, `Login_Anomalies`, binary flags (`Shipping_Billing_Mismatch`, `VPN_Proxy_Used`, `High_Risk_Country`, `New_Device`)
- Fraudulent transactions have significantly higher average amount and risk scores
- Temporal structure present → chronological train/val/test splits used
- Data is clean (0 missing, 0 duplicates)

## Design Principles

- Feature engineering is a reusable `Transformer` (works in training and future serving)
- All preprocessing fitted only on train data
- Time-aware splits to minimize leakage
- Artifacts stored in efficient formats (Parquet + joblib)
- Ready for tree-based models (LightGBM / XGBoost / CatBoost) without mandatory scaling
