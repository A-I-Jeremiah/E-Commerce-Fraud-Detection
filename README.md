# E-Commerce Fraud Detection System

Production-grade end-to-end machine learning pipeline for predicting and detecting fraudulent transactions.

## Project Status

**Current Phase: Phase 5 – Production Packaging & FastAPI Serving** ✅ Completed

Phases 1–5 are complete. The system is now a fully operational production service:

- Clean data foundation & exploratory analysis
- Feature engineering + preprocessing pipeline
- Trained **XGBoost** primary model + Logistic Regression baseline
- Probability calibration (Isotonic)
- Cost-sensitive final threshold
- SHAP explainability
- **FastAPI serving layer** with single & batch prediction endpoints
- Docker support for deployment

**Next:** Phase 6 – Monitoring, Drift Detection & Retraining

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

# Phase 5 – Start the API (local development)
python scripts/run_api.py
# Then open http://localhost:8000/docs
```

### Docker (optional)

```bash
cd docker
docker compose up --build
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
│   │   ├── calibration.py                # Isotonic / Platt calibrator
│   │   └── explain.py                    # SHAP utilities
│   └── utils/
├── models/                               # Versioned model artifacts
│   ├── xgboost_latest.json
│   ├── xgboost_latest_meta.json
│   ├── calibrator_latest.joblib
│   ├── decision_config_latest.json       # Final threshold + metadata
│   └── logistic_*.joblib
├── reports/                              # Evaluation outputs (Phase 4)
│   ├── plots/
│   └── shap_top_features.csv
├── api/                                  # Phase 5 – FastAPI service
│   ├── __init__.py
│   ├── main.py                           # FastAPI app + endpoints
│   ├── schemas.py                        # Pydantic request/response models
│   └── inference.py                      # Production inference engine
├── configs/
│   └── model_config.yaml
├── scripts/
│   ├── process_data.py                   # Phase 2
│   ├── validate_pipeline.py              # Phase 2
│   ├── train_model.py                    # Phase 3
│   ├── evaluate_model.py                 # Phase 4
│   └── run_api.py                        # Phase 5 – local server
├── docker/
│   ├── Dockerfile
│   └── docker-compose.yml
├── tests/
└── requirements.txt
```

---

## Phase Summaries

### Phase 1 – Foundations & EDA ✅
- Project skeleton and central configuration
- Immutable raw data + time-aware data loader
- Comprehensive EDA and business metric decision (PR-AUC + cost-sensitive threshold)

### Phase 2 – Feature Engineering & Preprocessing ✅
- 17 engineered features (logs, ratios, risk interactions, account flags, cyclic time)
- Full sklearn Pipeline (FeatureEngineer → ColumnTransformer)
- Artifacts: `preprocessing_pipeline.joblib`, Parquet matrices, metadata

### Phase 3 – Modeling (XGBoost Primary) ✅
- XGBoost with `scale_pos_weight` + early stopping
- Logistic Regression baseline
- Cost-sensitive threshold search on validation
- Versioned model artifacts (`xgboost_latest.json` + metadata)

### Phase 4 – Evaluation, Calibration, SHAP & Threshold Finalisation ✅
- Isotonic probability calibration
- Final cost-sensitive threshold locked
- SHAP explainability (summary + bar plots)
- Production decision config (`decision_config_latest.json`)
- PR curve, cost curve, and evaluation reports

### Phase 5 – Production Packaging & FastAPI Serving ✅

#### Added Components

| File | Purpose |
|------|---------|
| `api/schemas.py` | Pydantic models for strict input validation and response typing |
| `api/inference.py` | `FraudInferenceEngine` – loads pipeline + model + calibrator + threshold |
| `api/main.py` | FastAPI application with lifespan loading, CORS, and endpoints |
| `scripts/run_api.py` | Local development server (uvicorn + hot reload) |
| `docker/Dockerfile` | Production container image |
| `docker/docker-compose.yml` | Easy local/prod orchestration with health checks |

#### API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| `GET` | `/` | Service info |
| `GET` | `/health` | Health check (model loaded, calibrator loaded, current threshold) |
| `POST` | `/predict` | Score a single transaction |
| `POST` | `/predict/batch` | Score up to 500 transactions |
| `GET` | `/docs` | Interactive Swagger UI |

#### Inference Flow (Production)

1. Raw transaction features (validated by Pydantic)
2. Preprocessing pipeline (same as training – no train/serve skew)
3. XGBoost → raw probability
4. Isotonic calibrator → calibrated probability
5. Compare to final threshold → `is_fraud` decision

#### Key Design Decisions
- All artifacts loaded **once** at startup via FastAPI lifespan
- Input strictly validated before any computation
- Same preprocessing pipeline used in training and serving
- Threshold and calibrator come from Phase 4 decision config
- Docker support with health checks and optional volume mounts for model updates

---

## Business Metrics & Decision Logic

| Metric | Role |
|--------|------|
| **PR-AUC** | Primary ranking metric |
| ROC-AUC | Secondary |
| Precision / Recall / F1 | Operating-point metrics |
| Expected Cost | `FP × 5 + FN × 100` – drives threshold selection |
| Brier Score / Log-Loss | Calibration quality |

**Production decision:**  
`calibrated_probability >= final_threshold` → Fraud

---

## Design Principles

- **No leakage**: time-aware splits; preprocessing, calibrator and threshold fitted/selected only on train/validation
- **Train/serve consistency**: identical preprocessing pipeline at inference
- **Reusable & versioned artifacts**: models, calibrator and decision config are explicitly versioned
- **Explainable**: SHAP available for offline analysis
- **Production-ready API**: validation, health checks, structured logging, Docker
- **Imbalance-aware**: `scale_pos_weight` + cost-sensitive threshold

---

## Phase Roadmap

1. **Phase 1** – Foundations & EDA ✅  
2. **Phase 2** – Feature Engineering & Preprocessing Pipeline ✅  
3. **Phase 3** – Modeling (XGBoost primary + Logistic baseline) ✅  
4. **Phase 4** – Evaluation, Calibration, SHAP & Threshold Finalisation ✅  
5. **Phase 5** – Production Packaging & FastAPI Serving ✅  
6. **Phase 6** – Monitoring, Drift Detection, Retraining  
7. **Phase 7** – CI/CD, Tests, Hardening  

---

## How to Run the Full System

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Reproduce training pipeline (if needed)
python scripts/process_data.py
python scripts/train_model.py
python scripts/evaluate_model.py

# 3. Start the API
python scripts/run_api.py

# 4. Test
curl http://localhost:8000/health
# Open http://localhost:8000/docs for interactive testing
```

---

## Example Prediction Request

```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{
    "transaction": {
      "Transaction_Amount": 120.0,
      "Order_Quantity": 1,
      "Payment_Method": "Credit Card",
      "Device_Type": "Mobile",
      "Browser": "Chrome",
      "Operating_System": "Android",
      "Product_Category": "Electronics",
      "Customer_Region": "West",
      "Customer_Age": 29,
      "Customer_Tenure_Months": 8,
      "Account_Age_Days": 120,
      "Customer_Order_Count": 3,
      "IP_Risk_Score": 55.0,
      "Velocity_Score": 62.0,
      "Merchant_Risk_Score": 48.0,
      "Previous_Chargebacks": 1,
      "Transactions_Last_24H": 4,
      "Transactions_Last_7D": 9,
      "Failed_Payment_Attempts": 2,
      "Login_Anomalies": 1,
      "Discount_Percentage": 15,
      "Shipping_Billing_Mismatch": 1,
      "High_Risk_Country": 0,
      "New_Device": 1,
      "VPN_Proxy_Used": 1,
      "Is_Weekend": 0,
      "Transaction_Hour": 2,
      "Day_of_Week": 5,
      "Amount_per_Item": 120.0
    }
  }'
```

---

## Next Steps (Phase 6)

- Prediction & feature logging
- Data drift detection (PSI / KS on key features)
- Concept drift / performance monitoring
- Automated retraining triggers and approval gates
- Basic alerting
