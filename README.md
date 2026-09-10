# E-Commerce Fraud Detection System

Production-grade end-to-end machine learning pipeline for predicting and detecting fraudulent transactions.

## Project Status

**Current Phase: Phase 6 – Monitoring, Drift Detection & Retraining** ✅ Completed

Phases 1–6 are complete. The system is a fully operational, monitored production service:

- Clean data foundation & exploratory analysis
- Feature engineering + preprocessing pipeline
- Trained **XGBoost** primary model + Logistic Regression baseline
- Probability calibration (Isotonic)
- Cost-sensitive final threshold
- SHAP explainability
- FastAPI serving layer (single & batch prediction)
- Docker support
- **Prediction logging, data drift detection, performance monitoring & retrain triggers**

**Next:** Phase 7 – CI/CD, Tests & Hardening

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

# Phase 3 – Train models
python scripts/train_model.py

# Phase 4 – Calibration, SHAP & final threshold
python scripts/evaluate_model.py

# Phase 5 – Start the API
python scripts/run_api.py
# → http://localhost:8000/docs

# Phase 6 – Run monitoring / drift detection
python scripts/monitor.py
```

### Docker

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
│   ├── raw/
│   ├── processed/
│   └── external/
├── notebooks/
│   ├── 01_eda.py
│   └── eda_plots/
├── src/
│   ├── config.py
│   ├── data/
│   │   ├── load.py
│   │   └── features.py
│   ├── pipeline/
│   │   └── fraud_pipeline.py
│   ├── models/
│   │   ├── train.py
│   │   ├── evaluate.py
│   │   ├── calibration.py
│   │   └── explain.py
│   ├── monitoring/                     # Phase 6
│   │   ├── __init__.py
│   │   ├── logger.py                   # Prediction / feature logging
│   │   ├── drift.py                    # PSI + KS drift detection
│   │   ├── performance.py              # Live performance metrics
│   │   └── retrain.py                  # Retrain trigger logic
│   └── utils/
├── models/
│   ├── xgboost_latest.json
│   ├── calibrator_latest.joblib
│   ├── decision_config_latest.json
│   └── ...
├── reports/
│   ├── plots/                          # PR, cost, SHAP plots
│   └── monitoring/                     # Drift reports (Phase 6)
├── logs/                               # Phase 6
│   ├── predictions/                    # Daily JSONL prediction logs
│   └── retrain_triggers/               # Retrain decision records
├── api/
│   ├── main.py
│   ├── schemas.py
│   └── inference.py                    # Integrated with prediction logger
├── configs/
├── scripts/
│   ├── process_data.py
│   ├── validate_pipeline.py
│   ├── train_model.py
│   ├── evaluate_model.py
│   ├── run_api.py
│   └── monitor.py                      # Phase 6 monitoring job
├── docker/
│   ├── Dockerfile
│   └── docker-compose.yml
├── tests/
└── requirements.txt
```

---

## Phase Summaries

### Phase 1 – Foundations & EDA ✅
- Project skeleton, central config, time-aware data loader
- Full EDA and business metric decision (PR-AUC + cost-sensitive threshold)

### Phase 2 – Feature Engineering & Preprocessing ✅
- 17 engineered features + full sklearn Pipeline
- Train-only fitting, Parquet artifacts, reusable preprocessing pipeline

### Phase 3 – Modeling (XGBoost Primary) ✅
- XGBoost with `scale_pos_weight` + early stopping
- Logistic baseline, cost-sensitive threshold, versioned model artifacts

### Phase 4 – Evaluation, Calibration, SHAP & Threshold Finalisation ✅
- Isotonic calibration, final locked threshold, SHAP explainability
- Production decision config (`decision_config_latest.json`)

### Phase 5 – Production Packaging & FastAPI Serving ✅
- FastAPI app (`/predict`, `/predict/batch`, `/health`)
- Pydantic validation, inference engine, Docker support
- Startup loading of pipeline + model + calibrator + threshold

### Phase 6 – Monitoring, Drift Detection & Retraining ✅

#### Added Components

| File | Purpose |
|------|---------|
| `src/monitoring/logger.py` | Daily JSONL logging of every prediction + features |
| `src/monitoring/drift.py` | Population Stability Index (PSI) + Kolmogorov-Smirnov tests |
| `src/monitoring/performance.py` | Live performance metrics + degradation checks (when labels exist) |
| `src/monitoring/retrain.py` | Rule-based retrain trigger and decision logging |
| `scripts/monitor.py` | One-command monitoring job |

#### Key Capabilities

- **Prediction logging** – every API call is recorded under `logs/predictions/`
- **Data drift detection** – PSI & KS on high-signal features vs training distribution
- **Drift severity** – PSI < 0.1 (stable), 0.1–0.25 (moderate), > 0.25 (significant)
- **Performance monitoring** – ready for labelled production windows (PR-AUC drop detection)
- **Retrain triggers** – automatic decision when drift or performance degradation is detected; decisions stored in `logs/retrain_triggers/`
- **API integration** – `FraudInferenceEngine` automatically logs predictions

#### Typical Monitoring Workflow

1. API serves traffic → predictions logged automatically
2. Run `python scripts/monitor.py` periodically (or via cron/scheduler)
3. Review drift report in `reports/monitoring/`
4. If `should_retrain = true`, launch a new training cycle (Phase 3 → 4) and promote new `*_latest` artifacts
5. Restart API (or rely on volume mounts) to pick up the new model

---

## Business Metrics & Decision Logic

| Metric | Role |
|--------|------|
| **PR-AUC** | Primary ranking metric |
| ROC-AUC | Secondary |
| Precision / Recall / F1 | Operating-point metrics |
| Expected Cost | `FP × 5 + FN × 100` – drives threshold selection |
| Brier Score / Log-Loss | Calibration quality |
| PSI / KS | Data drift |
| PR-AUC drop | Concept / performance drift |

**Production decision flow:**  
Raw features → Preprocessing → XGBoost → Isotonic calibration → `probability >= final_threshold` → Fraud / Legitimate

---

## Design Principles

- **No leakage** – time-aware splits; all fitting/selection on train/validation only
- **Train/serve consistency** – identical preprocessing pipeline at inference
- **Observable** – every prediction is logged for drift & retraining
- **Actionable monitoring** – clear drift thresholds and retrain triggers
- **Versioned artifacts** – models, calibrator and decision config are explicit
- **Production-ready** – validation, health checks, Docker, structured logging

---

## Phase Roadmap

1. **Phase 1** – Foundations & EDA ✅  
2. **Phase 2** – Feature Engineering & Preprocessing Pipeline ✅  
3. **Phase 3** – Modeling (XGBoost primary + Logistic baseline) ✅  
4. **Phase 4** – Evaluation, Calibration, SHAP & Threshold Finalisation ✅  
5. **Phase 5** – Production Packaging & FastAPI Serving ✅  
6. **Phase 6** – Monitoring, Drift Detection & Retraining ✅  
7. **Phase 7** – CI/CD, Tests & Hardening  

---

## How to Run Monitoring

```bash
# Ensure the API has served some traffic (predictions are logged automatically)

# Run drift detection + retrain decision
python scripts/monitor.py

# Inspect outputs
ls logs/predictions/
ls logs/retrain_triggers/
ls reports/monitoring/
```

---

## Next Steps (Phase 7)

- Unit & integration tests
- CI pipeline (lint, tests, model regression checks)
- Pre-commit hooks / code quality
- Security hardening (input limits, rate limiting, secrets)
- Final documentation & model card
- Deployment checklist
