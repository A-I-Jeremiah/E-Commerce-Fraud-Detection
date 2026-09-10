# E-Commerce Fraud Detection System

Production-grade end-to-end machine learning pipeline for predicting and detecting fraudulent transactions.

## Project Status

**Backend Development Complete** ✅  

**Phases 1–7 finished.** The system is a fully operational, tested, monitored, and production-ready fraud detection backend.

### What the system includes

- Clean data foundation & exploratory analysis
- Feature engineering + preprocessing pipeline
- Trained **XGBoost** primary model + Logistic Regression baseline
- Probability calibration (Isotonic)
- Cost-sensitive final threshold
- SHAP explainability
- FastAPI serving layer (single & batch prediction)
- Docker support
- Prediction logging, data drift detection, performance monitoring & retrain triggers
- Unit & integration tests, CI workflow, hardening checklist, and model card

**Optional next step:** Streamlit operations / analyst console (frontend layer)

---

## Quick Start

```bash
# From project root
pip install -r requirements.txt

# Full pipeline
python notebooks/01_eda.py
python scripts/process_data.py
python scripts/train_model.py
python scripts/evaluate_model.py

# Start API
python scripts/run_api.py
# → http://localhost:8000/docs

# Monitoring
python scripts/monitor.py

# Tests
make test
# or
pytest tests/unit -v
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
│   ├── monitoring/
│   │   ├── logger.py
│   │   ├── drift.py
│   │   ├── performance.py
│   │   └── retrain.py
│   └── utils/
├── models/                             # Versioned artifacts
│   ├── xgboost_latest.json
│   ├── calibrator_latest.joblib
│   ├── decision_config_latest.json
│   └── ...
├── reports/
│   ├── plots/
│   └── monitoring/
├── logs/
│   ├── predictions/
│   └── retrain_triggers/
├── api/                                # FastAPI service
│   ├── main.py
│   ├── schemas.py
│   └── inference.py
├── tests/
│   ├── unit/
│   └── integration/
├── docs/
│   ├── HARDENING.md
│   └── MODEL_CARD.md
├── configs/
├── scripts/
│   ├── process_data.py
│   ├── validate_pipeline.py
│   ├── train_model.py
│   ├── evaluate_model.py
│   ├── run_api.py
│   └── monitor.py
├── docker/
│   ├── Dockerfile
│   └── docker-compose.yml
├── .github/workflows/
│   └── ci.yml
├── Makefile
├── pytest.ini
└── requirements.txt
```

---

## Phase Summaries (Backend Complete)

| Phase | Status | Summary |
|-------|--------|---------|
| **1. Foundations & EDA** | ✅ | Project skeleton, config, time-aware loader, full EDA, business metrics |
| **2. Feature Engineering & Preprocessing** | ✅ | 17 engineered features, sklearn Pipeline, train-only fitting, Parquet artifacts |
| **3. Modeling** | ✅ | XGBoost (primary) + Logistic baseline, cost-sensitive threshold, versioned models |
| **4. Evaluation, Calibration, SHAP** | ✅ | Isotonic calibration, locked threshold, SHAP, decision config |
| **5. Production Packaging & Serving** | ✅ | FastAPI (`/predict`, `/predict/batch`, `/health`), Docker, inference engine |
| **6. Monitoring & Retraining** | ✅ | Prediction logging, PSI/KS drift, performance checks, retrain triggers |
| **7. CI/CD, Tests & Hardening** | ✅ | Unit + integration tests, GitHub Actions CI, Makefile, hardening checklist, model card |

---

## Phase 7 – What Was Added

| Component | Description |
|-----------|-------------|
| `tests/unit/` | Tests for feature engineering, metrics, and drift detection |
| `tests/integration/` | FastAPI health and predict endpoint tests |
| `pytest.ini` | Pytest configuration |
| `.github/workflows/ci.yml` | CI pipeline (install → lint → unit tests) |
| `Makefile` | One-command shortcuts for the entire workflow |
| `docs/HARDENING.md` | Security, operational, and deployment checklist |
| `docs/MODEL_CARD.md` | Model documentation (intended use, limitations, ethics) |

---

## Core Capabilities

### Scoring
- Real-time single and batch prediction
- Calibrated probabilities + final fraud decision
- Identical preprocessing path in training and serving (no train/serve skew)

### Explainability
- SHAP values (instance-level and global)
- Feature importance and contribution analysis

### Monitoring
- Automatic prediction logging
- Data drift detection (PSI + KS)
- Performance degradation checks (when labels are available)
- Rule-based retrain triggers

### Operations
- Health endpoint
- Dockerised deployment
- Versioned model artifacts with easy rollback
- CI tests and hardening guidance

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

- **No leakage** – time-aware splits; all fitting and threshold selection on train/validation only
- **Train/serve consistency** – same preprocessing pipeline at inference
- **Observable** – every prediction logged for drift analysis and retraining
- **Actionable monitoring** – clear drift thresholds and retrain triggers
- **Versioned & reproducible** – explicit model, calibrator, and decision-config artifacts
- **Tested & hardened** – unit/integration tests, CI, security checklist
- **Production-ready API** – validation, health checks, structured logging, Docker

---

## Common Commands (Makefile)

```bash
make install    # Install dependencies
make eda        # Run EDA
make process    # Feature engineering + preprocessing
make train      # Train models
make evaluate   # Calibration, SHAP, final threshold
make api        # Start FastAPI server
make monitor    # Run drift detection + retrain decision
make test       # Run tests
make lint       # Lint check
make clean      # Remove caches
```

---

## API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| `GET` | `/` | Service info |
| `GET` | `/health` | Health check |
| `POST` | `/predict` | Score one transaction |
| `POST` | `/predict/batch` | Score up to 500 transactions |
| `GET` | `/docs` | Interactive Swagger UI |

---

## How to Reproduce the Full Backend

```bash
pip install -r requirements.txt

python notebooks/01_eda.py
python scripts/process_data.py
python scripts/train_model.py
python scripts/evaluate_model.py

python scripts/run_api.py          # API
python scripts/monitor.py          # Monitoring

pytest tests/unit -v               # Tests
```

---

## Optional Next Step – Streamlit Frontend

A Streamlit operations console can be built on top of this backend to provide:

- Interactive single & batch scoring
- SHAP explanations
- Drift and monitoring dashboards
- Threshold / cost exploration lab
- Data explorer and model card views

The backend is complete and stable; any UI layer can consume the existing inference engine or the FastAPI service.

---

## Documentation

- `docs/MODEL_CARD.md` – model details, intended use, limitations
- `docs/HARDENING.md` – security and production readiness checklist
- This README – full system overview

---

**Backend development is complete.**  
The fraud detection system is ready for deployment, monitoring, and optional frontend extension.
