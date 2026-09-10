# Model Card – E-Commerce Fraud Detection

## Model Details
- **Architecture**: XGBoost (binary:logistic)
- **Calibration**: Isotonic Regression
- **Threshold**: Cost-sensitive (Cost_FP=5, Cost_FN=100)
- **Versioning**: `xgboost_latest.json` + `decision_config_latest.json`

## Intended Use
- Real-time scoring of e-commerce transactions
- Output: calibrated fraud probability + binary decision

## Training Data
- 15,000 transactions (2022–2025)
- ~24.9% fraud rate
- Time-aware train/val/test split

## Evaluation
- Primary metric: PR-AUC
- Secondary: ROC-AUC, Precision, Recall, F1, Expected Cost
- Threshold selected on validation, evaluated on held-out test set

## Limitations
- Performance depends on feature quality and stability
- Concept drift possible if fraud patterns change significantly
- Does not replace human review for high-value or ambiguous cases

## Ethical Considerations
- False positives create customer friction
- False negatives create financial loss
- Threshold chosen to balance these costs; should be reviewed with business stakeholders