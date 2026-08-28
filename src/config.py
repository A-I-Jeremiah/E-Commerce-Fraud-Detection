"""
Central configuration for the Fraud Detection Project.
All default hyperparameters, paths, constants live here.
"""

from pathlib import Path

from src.data.features import ENGINEERED_FEATURES

# ----------------------- Project Paths --------------------
PROJECT_ROOT = Path(__file__).resolve().parents[1]

DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
EXTERNAL_DATA_DIR = DATA_DIR / "external"

MODELS_DIR = PROJECT_ROOT / "models"
NOTEBOOKS_DIR = PROJECT_ROOT / "notebooks"
CONFIGS_DIR = PROJECT_ROOT / "configs"
SCRIPTS_DIR = PROJECT_ROOT / "scripts"
API_DIR = PROJECT_ROOT / "api"

# Primary Dataset
RAW_DATA_PATH = RAW_DATA_DIR / "ecommerce_fraud_detection.csv"

# -------------------- Target IDs and Columns --------------------
TARGET_COL = "Fraud_Flag"
TARGET_LABEL_COL = "Fraud_Status"
ID_COL = "Transaction_ID"
DATETIME_COL = "Transaction_Date"

# Columns that must never leak into the model
LEAKAGE_COLS = [TARGET_COL, TARGET_LABEL_COL]

# Feature Grougps (Useful for processing and monitoring)
NUMERIC_FEATURES = [
    "Transaction_Amount",
    "Order_Quantity",
    "Customer_Age",
    "Customer_Tenure_Months",
    "Account_Age_Days",
    "Customer_Order_Count",
    "IP_Risk_Score",
    "Velocity_Score",
    "Merchant_Risk_Score",
    "Previous_Chargebacks",
    "Transactions_Last_24H",
    "Transactions_Last_7D",
    "Failed_Payment_Attempts",
    "Login_Anomalies",
    "Discount_Percentage",
    "Transaction_Hour",
    "Day_of_Week",
    "Amount_per_Item",
]

BINARY_FEATURES = [
    "Shipping_Billing_Mismatch",
    "High_Risk_Country",
    "New_Device",
    "VPN_Proxy_Used",
    "Is_Weekend",
]

CATEGORICAL_FEATURES = [
    "Payment_Method",
    "Device_Type",
    "Browser",
    "Operating_System",
    "Product_Category",
    "Customer_Region",
]

# ---------------------------------------------------------------------------
# Engineered features (populated by FeatureEngineer)
# ---------------------------------------------------------------------------
ENGINEERED_NUMERIC = [
    "Log_Transaction_Amount",
    "Log_Amount_per_Item",
    "Log_Velocity_Score",
    "Log_IP_Risk_Score",
    "Log_Transactions_Last_24H",
    "Amount_per_Order",
    "Txn_Velocity_24H_vs_7D",
    "Failed_Payment_Rate",
    "IP_x_Velocity",
    "IP_x_Merchant",
    "Orders_per_Month",
    "Hour_Sin",
    "Hour_Cos",
    "Dow_Sin",
    "Dow_Cos",
]

ENGINEERED_BINARY = [
    "NewDevice_x_VPN",
    "Mismatch_x_HighRiskCountry",
    "Chargeback_x_FailedPayments",
    "Is_New_Account",
    "Is_Low_Tenure",
]

# Features which will be used by the model
# All features that will be used by the model
FEATURE_COLS = NUMERIC_FEATURES + BINARY_FEATURES + CATEGORICAL_FEATURES + ENGINEERED_FEATURES + ENGINEERED_BINARY

# ---------------------------------------------------------------------------
# Business / Evaluation defaults
# ---------------------------------------------------------------------------
# Cost assumptions (can be overridden later)
COST_FALSE_POSITIVE = 5.0      # cost of reviewing / blocking a legitimate tx
COST_FALSE_NEGATIVE = 100.0    # average loss when fraud is missed

RANDOM_SEED = 42
TEST_SIZE = 0.15
VAL_SIZE = 0.15                # of the remaining train portion

# ---------------------------------------------------------------------------
# Model defaults (will be refined in later phases)
# ---------------------------------------------------------------------------
DEFAULT_MODEL_PARAMS = {
    "xgboost": {
        "n_estimators":300,
        "random_state":42,
        "objective":"binary:logistic",
        "eval_metric":"aucpr",
        "max_depth":4,
        "learning_rate":0.05,
        "subsample":0.85,
        "colsample_bytree":0.85,
        "tree_method":"hist",
        "n_jobs":1,
        "random_state": RANDOM_SEED
    }
}