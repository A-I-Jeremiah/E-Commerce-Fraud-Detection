"""
Production preprocessing + feature engineering pipeline.
"""

from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler, FunctionTransformer
from sklearn.impute import SimpleImputer

from src.config import (
    NUMERIC_FEATURES,
    BINARY_FEATURES,
    CATEGORICAL_FEATURES,
)
from src.data.features import FeatureEngineer, ENGINEERED_FEATURES


# ---------------------------------------------------------------------------
# Final feature lists after engineering
# ---------------------------------------------------------------------------
# Numeric features that will be scaled
NUMERIC_FEATURES_FINAL = NUMERIC_FEATURES + [
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

# Binary features (including newly created ones)
BINARY_FEATURES_FINAL = BINARY_FEATURES + [
    "NewDevice_x_VPN",
    "Mismatch_x_HighRiskCountry",
    "Chargeback_x_FailedPayments",
    "Is_New_Account",
    "Is_Low_Tenure",
]

CATEGORICAL_FEATURES_FINAL = CATEGORICAL_FEATURES


def build_preprocessing_pipeline(scale_numeric: bool = True) -> Pipeline:
    """
    Returns a full sklearn Pipeline:
    1. FeatureEngineer (adds derived features)
    2. ColumnTransformer (handles numeric / binary / categorical)
    """

    # Numeric branch
    if scale_numeric:
        numeric_transformer = Pipeline(steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ])
    else:
        numeric_transformer = Pipeline(steps=[
            ("imputer", SimpleImputer(strategy="median")),
        ])

    # Binary branch – just impute (already 0/1)
    binary_transformer = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="most_frequent")),
    ])

    # Categorical branch
    categorical_transformer = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
    ])

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, NUMERIC_FEATURES_FINAL),
            ("bin", binary_transformer, BINARY_FEATURES_FINAL),
            ("cat", categorical_transformer, CATEGORICAL_FEATURES_FINAL),
        ],
        remainder="drop",
        verbose_feature_names_out=False,
    )

    # Full pipeline
    pipe = Pipeline(steps=[
        ("engineer", FeatureEngineer()),
        ("preprocess", preprocessor),
    ])

    return pipe


def get_feature_names(pipeline: Pipeline) -> list[str]:
    """Extract final feature names after OneHot encoding."""
    preprocessor = pipeline.named_steps["preprocess"]
    return list(preprocessor.get_feature_names_out())