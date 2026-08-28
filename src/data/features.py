"""
Feature engineering utilities for the Fraud Detection project.
All new features are created here so they can be reused in training and serving.
"""

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin


class FeatureEngineer(BaseEstimator, TransformerMixin):
    """
    Custom transformer that adds derived features.
    Compatible with sklearn Pipeline.
    """

    def __init__(self):
        pass

    def fit(self, X, y=None):
        # Stateless for now – can store means/stds later if needed
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        X = X.copy()

        # ------------------------------------------------------------------
        # 1. Log transforms for heavy-tailed amount features
        # ------------------------------------------------------------------
        X["Log_Transaction_Amount"] = np.log1p(X["Transaction_Amount"])
        X["Log_Amount_per_Item"] = np.log1p(X["Amount_per_Item"])
        X["Log_Velocity_Score"] = np.log1p(X["Velocity_Score"])
        X["Log_IP_Risk_Score"] = np.log1p(X["IP_Risk_Score"])
        X["Log_Transactions_Last_24H"] = np.log1p(X["Transactions_Last_24H"])

        # ------------------------------------------------------------------
        # 2. Ratios & velocity-style features
        # ------------------------------------------------------------------
        X["Amount_per_Order"] = X["Transaction_Amount"] / X["Order_Quantity"].clip(lower=1)
        X["Txn_Velocity_24H_vs_7D"] = (
            X["Transactions_Last_24H"] / X["Transactions_Last_7D"].clip(lower=1)
        )
        X["Failed_Payment_Rate"] = (
            X["Failed_Payment_Attempts"] / X["Customer_Order_Count"].clip(lower=1)
        )

        # ------------------------------------------------------------------
        # 3. Risk interactions (high-signal combinations from EDA)
        # ------------------------------------------------------------------
        X["IP_x_Velocity"] = X["IP_Risk_Score"] * X["Velocity_Score"]
        X["IP_x_Merchant"] = X["IP_Risk_Score"] * X["Merchant_Risk_Score"]
        X["NewDevice_x_VPN"] = X["New_Device"] * X["VPN_Proxy_Used"]
        X["Mismatch_x_HighRiskCountry"] = (
            X["Shipping_Billing_Mismatch"] * X["High_Risk_Country"]
        )
        X["Chargeback_x_FailedPayments"] = (
            X["Previous_Chargebacks"] * X["Failed_Payment_Attempts"]
        )

        # ------------------------------------------------------------------
        # 4. Account maturity signals
        # ------------------------------------------------------------------
        X["Is_New_Account"] = (X["Account_Age_Days"] < 30).astype(int)
        X["Is_Low_Tenure"] = (X["Customer_Tenure_Months"] < 3).astype(int)
        X["Orders_per_Month"] = (
            X["Customer_Order_Count"] / X["Customer_Tenure_Months"].clip(lower=1)
        )

        # ------------------------------------------------------------------
        # 5. Time-based cyclic features (optional but useful)
        # ------------------------------------------------------------------
        X["Hour_Sin"] = np.sin(2 * np.pi * X["Transaction_Hour"] / 24)
        X["Hour_Cos"] = np.cos(2 * np.pi * X["Transaction_Hour"] / 24)
        X["Dow_Sin"] = np.sin(2 * np.pi * X["Day_of_Week"] / 7)
        X["Dow_Cos"] = np.cos(2 * np.pi * X["Day_of_Week"] / 7)

        return X


# List of all engineered features (for documentation & monitoring)
ENGINEERED_FEATURES = [
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
    "NewDevice_x_VPN",
    "Mismatch_x_HighRiskCountry",
    "Chargeback_x_FailedPayments",
    "Is_New_Account",
    "Is_Low_Tenure",
    "Orders_per_Month",
    "Hour_Sin",
    "Hour_Cos",
    "Dow_Sin",
    "Dow_Cos",
]