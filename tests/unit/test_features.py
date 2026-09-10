"""Unit tests for feature engineering."""

import numpy as np
import pandas as pd
import pytest

from src.data.features import FeatureEngineer, ENGINEERED_FEATURES


@pytest.fixture
def sample_df():
    return pd.DataFrame({
        "Transaction_Amount": [100.0, 50.0],
        "Order_Quantity": [2, 1],
        "Amount_per_Item": [50.0, 50.0],
        "Transactions_Last_24H": [3, 1],
        "Transactions_Last_7D": [10, 4],
        "Failed_Payment_Attempts": [1, 0],
        "Customer_Order_Count": [5, 2],
        "IP_Risk_Score": [30.0, 10.0],
        "Velocity_Score": [40.0, 15.0],
        "Merchant_Risk_Score": [25.0, 20.0],
        "New_Device": [1, 0],
        "VPN_Proxy_Used": [1, 0],
        "Shipping_Billing_Mismatch": [0, 0],
        "High_Risk_Country": [0, 0],
        "Previous_Chargebacks": [0, 0],
        "Account_Age_Days": [10, 400],
        "Customer_Tenure_Months": [1, 24],
        "Transaction_Hour": [14, 3],
        "Day_of_Week": [2, 6],
    })


def test_feature_engineer_adds_columns(sample_df):
    fe = FeatureEngineer()
    result = fe.fit_transform(sample_df)
    for col in ENGINEERED_FEATURES:
        assert col in result.columns, f"Missing engineered feature: {col}"


def test_log_transforms(sample_df):
    fe = FeatureEngineer()
    result = fe.fit_transform(sample_df)
    assert np.allclose(result["Log_Transaction_Amount"], np.log1p(sample_df["Transaction_Amount"]))


def test_binary_interactions(sample_df):
    fe = FeatureEngineer()
    result = fe.fit_transform(sample_df)
    assert result["NewDevice_x_VPN"].iloc[0] == 1
    assert result["NewDevice_x_VPN"].iloc[1] == 0