"""
Integration tests for the FastAPI application.
Requires model artifacts to be present (run Phases 2–4 first).
"""

import pytest
from fastapi.testclient import TestClient

# Import the app
from api.main import app

client = TestClient(app)


def test_health_endpoint():
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert "status" in data
    assert "model_loaded" in data


def test_root():
    response = client.get("/")
    assert response.status_code == 200
    assert "service" in response.json()


@pytest.fixture
def sample_transaction():
    return {
        "transaction": {
            "Transaction_Amount": 85.0,
            "Order_Quantity": 2,
            "Payment_Method": "Credit Card",
            "Device_Type": "Mobile",
            "Browser": "Chrome",
            "Operating_System": "Android",
            "Product_Category": "Electronics",
            "Customer_Region": "West",
            "Customer_Age": 32,
            "Customer_Tenure_Months": 14,
            "Account_Age_Days": 400,
            "Customer_Order_Count": 8,
            "IP_Risk_Score": 28.0,
            "Velocity_Score": 22.0,
            "Merchant_Risk_Score": 35.0,
            "Previous_Chargebacks": 0,
            "Transactions_Last_24H": 1,
            "Transactions_Last_7D": 4,
            "Failed_Payment_Attempts": 0,
            "Login_Anomalies": 0,
            "Discount_Percentage": 10,
            "Shipping_Billing_Mismatch": 0,
            "High_Risk_Country": 0,
            "New_Device": 0,
            "VPN_Proxy_Used": 0,
            "Is_Weekend": 0,
            "Transaction_Hour": 11,
            "Day_of_Week": 2,
            "Amount_per_Item": 42.5,
        }
    }


def test_predict_endpoint(sample_transaction):
    response = client.post("/predict", json=sample_transaction)
    # 200 if artifacts loaded, 503 if not
    assert response.status_code in (200, 503)
    if response.status_code == 200:
        data = response.json()
        assert "fraud_probability" in data
        assert "is_fraud" in data
        assert "threshold_used" in data
        assert 0.0 <= data["fraud_probability"] <= 1.0