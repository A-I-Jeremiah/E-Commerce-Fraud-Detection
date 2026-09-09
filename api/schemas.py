"""
Request / response schemas for the Fraud Detection API.
"""

from typing import List, Optional, Dict, Any
from pydantic import BaseModel, Field, ConfigDict

class TransactionFeatures(BaseModel):
    """
    Single transaction features.
    All fields that the preprocessing pipeline expects must be present.
    """
    model_config = ConfigDict(extra="forbid")

    # Core Transactional Features
    Transaction_Amount: float = Field(..., example=89.50)
    Order_Quantity: int = Field(..., example=2)
    Payment_Method: str = Field(..., example="Credit Card")
    Device_Type: str = Field(..., example="Mobile")
    Browser: str = Field(..., example="Chrome")
    Operating_System: str = Field(..., example="Android")
    Product_Category:str = Field(..., example="Electronics")
    Customer_Region: str = Field(..., example="West")

    # Customer Information
    Customer_Age: int = Field(..., example=32)
    Customer_Tenure_Months: int = Field(..., example=18)
    Account_Age_Days: int = Field(..., example=540)
    Customer_Order_Count: int = Field(..., example=12)

    # Risk Scores & Behaviour
    IP_Risk_Score: float = Field(..., example=32.5)
    Velocity_Score: float = Field(..., example=28.1)
    Merchant_Risk_Score: float = Field(..., example=41.0)
    Previous_Chargebacks: int = Field(..., example=0)
    Transactions_Last_24H: int = Field(..., example=2)
    Transactions_Last_7D: int = Field(..., example=5)
    Failed_Payment_Attempts: int = Field(..., example=0)
    Login_Anomalies: int = Field(..., example=0)
    Discount_Percentage: int = Field(..., example=10)

    # Binary Flags
    Shipping_Billing_Mismatch: int = Field(..., ge=0, le=1, example=0)
    High_Risk_Country: int = Field(..., ge=0, le=1, example=0)
    New_Device: int = Field(..., ge=0, le=1, example=0)
    VPN_Proxy_Used: int = Field(..., ge=0, le=1, example=0)
    Is_Weekend: int = Field(..., ge=0, le=1, example=0)

    # Time features (already engineered in raw data)
    Transaction_Hour: int = Field(..., ge=0, le=23, example=14)
    Day_of_Week: int = Field(..., ge=0, le=6, example=2)
    Amount_per_Item: float = Field(..., example=44.75)

class PredictionRequest(BaseModel):
    """Single transaction prediction request."""
    transaction: TransactionFeatures


class BatchPredictionRequest(BaseModel):
    """Batch prediction request."""
    transactions: List[TransactionFeatures] = Field(..., min_length=1, max_length=500)


class PredictionResult(BaseModel):
    """Single prediction output."""
    fraud_probability: float = Field(..., description="Calibrated probability of fraud")
    is_fraud: bool = Field(..., description="Final decision using the production threshold")
    threshold_used: float
    model_version: str = "xgboost_latest"


class BatchPredictionResponse(BaseModel):
    """Batch prediction response."""
    predictions: List[PredictionResult]
    count: int


class HealthResponse(BaseModel):
    status: str
    model_loaded: bool
    calibrator_loaded: bool
    threshold: Optional[float] = None