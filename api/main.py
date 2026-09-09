"""
FastAPI application for Fraud Detection.
"""

import logging
from contextlib import asynccontextmanager
from typing import List

from fastapi import FastAPI, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware

from api.schemas import (
    PredictionRequest,
    BatchPredictionRequest,
    PredictionResult,
    BatchPredictionResponse,
    HealthResponse,
)
from api.inference import engine

# Logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
)
logger = logging.getLogger("fraud-api")


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load model artifacts on startup."""
    try:
        engine.load()
        logger.info("Startup complete – model ready")
    except Exception as e:
        logger.exception("Failed to load artifacts on startup")
        raise
    yield
    logger.info("Shutting down")


app = FastAPI(
    title="E-Commerce Fraud Detection API",
    description="Production API for real-time fraud scoring using XGBoost + calibration",
    version="1.0.0",
    lifespan=lifespan,
)

# CORS (adjust for your frontend domains in production)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health", response_model=HealthResponse, tags=["Monitoring"])
def health():
    """Health check endpoint."""
    return HealthResponse(
        status="ok" if engine.is_ready else "degraded",
        model_loaded=engine.model is not None,
        calibrator_loaded=engine.calibrator is not None,
        threshold=engine.threshold if engine.is_ready else None,
    )


@app.post("/predict", response_model=PredictionResult, tags=["Prediction"])
def predict(request: PredictionRequest):
    """
    Score a single transaction.
    Returns calibrated fraud probability and final decision.
    """
    if not engine.is_ready:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Model not loaded",
        )

    try:
        record = request.transaction.model_dump()
        results = engine.predict([record])
        return PredictionResult(**results[0])
    except Exception as e:
        logger.exception("Prediction failed")
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Prediction error: {str(e)}",
        )


@app.post("/predict/batch", response_model=BatchPredictionResponse, tags=["Prediction"])
def predict_batch(request: BatchPredictionRequest):
    """
    Score a batch of transactions (max 500).
    """
    if not engine.is_ready:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Model not loaded",
        )

    try:
        records = [t.model_dump() for t in request.transactions]
        results = engine.predict(records)
        return BatchPredictionResponse(
            predictions=[PredictionResult(**r) for r in results],
            count=len(results),
        )
    except Exception as e:
        logger.exception("Batch prediction failed")
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Batch prediction error: {str(e)}",
        )


@app.get("/", tags=["Root"])
def root():
    return {
        "service": "E-Commerce Fraud Detection API",
        "version": "1.0.0",
        "docs": "/docs",
        "health": "/health",
    }