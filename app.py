"""
PHASE 7: FASTAPI SERVING APPLICATION
====================================
Production-ready API for fraud detection.

Endpoints:
  POST /score - Batch predictions
  POST /stream - Single prediction (webhook/Kafka)
  GET /health - Health check
  GET /docs - Interactive API documentation
"""

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from typing import List, Dict, Any
import joblib
import numpy as np
import pandas as pd
import logging
import json
from datetime import datetime

# ============================================================================
# SETUP
# ============================================================================

app = FastAPI(
    title="Credit Card Fraud Detection API",
    description="Real-time fraud detection scoring service",
    version="1.0.0"
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load model and preprocessor at startup
logger.info("🚀 Loading models...")
try:
    bundle = joblib.load('models/fraud_model_bundle.joblib')
    MODEL = bundle['model']
    THRESHOLD = bundle['threshold']
    logger.info(f"✅ Model loaded | Threshold: {THRESHOLD:.3f}")
except Exception as e:
    logger.error(f"❌ Failed to load model: {e}")
    MODEL = None
    THRESHOLD = 0.5

try:
    PREPROCESSOR = joblib.load('models/preprocessor.joblib')
    logger.info("✅ Preprocessor loaded")
except Exception as e:
    logger.error(f"❌ Failed to load preprocessor: {e}")
    PREPROCESSOR = None


# ============================================================================
# PYDANTIC MODELS (Request/Response Schemas)
# ============================================================================

class TransactionRequest(BaseModel):
    """Schema for a single transaction."""
    amount: float = Field(..., gt=0, description="Transaction amount (must be > 0)")
    merchant_cat: str = Field(..., description="Merchant category")
    merchant_id_hash: str = Field(..., description="Hashed merchant ID")
    card_id_hash: str = Field(..., description="Hashed card ID")
    city: str = Field(..., description="Transaction city")
    country: str = Field(..., description="Country code (e.g., 'IN', 'US')")
    device_type: str = Field(..., description="Device type (mobile/web/chip/atm)")
    channel: str = Field(..., description="Transaction channel (online/pos/atm)")
    hour: int = Field(..., ge=0, le=23, description="Hour of transaction (0-23)")
    dayofweek: int = Field(..., ge=0, le=6, description="Day of week (0=Mon, 6=Sun)")
    prev_24h_tx_count_card: float = Field(..., ge=0, description="Transactions in last 24h")
    prev_24h_amt_card: float = Field(..., ge=0, description="Amount in last 24h")
    prev_1h_tx_count_card: float = Field(..., ge=0, description="Transactions in last 1h")
    velocity_amt_1h: float = Field(..., ge=0, description="Amount in last 1h")
    is_international: bool = Field(..., description="Is international transaction?")
    is_night: bool = Field(..., description="Is night transaction (10 PM - 6 AM)?")


class ScoringResponse(BaseModel):
    """Schema for scoring response."""
    fraud_probability: float = Field(..., ge=0.0, le=1.0, description="Fraud probability [0, 1]")
    decision: str = Field(..., description="Decision: ALLOW or REVIEW")
    confidence: str = Field(..., description="Confidence level: HIGH, MEDIUM, LOW")
    risk_level: str = Field(..., description="Risk level: LOW, MEDIUM, HIGH, CRITICAL")


class BatchScoringRequest(BaseModel):
    """Schema for batch scoring."""
    transactions: List[TransactionRequest]


class BatchScoringResponse(BaseModel):
    """Schema for batch scoring response."""
    results: List[Dict[str, Any]]
    total_processed: int
    fraud_count: int
    review_count: int


# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def get_risk_level(fraud_prob: float) -> str:
    """Classify risk level based on fraud probability."""
    if fraud_prob >= 0.8:
        return "CRITICAL"
    elif fraud_prob >= 0.6:
        return "HIGH"
    elif fraud_prob >= 0.3:
        return "MEDIUM"
    else:
        return "LOW"


def get_confidence(fraud_prob: float) -> str:
    """Get confidence level based on distance from threshold."""
    distance = abs(fraud_prob - THRESHOLD)
    if distance > 0.2:
        return "HIGH"
    elif distance > 0.1:
        return "MEDIUM"
    else:
        return "LOW"


def score_transaction(tx_dict: Dict[str, Any]) -> Dict[str, Any]:
    """Score a single transaction."""
    try:
        # Create DataFrame
        df = pd.DataFrame([tx_dict])
        
        # Preprocess
        X_scaled = PREPROCESSOR.transform(df)
        
        # Predict
        fraud_prob = float(MODEL.predict_proba(X_scaled)[0, 1])
        decision = "REVIEW" if fraud_prob >= THRESHOLD else "ALLOW"
        
        return {
            'fraud_probability': round(fraud_prob, 4),
            'decision': decision,
            'confidence': get_confidence(fraud_prob),
            'risk_level': get_risk_level(fraud_prob),
            'threshold_used': float(THRESHOLD),
            'distance_from_threshold': round(fraud_prob - THRESHOLD, 4),
        }
    except Exception as e:
        logger.error(f"❌ Scoring error: {e}")
        raise HTTPException(status_code=500, detail=f"Scoring failed: {str(e)}")


# ============================================================================
# ENDPOINTS
# ============================================================================

@app.get("/health", tags=["System"])
async def health_check() -> Dict[str, Any]:
    """
    Health check endpoint.
    
    Returns:
        Status information
    """
    return {
        "status": "healthy",
        "model_loaded": MODEL is not None,
        "threshold": float(THRESHOLD),
        "timestamp": datetime.now().isoformat(),
    }


@app.post("/score", response_model=BatchScoringResponse, tags=["Scoring"])
async def batch_score(request: BatchScoringRequest) -> BatchScoringResponse:
    """
    Score multiple transactions at once (batch endpoint).
    
    Args:
        request: BatchScoringRequest with list of transactions
    
    Returns:
        Predictions for all transactions
    
    Example:
        POST /score
        {
            "transactions": [
                {
                    "amount": 500,
                    "merchant_cat": "grocery",
                    ...
                }
            ]
        }
    """
    if MODEL is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    if not request.transactions:
        raise HTTPException(status_code=400, detail="No transactions provided")
    
    logger.info(f"📊 Processing {len(request.transactions)} transactions...")
    
    results = []
    fraud_count = 0
    review_count = 0
    
    for tx in request.transactions:
        try:
            score_result = score_transaction(tx.model_dump())
            results.append(score_result)
            
            if score_result['decision'] == 'REVIEW':
                review_count += 1
                if score_result['fraud_probability'] > 0.7:
                    fraud_count += 1
        except Exception as e:
            logger.error(f"❌ Error scoring transaction: {e}")
            results.append({'error': str(e)})
    
    logger.info(f"✅ Processed: {len(results)} | REVIEW: {review_count} | FRAUD: {fraud_count}")
    
    return BatchScoringResponse(
        results=results,
        total_processed=len(results),
        fraud_count=fraud_count,
        review_count=review_count,
    )


@app.post("/stream", response_model=ScoringResponse, tags=["Scoring"])
async def stream_score(transaction: TransactionRequest) -> ScoringResponse:
    """
    Score a single transaction (webhook/Kafka consumer endpoint).
    
    Args:
        transaction: Single TransactionRequest
    
    Returns:
        ScoringResponse with prediction
    
    Example:
        POST /stream
        {
            "amount": 5000,
            "merchant_cat": "jewelry",
            ...
        }
    """
    if MODEL is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    logger.info(f"⚡ Streaming prediction for amount: {transaction.amount}")
    
    score_result = score_transaction(transaction.model_dump())
    
    return ScoringResponse(
        fraud_probability=score_result['fraud_probability'],
        decision=score_result['decision'],
        confidence=score_result['confidence'],
        risk_level=score_result['risk_level'],
    )


@app.get("/metrics", tags=["System"])
async def get_metrics() -> Dict[str, Any]:
    """Get current model metrics."""
    return {
        "model": "XGBoost Fraud Detector",
        "threshold": float(THRESHOLD),
        "version": "1.0.0",
        "last_updated": datetime.now().isoformat(),
    }


# ============================================================================
# ERROR HANDLERS
# ============================================================================

@app.exception_handler(ValueError)
async def value_error_handler(request, exc):
    """Handle ValueError exceptions."""
    logger.error(f"❌ Validation error: {exc}")
    return {
        "error": "Validation error",
        "detail": str(exc),
        "timestamp": datetime.now().isoformat(),
    }


# ============================================================================
# STARTUP & SHUTDOWN
# ============================================================================

@app.on_event("startup")
async def startup_event():
    """Log startup information."""
    logger.info("🚀 API Server starting...")
    logger.info(f"   Model: Loaded")
    logger.info(f"   Threshold: {THRESHOLD:.3f}")
    logger.info(f"   API docs available at: /docs")


@app.on_event("shutdown")
async def shutdown_event():
    """Log shutdown information."""
    logger.info("🛑 API Server shutting down...")


# ============================================================================
# TEST DATA
# ============================================================================

if __name__ == "__main__":
    import uvicorn
    
    logger.info("📝 Starting server...")
    logger.info("   API will be available at: http://localhost:8000")
    logger.info("   Docs: http://localhost:8000/docs")
    logger.info("   ReDoc: http://localhost:8000/redoc")
    
    # Run server
    uvicorn.run(app, host="0.0.0.0", port=8000, log_level="info")
