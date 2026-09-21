"""
Intent Classifier FastAPI Serving Application
Author: Manjunath
Project: Intent Classifier MLOps Production Pipeline
"""

from contextlib import asynccontextmanager
from typing import Any, Dict, List, Optional
from fastapi import FastAPI, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from model.intent_model import IntentModel

# Model instance holder
model_instance: Optional[IntentModel] = None


def get_model() -> Optional[IntentModel]:
    """Retrieve or lazily initialize the IntentModel instance."""
    global model_instance
    if model_instance is None:
        try:
            model_instance = IntentModel()
        except Exception:
            pass
    return model_instance


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Lifecycle manager to load model on startup."""
    model = get_model()
    if model:
        print("[INFO] IntentModel successfully loaded into memory.")
    else:
        print("[WARNING] Model not found on startup. Train first using python model/train.py")
    yield
    print("[INFO] Shutting down Intent Classifier API.")


app = FastAPI(
    title="Intent Classifier API - by Manjunath",
    description=(
        "Production-grade Intent Classification Service built with Scikit-learn, "
        "MLflow experiment tracking, and FastAPI serving. Created by Manjunath."
    ),
    version="1.0.0",
    contact={
        "name": "Manjunath",
    },
    lifespan=lifespan,
)

# Enable CORS for web applications and dashboards
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# --- Pydantic Schemas ---
class PredictionRequest(BaseModel):
    text: str = Field(
        ...,
        min_length=1,
        max_length=1000,
        description="Text message or customer query to classify",
        examples=["I want to cancel my subscription"],
    )
    confidence_threshold: float = Field(
        default=0.20,
        ge=0.0,
        le=1.0,
        description="Minimum confidence threshold. Queries below this will be flagged as fallback.",
        examples=[0.20],
    )


class PredictionResponse(BaseModel):
    text: str
    intent: str
    confidence: float
    probabilities: Dict[str, float]
    is_fallback: bool


class BatchPredictionRequest(BaseModel):
    texts: List[str] = Field(
        ...,
        min_length=1,
        max_length=100,
        description="List of text messages to classify",
        examples=[["Hello support", "Where is my order?", "I want a refund"]],
    )
    confidence_threshold: float = Field(
        default=0.20,
        ge=0.0,
        le=1.0,
        description="Minimum confidence threshold for classification.",
    )


class BatchPredictionResponse(BaseModel):
    count: int
    predictions: List[PredictionResponse]


class ModelInfoResponse(BaseModel):
    service_name: str
    author: str
    version: str
    num_classes: int
    classes: List[str]
    model_loaded: bool


class HealthResponse(BaseModel):
    status: str
    model_loaded: bool
    author: str


# --- API Endpoints ---
@app.get("/", tags=["General"])
def root():
    """Welcome endpoint with project metadata and documentation link."""
    return {
        "project": "Intent Classifier MLOps Pipeline",
        "author": "Manjunath",
        "version": "1.0.0",
        "docs_url": "/docs",
        "redoc_url": "/redoc",
        "health_check": "/health",
    }


@app.get("/health", response_model=HealthResponse, tags=["Monitoring"])
def health():
    """Readiness and liveness probe for Kubernetes / monitoring."""
    model = get_model()
    is_loaded = model is not None
    return {
        "status": "healthy" if is_loaded else "degraded",
        "model_loaded": is_loaded,
        "author": "Manjunath",
    }


@app.get("/model-info", response_model=ModelInfoResponse, tags=["Model Info"])
def model_info():
    """Return model classes, version, and author metadata."""
    model = get_model()
    if model is None:
        return {
            "service_name": "Intent Classifier",
            "author": "Manjunath",
            "version": "1.0.0",
            "num_classes": 0,
            "classes": [],
            "model_loaded": False,
        }

    classes = model.get_classes()
    return {
        "service_name": "Intent Classifier",
        "author": "Manjunath",
        "version": "1.0.0",
        "num_classes": len(classes),
        "classes": classes,
        "model_loaded": True,
    }


@app.post("/predict", response_model=PredictionResponse, tags=["Inference"])
def predict(request: PredictionRequest):
    """Predict the customer intent for a single message with probability scores."""
    model = get_model()
    if model is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Model is not loaded. Please ensure the model artifact exists.",
        )

    result = model.predict(
        text=request.text,
        confidence_threshold=request.confidence_threshold,
    )
    return result


@app.post("/predict-batch", response_model=BatchPredictionResponse, tags=["Inference"])
def predict_batch(request: BatchPredictionRequest):
    """Predict intents for a batch of messages."""
    model = get_model()
    if model is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Model is not loaded. Please ensure the model artifact exists.",
        )

    results = model.predict_batch(
        texts=request.texts,
        confidence_threshold=request.confidence_threshold,
    )
    return {
        "count": len(results),
        "predictions": results,
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=True)
