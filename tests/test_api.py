"""
Integration Tests for FastAPI Serving Application
Author: Manjunath
Project: Intent Classifier MLOps Production Pipeline
"""

import pytest
from pathlib import Path
from fastapi.testclient import TestClient

from app import app
import app as app_module
from model.intent_model import IntentModel, DEFAULT_MODEL_PATH


@pytest.fixture(scope="module", autouse=True)
def ensure_model_loaded():
    """Ensure the model artifact exists and initialize the model instance."""
    if not DEFAULT_MODEL_PATH.exists():
        from model.train import train_model
        base_dir = Path(__file__).resolve().parent.parent
        data_path = base_dir / "data" / "intents.csv"
        artifacts_dir = base_dir / "model" / "artifacts"
        train_model(data_path, artifacts_dir)

    app_module.model_instance = IntentModel(DEFAULT_MODEL_PATH)
    yield
    app_module.model_instance = None


@pytest.fixture
def client():
    return TestClient(app)


def test_root_endpoint(client):
    """Verify welcome root endpoint returns author Manjunath and docs link."""
    response = client.get("/")
    assert response.status_code == 200
    data = response.json()
    assert data["author"] == "Manjunath"
    assert data["docs_url"] == "/docs"


def test_health_endpoint(client):
    """Verify health endpoint indicates healthy service."""
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "healthy"
    assert data["model_loaded"] is True
    assert data["author"] == "Manjunath"


def test_model_info_endpoint(client):
    """Verify model metadata endpoint."""
    response = client.get("/model-info")
    assert response.status_code == 200
    data = response.json()
    assert data["author"] == "Manjunath"
    assert data["num_classes"] >= 5
    assert "greeting" in data["classes"]


def test_predict_endpoint_valid(client):
    """Verify single prediction with valid input."""
    payload = {"text": "I want to cancel my subscription"}
    response = client.post("/predict", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert data["intent"] == "cancel_subscription"
    assert "confidence" in data
    assert "probabilities" in data
    assert isinstance(data["probabilities"], dict)
    assert data["is_fallback"] is False


def test_predict_endpoint_empty_text(client):
    """Verify validation error when text is empty string."""
    payload = {"text": ""}
    response = client.post("/predict", json=payload)
    assert response.status_code == 422  # Pydantic min_length=1 validation


def test_predict_endpoint_missing_payload(client):
    """Verify error on missing payload."""
    response = client.post("/predict", json={})
    assert response.status_code == 422


def test_predict_batch_endpoint(client):
    """Verify batch prediction endpoint."""
    payload = {
        "texts": [
            "good morning team",
            "where is my order?",
            "can I get a refund?",
        ]
    }
    response = client.post("/predict-batch", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert data["count"] == 3
    assert len(data["predictions"]) == 3
    assert data["predictions"][0]["intent"] == "greeting"
    assert data["predictions"][1]["intent"] == "order_status"
    assert data["predictions"][2]["intent"] == "refund_request"
