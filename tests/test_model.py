"""
Unit Tests for IntentModel
Author: Manjunath
Project: Intent Classifier MLOps Production Pipeline
"""

import pytest
from pathlib import Path
from model.intent_model import IntentModel, DEFAULT_MODEL_PATH


@pytest.fixture(scope="module")
def model():
    """Ensure model artifact exists and return IntentModel instance."""
    if not DEFAULT_MODEL_PATH.exists():
        from model.train import train_model
        base_dir = Path(__file__).resolve().parent.parent
        data_path = base_dir / "data" / "intents.csv"
        artifacts_dir = base_dir / "model" / "artifacts"
        train_model(data_path, artifacts_dir)
    return IntentModel()


def test_model_loading(model):
    """Verify that model artifact loads with valid classes."""
    classes = model.get_classes()
    assert len(classes) >= 5
    assert "greeting" in classes
    assert "cancel_subscription" in classes
    assert "order_status" in classes


def test_predict_greeting(model):
    """Verify greeting intent classification."""
    result = model.predict("hello there, good morning")
    assert result["intent"] == "greeting"
    assert result["confidence"] > 0.3
    assert result["is_fallback"] is False
    assert "greeting" in result["probabilities"]


def test_predict_cancel_subscription(model):
    """Verify cancellation intent classification."""
    result = model.predict("I need to cancel my paid subscription immediately")
    assert result["intent"] == "cancel_subscription"
    assert result["confidence"] > 0.3
    assert result["is_fallback"] is False


def test_predict_order_status(model):
    """Verify order tracking intent classification."""
    result = model.predict("where is my order and when will it arrive?")
    assert result["intent"] == "order_status"
    assert result["is_fallback"] is False


def test_probabilities_sum_to_one(model):
    """Verify that predicted probabilities sum to approximately 1.0."""
    result = model.predict("I want my money back")
    probs = result["probabilities"].values()
    assert sum(probs) == pytest.approx(1.0, abs=0.05)


def test_empty_string_handling(model):
    """Verify handling of empty string or whitespace."""
    result = model.predict("   ")
    assert result["intent"] == "unknown"
    assert result["confidence"] == 0.0
    assert result["is_fallback"] is True


def test_confidence_threshold_fallback(model):
    """Verify fallback when confidence threshold is set very high."""
    result = model.predict("xyzabc random gibberish 9999", confidence_threshold=0.99)
    assert result["is_fallback"] is True
    assert result["intent"] == "fallback"


def test_predict_batch(model):
    """Verify batch prediction returns matching number of responses."""
    queries = [
        "hi",
        "where is my package?",
        "cancel my plan",
        "thank you so much",
    ]
    results = model.predict_batch(queries)
    assert len(results) == len(queries)
    for res in results:
        assert "intent" in res
        assert "confidence" in res
        assert "probabilities" in res
