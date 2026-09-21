"""
Intent Classifier Model Wrapper
Author: Manjunath
Project: Intent Classifier MLOps Production Pipeline
"""

from pathlib import Path
from typing import Any, Dict, List, Optional
import joblib


DEFAULT_MODEL_PATH = Path(__file__).resolve().parent / "artifacts" / "intent_model.pkl"


class IntentModel:
    """Production wrapper for intent classification and probability estimation."""

    def __init__(self, path: Optional[Path | str] = None):
        self.path = Path(path) if path else DEFAULT_MODEL_PATH
        if not self.path.exists():
            raise FileNotFoundError(
                f"Model artifact not found at '{self.path}'. "
                f"Please run 'python model/train.py' first."
            )
        self.pipeline = joblib.load(self.path)
        self.classes_: List[str] = list(getattr(self.pipeline, "classes_", []))

    def predict(
        self,
        text: str,
        confidence_threshold: float = 0.20
    ) -> Dict[str, Any]:
        """
        Predict intent for a single text input with full probability distribution.
        
        Args:
            text: Input string to classify.
            confidence_threshold: Minimum probability required to assign the top intent;
                                 otherwise marks as fallback.
        Returns:
            Dict containing text, predicted intent, confidence, probabilities, and fallback flag.
        """
        cleaned_text = (text or "").strip()
        if not cleaned_text:
            return {
                "text": text,
                "intent": "unknown",
                "confidence": 0.0,
                "probabilities": {c: 0.0 for c in self.classes_},
                "is_fallback": True,
            }

        probs = self.pipeline.predict_proba([cleaned_text])[0]
        best_idx = int(probs.argmax())
        best_intent = str(self.classes_[best_idx])
        best_confidence = float(probs[best_idx])

        # Sort probability map from highest to lowest
        prob_dict = {
            cls: round(float(p), 4)
            for cls, p in sorted(
                zip(self.classes_, probs), key=lambda item: item[1], reverse=True
            )
        }

        is_fallback = best_confidence < confidence_threshold
        predicted_intent = "fallback" if is_fallback else best_intent

        return {
            "text": cleaned_text,
            "intent": predicted_intent,
            "confidence": round(best_confidence, 4),
            "probabilities": prob_dict,
            "is_fallback": is_fallback,
        }

    def predict_batch(
        self,
        texts: List[str],
        confidence_threshold: float = 0.20
    ) -> List[Dict[str, Any]]:
        """Predict intents for a batch of strings."""
        return [self.predict(t, confidence_threshold=confidence_threshold) for t in texts]

    def get_classes(self) -> List[str]:
        """Return list of supported intent classes."""
        return self.classes_
