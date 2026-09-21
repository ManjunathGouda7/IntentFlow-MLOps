"""
Intent Classifier FastAPI Serving Application
Author: Manjunath
Project: Intent Classifier MLOps Production Pipeline
"""

from contextlib import asynccontextmanager
from typing import Any, Dict, List, Optional
from fastapi import FastAPI, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse
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
        "interactive_ui": "/ui",
        "docs_url": "/docs",
        "redoc_url": "/redoc",
        "health_check": "/health",
    }


@app.get("/ui", response_class=HTMLResponse, tags=["General"])
def interactive_ui():
    """Interactive visual web demo for real-time intent classification."""
    return """<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>Intent Classifier AI - by Manjunath</title>
  <link rel="preconnect" href="https://fonts.googleapis.com">
  <link href="https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@400;500;600;700&display=swap" rel="stylesheet">
  <style>
    :root {
      --bg: #0b0f19;
      --card-bg: rgba(22, 28, 45, 0.75);
      --border: rgba(255, 255, 255, 0.08);
      --accent: #6366f1;
      --accent-hover: #4f46e5;
      --accent-grad: linear-gradient(135deg, #6366f1 0%, #a855f7 100%);
      --text: #f3f4f6;
      --text-muted: #9ca3af;
      --success: #10b981;
      --warning: #f59e0b;
    }
    * { box-sizing: border-box; margin: 0; padding: 0; }
    body {
      font-family: 'Plus Jakarta Sans', sans-serif;
      background: var(--bg);
      color: var(--text);
      min-height: 100vh;
      display: flex;
      flex-direction: column;
      align-items: center;
      padding: 2.5rem 1rem;
      background-image: radial-gradient(circle at 50% 0%, rgba(99, 102, 241, 0.15) 0%, transparent 60%);
    }
    .container {
      width: 100%;
      max-width: 780px;
    }
    .header {
      text-align: center;
      margin-bottom: 2rem;
    }
    .badge {
      display: inline-flex;
      align-items: center;
      gap: 0.5rem;
      padding: 0.35rem 0.85rem;
      border-radius: 9999px;
      font-size: 0.8rem;
      font-weight: 600;
      background: rgba(99, 102, 241, 0.15);
      color: #818cf8;
      border: 1px solid rgba(99, 102, 241, 0.3);
      margin-bottom: 1rem;
    }
    h1 {
      font-size: 2.4rem;
      font-weight: 700;
      background: var(--accent-grad);
      -webkit-background-clip: text;
      -webkit-text-fill-color: transparent;
      margin-bottom: 0.5rem;
    }
    p.subtitle {
      color: var(--text-muted);
      font-size: 1rem;
    }
    .card {
      background: var(--card-bg);
      backdrop-filter: blur(16px);
      border: 1px solid var(--border);
      border-radius: 1rem;
      padding: 1.75rem;
      box-shadow: 0 20px 40px -15px rgba(0,0,0,0.5);
      margin-bottom: 1.5rem;
    }
    label {
      display: block;
      font-weight: 600;
      margin-bottom: 0.5rem;
      font-size: 0.95rem;
    }
    textarea {
      width: 100%;
      padding: 0.9rem;
      background: rgba(11, 15, 25, 0.8);
      border: 1px solid var(--border);
      border-radius: 0.75rem;
      color: var(--text);
      font-family: inherit;
      font-size: 0.95rem;
      resize: vertical;
      min-height: 80px;
      outline: none;
      transition: border-color 0.2s;
    }
    textarea:focus { border-color: var(--accent); }
    .chips {
      display: flex;
      flex-wrap: wrap;
      gap: 0.5rem;
      margin-top: 0.85rem;
    }
    .chip {
      background: rgba(255,255,255,0.05);
      border: 1px solid var(--border);
      border-radius: 9999px;
      padding: 0.25rem 0.75rem;
      font-size: 0.75rem;
      color: var(--text-muted);
      cursor: pointer;
      transition: all 0.2s;
    }
    .chip:hover {
      background: rgba(99, 102, 241, 0.2);
      border-color: rgba(99, 102, 241, 0.4);
      color: #fff;
    }
    .btn {
      margin-top: 1rem;
      width: 100%;
      padding: 0.85rem;
      border: none;
      border-radius: 0.75rem;
      background: var(--accent-grad);
      color: #fff;
      font-weight: 600;
      font-size: 1rem;
      cursor: pointer;
      box-shadow: 0 4px 15px rgba(99, 102, 241, 0.4);
      transition: transform 0.15s, opacity 0.2s;
    }
    .btn:hover { opacity: 0.92; transform: translateY(-1px); }
    .btn:active { transform: translateY(0); }
    .result-section { display: none; margin-top: 1rem; }
    .top-result {
      display: flex;
      justify-content: space-between;
      align-items: center;
      padding: 1.25rem;
      background: rgba(99, 102, 241, 0.08);
      border: 1px solid rgba(99, 102, 241, 0.25);
      border-radius: 0.75rem;
      margin-bottom: 1.5rem;
    }
    .intent-name {
      font-size: 1.4rem;
      font-weight: 700;
      color: #a5b4fc;
      text-transform: capitalize;
    }
    .intent-badge {
      font-size: 0.85rem;
      font-weight: 600;
      padding: 0.35rem 0.85rem;
      border-radius: 9999px;
      background: rgba(16, 185, 129, 0.2);
      color: #34d399;
      border: 1px solid rgba(16, 185, 129, 0.3);
    }
    .intent-badge.fallback {
      background: rgba(245, 158, 11, 0.2);
      color: #fbbf24;
      border-color: rgba(245, 158, 11, 0.4);
    }
    .prob-bar-container {
      display: flex;
      flex-direction: column;
      gap: 0.75rem;
    }
    .prob-row {
      display: grid;
      grid-template-columns: 140px 1fr 60px;
      align-items: center;
      gap: 0.75rem;
      font-size: 0.85rem;
    }
    .prob-name {
      color: var(--text-muted);
      overflow: hidden;
      text-overflow: ellipsis;
      white-space: nowrap;
    }
    .bar-bg {
      height: 8px;
      background: rgba(255, 255, 255, 0.06);
      border-radius: 9999px;
      overflow: hidden;
    }
    .bar-fill {
      height: 100%;
      background: var(--accent-grad);
      border-radius: 9999px;
      width: 0%;
      transition: width 0.4s ease;
    }
    .prob-val {
      text-align: right;
      font-weight: 600;
      font-size: 0.8rem;
    }
    .footer-links {
      display: flex;
      justify-content: center;
      gap: 1.5rem;
      margin-top: 1.5rem;
      font-size: 0.85rem;
    }
    .footer-links a {
      color: var(--text-muted);
      text-decoration: none;
      transition: color 0.2s;
    }
    .footer-links a:hover { color: #818cf8; }
  </style>
</head>
<body>
  <div class="container">
    <div class="header">
      <div class="badge">MLOps Production Pipeline</div>
      <h1>Intent Classifier AI</h1>
      <p class="subtitle">Built with Scikit-learn, MLflow & FastAPI • By <strong>Manjunath</strong></p>
    </div>

    <div class="card">
      <label for="text-input">Enter Customer Query or Message:</label>
      <textarea id="text-input" placeholder="Type a message (e.g., 'I want to cancel my subscription' or 'where is my package?')..."></textarea>
      
      <div class="chips">
        <span class="chip" onclick="setQuery('I want to cancel my subscription')">Cancel subscription</span>
        <span class="chip" onclick="setQuery('Where is my package and delivery status?')">Order status</span>
        <span class="chip" onclick="setQuery('Please refund my money, item arrived broken')">Request refund</span>
        <span class="chip" onclick="setQuery('The dashboard keeps showing 500 internal error')">Technical support</span>
        <span class="chip" onclick="setQuery('Why was I charged twice this month?')">Billing inquiry</span>
        <span class="chip" onclick="setQuery('You guys have the best support, thank you!')">Praise</span>
        <span class="chip" onclick="setQuery('Quantum tachyon relativistic wormhole')">Out of scope (Fallback)</span>
      </div>

      <button class="btn" id="submit-btn" onclick="classifyIntent()">Classify Intent</button>

      <div class="result-section" id="result-section">
        <div class="top-result">
          <div>
            <div style="font-size: 0.8rem; color: var(--text-muted); margin-bottom: 0.2rem;">Predicted Intent</div>
            <div class="intent-name" id="intent-title">-</div>
          </div>
          <div class="intent-badge" id="confidence-badge">-</div>
        </div>

        <div style="font-size: 0.85rem; font-weight: 600; margin-bottom: 0.75rem;">Class Probability Distribution:</div>
        <div class="prob-bar-container" id="prob-bars"></div>
      </div>
    </div>

    <div class="footer-links">
      <a href="/docs" target="_blank">📚 Swagger API Docs</a>
      <a href="/redoc" target="_blank">📖 ReDoc</a>
      <a href="/model-info" target="_blank">ℹ️ Model Info</a>
      <a href="/health" target="_blank">🩺 Health Probe</a>
      <a href="http://127.0.0.1:5000" target="_blank">📊 MLflow UI</a>
    </div>
  </div>

  <script>
    function setQuery(text) {
      document.getElementById('text-input').value = text;
      classifyIntent();
    }

    async function classifyIntent() {
      const input = document.getElementById('text-input').value.trim();
      if (!input) return;

      const btn = document.getElementById('submit-btn');
      btn.innerText = 'Classifying...';
      btn.disabled = true;

      try {
        const res = await fetch('/predict', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ text: input, confidence_threshold: 0.20 })
        });
        const data = await res.json();

        document.getElementById('result-section').style.display = 'block';
        document.getElementById('intent-title').innerText = data.intent.replace(/_/g, ' ');
        
        const badge = document.getElementById('confidence-badge');
        const pct = (data.confidence * 100).toFixed(1) + '%';
        if (data.is_fallback) {
          badge.innerText = `Fallback (${pct})`;
          badge.className = 'intent-badge fallback';
        } else {
          badge.innerText = `Confidence: ${pct}`;
          badge.className = 'intent-badge';
        }

        const barContainer = document.getElementById('prob-bars');
        barContainer.innerHTML = '';

        for (const [cls, prob] of Object.entries(data.probabilities)) {
          const probPct = (prob * 100).toFixed(1);
          const row = document.createElement('div');
          row.className = 'prob-row';
          row.innerHTML = `
            <div class="prob-name" title="${cls}">${cls.replace(/_/g, ' ')}</div>
            <div class="bar-bg"><div class="bar-fill" style="width: ${probPct}%;"></div></div>
            <div class="prob-val">${probPct}%</div>
          `;
          barContainer.appendChild(row);
        }
      } catch (err) {
        alert('Failed to connect to API: ' + err.message);
      } finally {
        btn.innerText = 'Classify Intent';
        btn.disabled = false;
      }
    }

    document.getElementById('text-input').addEventListener('keydown', (e) => {
      if (e.key === 'Enter' && !e.shiftKey) {
        e.preventDefault();
        classifyIntent();
      }
    });
  </script>
</body>
</html>"""


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
