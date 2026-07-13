"""
app.py

FastAPI wrapper around a trained PyTorch model (IrisNet).

Endpoints:
    GET  /health   -> simple liveness check
    POST /predict  -> run inference on one set of 4 iris features

Run locally:
    uvicorn app:app --host 0.0.0.0 --port 8000
"""

import os

import joblib
import torch
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from model import IrisNet, CLASS_NAMES

ARTIFACT_DIR = "artifacts"
MODEL_PATH = os.path.join(ARTIFACT_DIR, "model.pth")
SCALER_PATH = os.path.join(ARTIFACT_DIR, "scaler.joblib")

app = FastAPI(
    title="Iris Classifier API",
    description="Serves predictions from a PyTorch model trained on the Iris dataset.",
    version="1.0.0",
)

# --- Load model + scaler once at startup ---
model = IrisNet()

if not os.path.exists(MODEL_PATH) or not os.path.exists(SCALER_PATH):
    raise RuntimeError(
        f"Model artifacts not found in '{ARTIFACT_DIR}/'. "
        "Run `python train.py` first to generate model.pth and scaler.joblib."
    )

model.load_state_dict(torch.load(MODEL_PATH, map_location="cpu"))
model.eval()
scaler = joblib.load(SCALER_PATH)


class PredictRequest(BaseModel):
    sepal_length: float = Field(..., example=5.1)
    sepal_width: float = Field(..., example=3.5)
    petal_length: float = Field(..., example=1.4)
    petal_width: float = Field(..., example=0.2)


class PredictResponse(BaseModel):
    predicted_class: str
    predicted_class_index: int
    probabilities: dict[str, float]


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/predict", response_model=PredictResponse)
def predict(request: PredictRequest):
    try:
        features = [[
            request.sepal_length,
            request.sepal_width,
            request.petal_length,
            request.petal_width,
        ]]
        scaled = scaler.transform(features)
        x = torch.tensor(scaled, dtype=torch.float32)

        with torch.no_grad():
            logits = model(x)
            probs = torch.softmax(logits, dim=1).squeeze(0)
            pred_idx = int(torch.argmax(probs).item())

        return PredictResponse(
            predicted_class=CLASS_NAMES[pred_idx],
            predicted_class_index=pred_idx,
            probabilities={
                CLASS_NAMES[i]: round(float(probs[i]), 4) for i in range(len(CLASS_NAMES))
            },
        )
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))
