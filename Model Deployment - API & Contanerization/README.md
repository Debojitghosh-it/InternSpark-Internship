# Iris Classifier API

A PyTorch model trained on the classic Iris dataset, served via FastAPI and
containerized with Docker.

## Project structure

```
.
├── model.py           # PyTorch model architecture (IrisNet)
├── train.py            # Trains the model, saves artifacts/model.pth + scaler.joblib
├── app.py               # FastAPI app that loads the artifacts and serves predictions
├── requirements.txt
├── Dockerfile
└── artifacts/            # Created by train.py (model weights + scaler)
```

## Endpoints

| Method | Path       | Description                          |
|--------|------------|---------------------------------------|
| GET    | `/health`  | Liveness check                       |
| POST   | `/predict` | Run inference on 4 iris measurements |

Interactive Swagger docs are available at `/docs` once the server is running.

## Run locally (no Docker)

```bash
pip install -r requirements.txt

# Train the model (only needed once, or whenever you want to retrain)
python train.py

# Start the API
uvicorn app:app --host 0.0.0.0 --port 8000
```

## Run with Docker

```bash
# Build (make sure artifacts/ exists — run train.py first if it doesn't)
docker build -t iris-api .

# Run
docker run -p 8000:8000 iris-api
```

The API will be available at `http://localhost:8000`.

## Example request & response

**Request:**

```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"sepal_length": 5.1, "sepal_width": 3.5, "petal_length": 1.4, "petal_width": 0.2}'
```

**Response:**

```json
{
    "predicted_class": "setosa",
    "predicted_class_index": 0,
    "probabilities": {
        "setosa": 0.9999,
        "versicolor": 0.0001,
        "virginica": 0.0
    }
}
```

A second example, with different measurements, correctly predicts a different class:

```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"sepal_length": 6.7, "sepal_width": 3.0, "petal_length": 5.2, "petal_width": 2.3}'
```

```json
{
    "predicted_class": "virginica",
    "predicted_class_index": 2,
    "probabilities": {
        "setosa": 0.0,
        "versicolor": 0.0,
        "virginica": 1.0
    }
}
```

**Health check:**

```bash
curl http://localhost:8000/health
# {"status":"ok"}
```

## Notes / swapping in your own model

This repo uses a small feed-forward net on the Iris dataset so everything
runs end-to-end without external downloads. To use your own trained model:

1. Replace `model.py` with your model's architecture (or loading logic).
2. Update `train.py` (or your own training script) to save weights into `artifacts/`.
3. Update the `PredictRequest` / `PredictResponse` schemas in `app.py` to match
   your model's real inputs and outputs.
4. Rebuild the Docker image.
