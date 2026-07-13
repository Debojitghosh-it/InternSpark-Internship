"""
train.py

Trains IrisNet on the classic Iris dataset and saves two artifacts into
./artifacts/:

  - model.pth      : trained PyTorch weights
  - scaler.joblib   : fitted StandardScaler used to normalize inputs

Run:
    python train.py
"""

import os

import joblib
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from model import IrisNet

ARTIFACT_DIR = "artifacts"
MODEL_PATH = os.path.join(ARTIFACT_DIR, "model.pth")
SCALER_PATH = os.path.join(ARTIFACT_DIR, "scaler.joblib")


def main():
    os.makedirs(ARTIFACT_DIR, exist_ok=True)

    # --- Data ---
    data = load_iris()
    X, y = data.data, data.target

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    X_train_t = torch.tensor(X_train, dtype=torch.float32)
    y_train_t = torch.tensor(y_train, dtype=torch.long)
    X_test_t = torch.tensor(X_test, dtype=torch.float32)
    y_test_t = torch.tensor(y_test, dtype=torch.long)

    # --- Model / training setup ---
    model = IrisNet()
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.01)

    # --- Train ---
    epochs = 200
    for epoch in range(1, epochs + 1):
        model.train()
        optimizer.zero_grad()
        outputs = model(X_train_t)
        loss = criterion(outputs, y_train_t)
        loss.backward()
        optimizer.step()

        if epoch % 40 == 0 or epoch == 1:
            model.eval()
            with torch.no_grad():
                preds = model(X_test_t).argmax(dim=1)
                acc = (preds == y_test_t).float().mean().item()
            print(f"epoch {epoch:3d} | loss {loss.item():.4f} | test_acc {acc:.4f}")

    # --- Final eval ---
    model.eval()
    with torch.no_grad():
        preds = model(X_test_t).argmax(dim=1)
        final_acc = (preds == y_test_t).float().mean().item()
    print(f"\nFinal test accuracy: {final_acc:.4f}")

    # --- Save artifacts ---
    torch.save(model.state_dict(), MODEL_PATH)
    joblib.dump(scaler, SCALER_PATH)
    print(f"Saved model weights to {MODEL_PATH}")
    print(f"Saved scaler to {SCALER_PATH}")


if __name__ == "__main__":
    main()
