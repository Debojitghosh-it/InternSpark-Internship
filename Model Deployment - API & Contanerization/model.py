"""
model.py

Defines the PyTorch model architecture. This module is imported by both
train.py (to build + train the model) and app.py (to rebuild the same
architecture before loading saved weights).
"""

import torch
import torch.nn as nn


class IrisNet(nn.Module):
    """
    A small feed-forward classifier for the classic Iris dataset.

    Input:  4 features (sepal length, sepal width, petal length, petal width)
    Output: 3 class logits (setosa, versicolor, virginica)
    """

    def __init__(self, input_dim: int = 4, hidden_dim: int = 16, num_classes: int = 3):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


CLASS_NAMES = ["setosa", "versicolor", "virginica"]
