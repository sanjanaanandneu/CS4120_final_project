"""PyTorch CNN text classifier for AI-text detection.

Accepts pre-extracted Word2Vec embeddings as input — shape (n, seq_len, embedding_dim).

Architecture:
    nn.Conv1d -> ReLU -> global max pool
    -> Dropout -> Linear(num_filters, 32) -> ReLU -> Linear(32, 1) -> sigmoid
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

from src.models.base import BaseModel


class CNN(nn.Module, BaseModel):
    def __init__(
        self,
        embedding_dim: int = 100,
        num_filters: int = 128,
        kernel_size: int = 5,
        dropout_rate: float = 0.2,
    ):
        nn.Module.__init__(self)
        self.embedding_dim = embedding_dim
        self.num_filters = num_filters
        self.kernel_size = kernel_size
        self.dropout_rate = dropout_rate

        self.conv = nn.Conv1d(embedding_dim, num_filters, kernel_size)
        self.dropout = nn.Dropout(dropout_rate)
        self.fc1 = nn.Linear(num_filters, 32)
        self.fc2 = nn.Linear(32, 1)

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        # X: (batch, seq_len, embedding_dim)
        x = X.permute(0, 2, 1)          # (batch, embedding_dim, seq_len)
        x = F.relu(self.conv(x))         # (batch, num_filters, seq_len - kernel_size + 1)
        x = torch.max(x, dim=2).values   # (batch, num_filters) — global max pool
        x = self.dropout(x)
        x = F.relu(self.fc1(x))          # (batch, 32)
        x = self.fc2(x)                  # (batch, 1)
        return torch.sigmoid(x).squeeze(1)  # (batch,)

    def fit(self, X: np.ndarray, y, epochs: int = 10, batch_size: int = 32, lr: float = 0.001) -> None:
        X_tensor = torch.tensor(X, dtype=torch.float32)
        y_tensor = torch.tensor(np.asarray(y), dtype=torch.float32)
        loader = DataLoader(TensorDataset(X_tensor, y_tensor), batch_size=batch_size, shuffle=True)
        optimizer = torch.optim.Adam(self.parameters(), lr=lr)
        loss_fn = nn.BCELoss()

        self.train()
        for epoch in range(epochs):
            epoch_loss = 0.0
            for X_batch, y_batch in loader:
                optimizer.zero_grad()
                loss = loss_fn(self.forward(X_batch), y_batch)
                loss.backward()
                optimizer.step()
                epoch_loss += loss.item()
            print(f"Epoch {epoch + 1}/{epochs} — Loss: {epoch_loss / len(loader):.4f}")

    def predict(self, X: np.ndarray, batch_size: int = 32) -> np.ndarray:
        X_tensor = torch.tensor(X, dtype=torch.float32)
        loader = DataLoader(TensorDataset(X_tensor), batch_size=batch_size)
        all_preds: list[np.ndarray] = []
        self.eval()
        with torch.no_grad():
            for (X_batch,) in loader:
                probs = self.forward(X_batch)
                all_preds.append((probs >= 0.5).int().numpy())
        return np.concatenate(all_preds)

    def save(self, filepath: str) -> None:
        Path(filepath).parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "state_dict": self.state_dict(),
                "config": {
                    "embedding_dim": self.embedding_dim,
                    "num_filters": self.num_filters,
                    "kernel_size": self.kernel_size,
                    "dropout_rate": self.dropout_rate,
                },
            },
            filepath,
        )

    @classmethod
    def load(cls, filepath: str) -> "CNN":
        checkpoint = torch.load(filepath, weights_only=False)
        model = cls(**checkpoint["config"])
        model.load_state_dict(checkpoint["state_dict"])
        return model
