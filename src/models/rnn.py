"""PyTorch RNN (stacked LSTM) text classifier for AI-text detection.

Accepts pre-extracted Word2Vec embeddings as input — shape (n, seq_len, embedding_dim).

Architecture:
    nn.LSTM(hidden_dim=128, all timesteps) -> Dropout
    -> nn.LSTM(hidden_dim=64, final hidden state) -> Dropout
    -> Linear(64, 32) -> ReLU -> Linear(32, 1) -> sigmoid
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

from src.models.base import BaseModel


class RNN(nn.Module, BaseModel):
    def __init__(
        self,
        embedding_dim: int = 100,
        hidden_dim: int = 128,
        dropout_rate: float = 0.2,
    ):
        nn.Module.__init__(self)
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        self.dropout_rate = dropout_rate

        self.lstm1 = nn.LSTM(embedding_dim, hidden_dim, batch_first=True)
        self.dropout1 = nn.Dropout(dropout_rate)
        self.lstm2 = nn.LSTM(hidden_dim, hidden_dim // 2, batch_first=True)
        self.dropout2 = nn.Dropout(dropout_rate)
        self.fc1 = nn.Linear(hidden_dim // 2, 32)
        self.fc2 = nn.Linear(32, 1)

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        # X: (batch, seq_len, embedding_dim)
        x, _ = self.lstm1(X)                # (batch, seq_len, hidden_dim) — all timesteps
        x = self.dropout1(x)
        _, (h_n, _) = self.lstm2(x)         # final hidden state: (1, batch, hidden_dim // 2)
        x = self.dropout2(h_n.squeeze(0))   # (batch, hidden_dim // 2)
        x = F.relu(self.fc1(x))             # (batch, 32)
        x = self.fc2(x)                     # (batch, 1)
        return torch.sigmoid(x).squeeze(1)  # (batch,)

    def fit(self, X: np.ndarray, y, epochs: int = 10, batch_size: int = 32, lr: float = 0.001) -> None:
        X_tensor = torch.tensor(X, dtype=torch.float32)
        y_tensor = torch.tensor(np.asarray(y), dtype=torch.float32)
        loader = DataLoader(TensorDataset(X_tensor, y_tensor), batch_size=batch_size, shuffle=True)
        optimizer = torch.optim.Adam(self.parameters(), lr=lr)
        loss_fn = nn.BCELoss()

        self.train_losses = []
        self.train_accuracies = []
        self.train()
        for epoch in range(epochs):
            epoch_loss = 0.0
            correct = 0
            total = 0
            for X_batch, y_batch in loader:
                optimizer.zero_grad()
                probs = self.forward(X_batch)
                loss = loss_fn(probs, y_batch)
                loss.backward()
                optimizer.step()
                epoch_loss += loss.item()
                correct += ((probs >= 0.5).int() == y_batch.int()).sum().item()
                total += y_batch.size(0)
            avg_loss = epoch_loss / len(loader)
            avg_acc = correct / total
            self.train_losses.append(avg_loss)
            self.train_accuracies.append(avg_acc)
            print(f"Epoch {epoch + 1}/{epochs} — Loss: {avg_loss:.4f}  Acc: {avg_acc:.4f}")

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
                    "hidden_dim": self.hidden_dim,
                    "dropout_rate": self.dropout_rate,
                },
            },
            filepath,
        )

    @classmethod
    def load(cls, filepath: str) -> "RNN":
        checkpoint = torch.load(filepath, weights_only=False)
        model = cls(**checkpoint["config"])
        model.load_state_dict(checkpoint["state_dict"])
        return model
