import torch
import torch.nn as nn
import numpy as np
from pathlib import Path
from torch.utils.data import DataLoader, TensorDataset

from src.models.base import BaseModel


class LSTMCell:
    def __init__(self, input_size, hidden_size):
        self.input_size  = input_size
        self.hidden_size = hidden_size

        scale = np.sqrt(1 / (input_size + hidden_size))
        def make_W():
            return torch.tensor(
                np.random.randn(input_size + hidden_size, hidden_size) * scale,
                dtype=torch.float32, requires_grad=True
            )
        self.W_f = make_W()
        self.W_i = make_W()
        self.W_o = make_W()
        self.W_c = make_W()
        self.b_f = torch.zeros(hidden_size, requires_grad=True)
        self.b_i = torch.zeros(hidden_size, requires_grad=True)
        self.b_o = torch.zeros(hidden_size, requires_grad=True)
        self.b_c = torch.zeros(hidden_size, requires_grad=True)

    def parameters(self):
        return [self.W_f, self.W_i, self.W_o, self.W_c,
                self.b_f, self.b_i, self.b_o, self.b_c]

    # forward logic
    def forward(self, x, h_prev, c_prev):
        xh      = torch.cat((x, h_prev), dim=1)
        f       = torch.sigmoid(xh @ self.W_f + self.b_f)   # forget gate
        i       = torch.sigmoid(xh @ self.W_i + self.b_i)   # input gate
        c_tilde = torch.tanh(xh   @ self.W_c + self.b_c)    # candidate
        o       = torch.sigmoid(xh @ self.W_o + self.b_o)   # output gate
        c       = f * c_prev + i * c_tilde                   # cell state
        h       = o * torch.tanh(c)                          # hidden state
        return h, c


class LSTM:
    def __init__(self, input_size, hidden_size, num_layers):
        self.hidden_size = hidden_size
        self.num_layers  = num_layers
        self.lstm        = nn.LSTM(input_size, hidden_size, 
                                   num_layers=num_layers, 
                                   batch_first=True)

    def parameters(self):
        return list(self.lstm.parameters())

    def forward(self, X):
        # X: (batch_size, max_len, input_size) — batch_first=True handles transpose
        _, (h_n, _) = self.lstm(X)
        return h_n[-1]           # (batch_size, hidden_size)

class LSTMClassifier(BaseModel):
    def __init__(self, input_size, hidden_size, num_layers):
        self.lstm       = LSTM(input_size, hidden_size, num_layers)
        self.classifier = nn.Linear(hidden_size, 1)  

    def parameters(self):
        return self.lstm.parameters() + list(self.classifier.parameters())

    def forward(self, X):
        h_final = self.lstm.forward(X)                        
        logit   = self.classifier(h_final)                    
        return torch.sigmoid(logit).squeeze(1)                

    def fit(self, X, y, learning_rate=0.001, batch_size=32, num_epochs=10):
        X_tensor   = torch.tensor(X, dtype=torch.float32)
        y_tensor   = torch.tensor(y, dtype=torch.float32)
        dataset    = TensorDataset(X_tensor, y_tensor)
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
        optimizer  = torch.optim.Adam(self.parameters(), lr=learning_rate)
        loss_fn    = nn.BCELoss()

        for epoch in range(num_epochs):
            epoch_loss = 0
            for X_batch, y_batch in dataloader:
                optimizer.zero_grad()
                probs = self.forward(X_batch)
                loss  = loss_fn(probs, y_batch)
                loss.backward()                     
                optimizer.step()
                epoch_loss += loss.item()
            print(f"Epoch {epoch+1}/{num_epochs} — Loss: {epoch_loss/len(dataloader):.4f}")

    def predict(self, X, batch_size=32):
        X_tensor   = torch.tensor(X, dtype=torch.float32)
        dataloader = DataLoader(TensorDataset(X_tensor), batch_size=batch_size)
        all_preds  = []
        with torch.no_grad():
            for (X_batch,) in dataloader:
                probs = self.forward(X_batch)
                all_preds.append((probs >= 0.5).int().numpy())
        return np.concatenate(all_preds)

    def save(self, filepath: str) -> None:
        Path(filepath).parent.mkdir(parents=True, exist_ok=True)
        torch.save(self, filepath)

    @classmethod
    def load(cls, filepath: str) -> "LSTMClassifier":
        return torch.load(filepath, weights_only=False)
