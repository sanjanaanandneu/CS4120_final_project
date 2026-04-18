import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import DataLoader, TensorDataset

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
        self.input_size  = input_size
        self.hidden_size = hidden_size
        self.num_layers  = num_layers

        self.cells = []
        for layer in range(num_layers):
            in_size = input_size if layer == 0 else hidden_size
            self.cells.append(LSTMCell(in_size, hidden_size))

    def parameters(self):
        params = []
        for cell in self.cells:
            params += cell.parameters()
        return params

    def init_hidden(self, batch_size):
        h0 = torch.zeros(self.num_layers, batch_size, self.hidden_size)
        c0 = torch.zeros(self.num_layers, batch_size, self.hidden_size)
        return h0, c0

    def forward(self, X):
        X = X.permute(1, 0, 2)                      
        seq_len, batch_size, _ = X.shape

        h, c = self.init_hidden(batch_size)

        for t in range(seq_len):
            x_t = X[t]
            for layer_idx, cell in enumerate(self.cells):
                h[layer_idx], c[layer_idx] = cell.forward(x_t, h[layer_idx], c[layer_idx])
                x_t = h[layer_idx]

        return h[-1]                                


class LSTMClassifier:
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


# --- Entry point ---
if __name__ == "__main__":
    DATASET = "hc3"
    X_train = np.load(f"data/processed/features/{DATASET}/word2vec_embeddings_train.npy")
    X_test  = np.load(f"data/processed/features/{DATASET}/word2vec_embeddings_test.npy")
    y_train = np.load(f"data/processed/features/{DATASET}/y_train.npy")
    y_test  = np.load(f"data/processed/features/{DATASET}/y_test.npy")

    model = LSTMClassifier(input_size=100, hidden_size=128, num_layers=2)
    model.fit(X_train, y_train)

    preds    = model.predict(X_test)
    accuracy = np.mean(preds == y_test)
    print(f"Test Accuracy: {accuracy:.4f}")