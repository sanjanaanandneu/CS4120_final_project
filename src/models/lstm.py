import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import DataLoader, TensorDataset
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

"""
In order to run this, run python src/models/lstm.py from project root.
"""

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

        self.cells = []
        for layer in range(num_layers):
            in_size = input_size if layer == 0 else hidden_size
            self.cells.append(LSTMCell(in_size, hidden_size))

    def parameters(self):
        params = []
        for cell in self.cells:
            params += cell.parameters()
        return params

    def forward(self, X):
        # X: (batch_size, max_len, input_size)
        X = X.permute(1, 0, 2)                       # (max_len, batch_size, input_size)
        seq_len, batch_size, _ = X.shape

        h = [torch.zeros(batch_size, self.hidden_size) for _ in range(self.num_layers)]
        c = [torch.zeros(batch_size, self.hidden_size) for _ in range(self.num_layers)]

        for t in range(seq_len):
            x_t = X[t]                               # (batch_size, input_size)
            for layer_idx, cell in enumerate(self.cells):
                h[layer_idx], c[layer_idx] = cell.forward(x_t, h[layer_idx], c[layer_idx])
                x_t = h[layer_idx]                   # next layer's input

        return h[-1]                                  # (batch_size, hidden_size)

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

    def fit(self, X, y, X_val=None, y_val=None, learning_rate=0.001, batch_size=32, num_epochs=10):
        X_tensor   = torch.tensor(X, dtype=torch.float32)
        y_tensor   = torch.tensor(y, dtype=torch.float32)
        dataset    = TensorDataset(X_tensor, y_tensor)
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
        optimizer  = torch.optim.Adam(self.parameters(), lr=learning_rate)
        loss_fn    = nn.BCELoss()

        self.train_losses      = []
        self.train_accuracies  = []
        self.val_losses        = []      
        self.val_accuracies    = []      

        for epoch in range(num_epochs):
            epoch_loss = 0
            correct    = 0
            total      = 0

            for X_batch, y_batch in dataloader:
                optimizer.zero_grad()
                probs = self.forward(X_batch)
                loss  = loss_fn(probs, y_batch)
                loss.backward()
                optimizer.step()

                epoch_loss += loss.item()
                preds    = (probs >= 0.5).int()
                correct += (preds == y_batch.int()).sum().item()
                total   += y_batch.size(0)

            avg_loss = epoch_loss / len(dataloader)
            avg_acc  = correct / total
            self.train_losses.append(avg_loss)
            self.train_accuracies.append(avg_acc)

            if X_val is not None:
                with torch.no_grad():
                    X_val_t   = torch.tensor(X_val, dtype=torch.float32)
                    y_val_t   = torch.tensor(y_val, dtype=torch.float32)
                    val_probs = self.forward(X_val_t)
                    val_loss  = loss_fn(val_probs, y_val_t).item()
                    val_acc   = ((val_probs >= 0.5).int() == y_val_t.int()).float().mean().item()
                self.val_losses.append(val_loss)
                self.val_accuracies.append(val_acc)
                print(f"Epoch {epoch+1}/{num_epochs} — loss: {avg_loss:.4f}  acc: {avg_acc:.4f}  val_loss: {val_loss:.4f}  val_acc: {val_acc:.4f}")
            else:
                print(f"Epoch {epoch+1}/{num_epochs} — loss: {avg_loss:.4f}  acc: {avg_acc:.4f}")

    def predict(self, X, batch_size=32):
        X_tensor   = torch.tensor(X, dtype=torch.float32)
        dataloader = DataLoader(TensorDataset(X_tensor), batch_size=batch_size)
        all_preds  = []
        with torch.no_grad():
            for (X_batch,) in dataloader:
                probs = self.forward(X_batch)
                all_preds.append((probs >= 0.5).int().numpy())
        return np.concatenate(all_preds)

if __name__ == "__main__":
    DATASET = "hc3"
    X_train = np.load(f"data/processed/features/{DATASET}/word2vec_embeddings_train.npy")
    X_test  = np.load(f"data/processed/features/{DATASET}/word2vec_embeddings_test.npy")
    y_train = np.load(f"data/processed/features/{DATASET}/y_train.npy")
    y_test  = np.load(f"data/processed/features/{DATASET}/y_test.npy")

    X_train, X_val, y_train, y_val = train_test_split(
        X_train, y_train, test_size=0.15, random_state=42
)

    model = LSTMClassifier(input_size=100, hidden_size=128, num_layers=2)
    model.fit(X_train, y_train, X_val=X_val, y_val=y_val)

    # Test predictions
    preds    = model.predict(X_test)
    test_acc = np.mean(preds == y_test)
    print(f"Test Accuracy: {test_acc:.4f}")

    # 1. Class balance
    print(f"\nClass balance (test):")
    print(f"  Human (0): {np.bincount(y_test)[0]}")
    print(f"  AI    (1): {np.bincount(y_test)[1]}")

    # 2. Train vs test accuracy
    train_preds = model.predict(X_train)
    train_acc   = np.mean(train_preds == y_train)
    print(f"\nTrain Accuracy: {train_acc:.4f}")
    print(f"Test Accuracy:  {test_acc:.4f}")
    print(f"Gap:            {train_acc - test_acc:.4f}")

    # 3. Full classification report
    from sklearn.metrics import classification_report
    print("\nClassification Report:")
    print(classification_report(y_test, preds, target_names=["human", "AI"]))

    epochs = range(1, len(model.train_losses) + 1)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
    fig.suptitle("LSTM Training Curves", fontsize=14, fontweight="bold")

    # Loss plot
    ax1.plot(epochs, model.train_losses, color="#e74c3c", linewidth=2, marker="o", markersize=4)
    ax1.plot(epochs, model.val_losses, color="#e67e22", linewidth=2, marker="o", markersize=4, label="Val")

    ax1.set_title("Training Loss")
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("BCE Loss")
    ax1.grid(True, alpha=0.3)
    ax1.set_xticks(epochs)

    # Accuracy plot
    ax2.plot(epochs, model.train_accuracies, color="#2ecc71", linewidth=2, marker="o", markersize=4)
    ax2.plot(epochs, model.val_accuracies, color="#1abc9c", linewidth=2, marker="o", markersize=4, label="Val")
    ax2.set_title("Training Accuracy")
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("Accuracy")
    ax2.set_ylim(0, 1)
    ax2.grid(True, alpha=0.3)
    ax2.set_xticks(epochs)

    plt.tight_layout()
    plt.savefig("LSTM_plots.png", dpi=150, bbox_inches="tight")
    plt.show()
