import numpy as np
from scipy import sparse
from tqdm import tqdm

from src.models.base import BaseModel


class LogisticRegression(BaseModel):
    """
    Custom implementation of Logistic Regression for binary classification.
    """
    
    def __init__(self, learning_rate=0.01, epochs=1000):
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.weights = None
        self.bias = None
    
    def _sigmoid(self, z):
        return 1 / (1 + np.exp(-z))
    
    def fit(self, X, y):
        sparse_X = sparse.issparse(X)
        if not sparse_X:
            X = np.array(X)
        y = np.array(y)
        n_samples, n_features = X.shape

        self.weights = np.zeros(n_features)
        self.bias = 0.0
        self.train_losses = []
        self.train_accuracies = []

        step = max(1, self.epochs // 200)

        for epoch in tqdm(range(self.epochs)):
            if sparse_X:
                linear_model = np.array(X.dot(self.weights)).ravel() + self.bias
            else:
                linear_model = np.dot(X, self.weights) + self.bias

            y_predicted = self._sigmoid(linear_model)

            if sparse_X:
                dw = (1 / n_samples) * np.array(X.T.dot(y_predicted - y)).ravel()
            else:
                dw = (1 / n_samples) * np.dot(X.T, (y_predicted - y))
            db = (1 / n_samples) * np.sum(y_predicted - y)

            self.weights -= self.learning_rate * dw
            self.bias -= self.learning_rate * db

            if epoch % step == 0:
                y_pred_clipped = np.clip(y_predicted, 1e-15, 1 - 1e-15)
                loss = -np.mean(y * np.log(y_pred_clipped) + (1 - y) * np.log(1 - y_pred_clipped))
                acc = np.mean((y_predicted >= 0.5).astype(int) == y)
                self.train_losses.append(loss)
                self.train_accuracies.append(acc)
    
    def predict(self, X):
        """
        Make predictions on new data.
        
        Args:
            X (array-like or sparse matrix): Input features.
        
        Returns:
            array-like: Predicted labels (0 or 1).
        """
        sparse_X = sparse.issparse(X)

        if not sparse_X:
            X = np.array(X)
        
        if sparse_X:
            linear_model = np.array(X.dot(self.weights)).ravel() + self.bias
        else:
            linear_model = np.dot(X, self.weights) + self.bias
        
        # Apply sigmoid
        y_predicted_prob = self._sigmoid(linear_model)
        
        # Threshold at 0.5 for binary classification
        y_predicted = (y_predicted_prob >= 0.5).astype(int)
        
        return y_predicted
    
    def save(self, filepath):
        """
        Save the model to a file.
        
        Args:
            filepath (str): Path to save the model (e.g., 'saved_models/lr_hc3_tfidf.pkl').
        """
        import joblib
        joblib.dump(self, filepath)
    
    @classmethod
    def load(cls, filepath):
        """
        Load a model from a file.
        
        Args:
            filepath (str): Path to the saved model file.
        
        Returns:
            LogisticRegression: Loaded model instance.
        """
        import joblib
        return joblib.load(filepath)

