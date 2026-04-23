import numpy as np
from scipy import sparse
from tqdm import tqdm

from src.models.base import BaseModel


class LinearSVC(BaseModel):
    """
    Custom implementation of Linear Support Vector Classifier (SVM) for binary classification.
    """
    
    def __init__(self, learning_rate=0.01, epochs=1000, C=1.0):
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.C = C
        self.weights = None
        self.bias = None
    
    def fit(self, X, y):
        sparse_X = sparse.issparse(X)
        if not sparse_X:
            X = np.array(X)
        y = np.array(y)
        y_svm = np.where(y == 0, -1, 1)
        n_samples, n_features = X.shape

        self.weights = np.zeros(n_features)
        self.bias = 0.0
        self.train_losses = []
        self.train_accuracies = []

        step = max(1, self.epochs // 200)

        for epoch in tqdm(range(self.epochs)):
            if sparse_X:
                decision = np.array(X.dot(self.weights)).ravel() + self.bias
            else:
                decision = np.dot(X, self.weights) + self.bias

            misclassified = (y_svm * decision < 1).astype(float)

            if sparse_X:
                dw = -(self.C / n_samples) * np.array(X.T.dot(y_svm * misclassified)).ravel() + self.weights
            else:
                dw = -(self.C / n_samples) * np.dot(X.T, y_svm * misclassified) + self.weights
            db = -(self.C / n_samples) * np.sum(y_svm * misclassified)

            self.weights -= self.learning_rate * dw
            self.bias -= self.learning_rate * db

            if epoch % step == 0:
                loss = np.mean(np.maximum(0, 1 - y_svm * decision))
                y_pred = np.where(np.sign(decision) == -1, 0, 1)
                acc = np.mean(y_pred == y)
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
            decision = np.array(X.dot(self.weights)).ravel() + self.bias
        else:
            decision = np.dot(X, self.weights) + self.bias
        
        # positive for class 1, negative for class -1
        y_predicted_svm = np.sign(decision)
        
        # Convert back to 0/1 labels
        y_predicted = np.where(y_predicted_svm == -1, 0, 1)
        
        return y_predicted
    
    def save(self, filepath):
        """
        Save the model to a file.
        
        Args:
            filepath (str): Path to save the model (e.g., 'saved_models/svm_hc3_word_ngram.pkl').
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
            LinearSVC: Loaded model instance.
        """
        import joblib
        return joblib.load(filepath)

