import numpy as np
from scipy import sparse
from tqdm import tqdm

from src.models.base import BaseModel


class LinearSVC(BaseModel):
    """
    Custom implementation of Linear Support Vector Classifier (SVM) for binary classification.
    """
    
    def __init__(self, learning_rate=0.01, max_iter=10000, C=1.0):
        """
        Initialize the Linear SVM model.
        
        Args:
            learning_rate (float): The step size for gradient descent updates.
            max_iter (int): Maximum number of iterations for gradient descent.
            C (float): Regularization parameter (inverse of regularization strength).
        """
        self.learning_rate = learning_rate
        self.max_iter = max_iter
        self.C = C
        self.weights = None  # Will be initialized in fit
        self.bias = None     # Will be initialized in fit
    
    def fit(self, X, y):
        """
        Train the linear SVM model using gradient descent on hinge loss.
        
        Args:
            X (array-like or sparse matrix): Training features.
            y (array-like): Training labels (0 or 1, will be converted to -1 or 1 internally).
        """
        sparse_X = sparse.issparse(X)

        if not sparse_X:
            X = np.array(X)
        
        y = np.array(y)
        
        # Convert labels to -1 and 1 for SVM
        y_svm = np.where(y == 0, -1, 1)
        
        n_samples, n_features = X.shape
        
        # Initialize weights and bias
        self.weights = np.zeros(n_features)
        self.bias = 0.0
        
        for _ in tqdm(range(self.max_iter)):
            if sparse_X:
                decision = np.array(X.dot(self.weights)).ravel() + self.bias
            else:
                decision = np.dot(X, self.weights) + self.bias
            
            # Compute hinge loss gradients
            # For hinge loss: max(0, 1 - y * decision)
            # Gradient w.r.t. weights: -C * sum(y_i * X_i) for misclassified points + 2 * weights (L2 reg)
            # Gradient w.r.t. bias: -C * sum(y_i) for misclassified points
            
            # Indicator for misclassified points (where y * decision < 1)
            misclassified = (y_svm * decision < 1).astype(float)
            
            # Gradients
            if sparse_X:
                dw = -self.C * np.array(X.T.dot(y_svm * misclassified)).ravel() + 2 * self.weights
            else:
                dw = -self.C * np.dot(X.T, y_svm * misclassified) + 2 * self.weights
            db = -self.C * np.sum(y_svm * misclassified)
            
            # Update
            self.weights -= self.learning_rate * dw
            self.bias -= self.learning_rate * db
    
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

