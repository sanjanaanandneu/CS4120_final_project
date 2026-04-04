import numpy as np
from scipy import sparse
from tqdm import tqdm

class LogisticRegression:
    """
    Custom implementation of Logistic Regression for binary classification.
    """
    
    def __init__(self, learning_rate=0.01, max_iter=1000):
        """
        Initialize the Logistic Regression model.
        
        Args:
            learning_rate (float): The step size for gradient descent updates.
            max_iter (int): Maximum number of iterations for gradient descent.
        """
        self.learning_rate = learning_rate
        self.max_iter = max_iter
        self.weights = None  # Will be initialized in fit
        self.bias = None     # Will be initialized in fit
    
    def _sigmoid(self, z):
        return 1 / (1 + np.exp(-z))
    
    def fit(self, X, y):
        """
        Train the logistic regression model using gradient descent.
        
        Args:
            X (array-like or sparse matrix): Training features.
            y (array-like): Training labels (0 or 1).
        """
        sparse_X = sparse.issparse(X)

        if not sparse_X:
            X = np.array(X)
        
        y = np.array(y)
        
        n_samples, n_features = X.shape
        
        # Initialize weights and bias
        self.weights = np.zeros(n_features)
        self.bias = 0.0
        
        for _ in tqdm(range(self.max_iter)):
            if sparse_X:
                linear_model = np.array(X.dot(self.weights)).ravel() + self.bias
            else:
                linear_model = np.dot(X, self.weights) + self.bias
            
            # Apply sigmoid
            y_predicted = self._sigmoid(linear_model)
            
            # Compute gradients
            if sparse_X:
                dw = (1 / n_samples) * np.array(X.T.dot(y_predicted - y)).ravel()
            else:
                dw = (1 / n_samples) * np.dot(X.T, (y_predicted - y))
            db = (1 / n_samples) * np.sum(y_predicted - y)
            
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
            linear_model = np.array(X.dot(self.weights)).ravel() + self.bias
        else:
            linear_model = np.dot(X, self.weights) + self.bias
        
        # Apply sigmoid
        y_predicted_prob = self._sigmoid(linear_model)
        
        # Threshold at 0.5 for binary classification
        y_predicted = (y_predicted_prob >= 0.5).astype(int)
        
        return y_predicted

def train_logistic_regression(X_train, y_train, learning_rate=0.01, max_iter=1000):
    """
    Train a logistic regression model.
    
    This function creates an instance of LogisticRegression, fits it to the training data,
    and returns the trained model.
    
    Args:
        X_train (array-like or sparse matrix): Training features.
        y_train (array-like): Training labels.
        learning_rate (float): Learning rate for gradient descent.
        max_iter (int): Maximum iterations for gradient descent.
    
    Returns:
        LogisticRegression: Trained model instance.
    """
    model = LogisticRegression(learning_rate=learning_rate, max_iter=max_iter)
    model.fit(X_train, y_train)
    return model