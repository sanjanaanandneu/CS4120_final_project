import numpy as np
from scipy import sparse

def _compute_confusion_matrix(y_true, y_pred):
    """
    Compute the confusion matrix elements for binary classification.
    
    Args:
        y_true (array-like): True labels (0 or 1).
        y_pred (array-like): Predicted labels (0 or 1).
    
    Returns:
        tuple: (TP, FP, TN, FN) where TP=True Positives, etc.
    """
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    
    TP = np.sum((y_true == 1) & (y_pred == 1))
    FP = np.sum((y_true == 0) & (y_pred == 1))
    TN = np.sum((y_true == 0) & (y_pred == 0))
    FN = np.sum((y_true == 1) & (y_pred == 0))
    
    return TP, FP, TN, FN

def compute_accuracy(y_true, y_pred):
    """
    Compute accuracy: (TP + TN) / (TP + TN + FP + FN).
    
    Args:
        y_true (array-like): True labels.
        y_pred (array-like): Predicted labels.
    
    Returns:
        float: Accuracy score.
    """
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    
    correct = np.sum(y_true == y_pred)
    total = len(y_true)
    
    return correct / total if total > 0 else 0.0

def compute_precision(y_true, y_pred):
    """
    Compute precision: TP / (TP + FP).
    
    Args:
        y_true (array-like): True labels.
        y_pred (array-like): Predicted labels.
    
    Returns:
        float: Precision score (0.0 if no positive predictions).
    """
    TP, FP, _, _ = _compute_confusion_matrix(y_true, y_pred)
    
    return TP / (TP + FP) if (TP + FP) > 0 else 0.0

def compute_recall(y_true, y_pred):
    """
    Compute recall: TP / (TP + FN).
    
    Args:
        y_true (array-like): True labels.
        y_pred (array-like): Predicted labels.
    
    Returns:
        float: Recall score (0.0 if no true positives).
    """
    TP, _, _, FN = _compute_confusion_matrix(y_true, y_pred)
    
    return TP / (TP + FN) if (TP + FN) > 0 else 0.0

def compute_f1(y_true, y_pred):
    """
    Compute F1 score: 2 * (precision * recall) / (precision + recall).
    
    Args:
        y_true (array-like): True labels.
        y_pred (array-like): Predicted labels.
    
    Returns:
        float: F1 score (0.0 if precision + recall == 0).
    """
    precision = compute_precision(y_true, y_pred)
    recall = compute_recall(y_true, y_pred)
    
    return 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0

def evaluate_model(model, X_test, y_true):
    """
    Evaluate a model by computing predictions and all metrics.
    
    Args:
        model: Model instance with a predict method.
        X_test (array-like or sparse matrix): Test features.
        y_true (array-like): True labels.
    
    Returns:
        dict: Dictionary with 'accuracy', 'precision', 'recall', 'f1'.
    """
    # Get predictions
    y_pred = model.predict(X_test)
    
    # Compute metrics
    accuracy = compute_accuracy(y_true, y_pred)
    precision = compute_precision(y_true, y_pred)
    recall = compute_recall(y_true, y_pred)
    f1 = compute_f1(y_true, y_pred)
    
    return {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1
    }