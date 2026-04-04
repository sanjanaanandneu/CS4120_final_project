import argparse
import os
import scipy.sparse as sp
import json

import sys
from pathlib import Path

# Ensure project root is on the path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.data.preprocess import load_dataset_splits
from src.models.logistic_regression import train_logistic_regression
from src.models.svm import train_linear_svc
from src.utils.evaluation import evaluate_model

def main():
    parser = argparse.ArgumentParser(description="Train and save a single model on a single dataset.")
    parser.add_argument('--model', required=True, choices=['lr', 'svm'], help='Model type: lr for Logistic Regression, svm for SVM')
    parser.add_argument('--dataset', required=True, choices=['hc3', 'turingbench', 'combined'], help='Dataset to use')
    parser.add_argument('--features', required=True, choices=['tfidf', 'word_ngram'], help='Feature type to use')
    parser.add_argument('--learning_rate', type=float, default=0.01, help='Learning rate (default: 0.01)')
    parser.add_argument('--max_iter', type=int, default=1000, help='Maximum iterations (default: 1000)')
    
    args = parser.parse_args()
    
    # Load dataset
    train_df, test_df = load_dataset_splits(args.dataset)
    y_train = train_df["label"].values
    
    # Load features
    feature_path = f"data/processed/features/{args.dataset}/{args.features}_train.npz"
    X_train = sp.load_npz(feature_path)
    
    # Train model
    if args.model == 'lr':
        model = train_logistic_regression(X_train, y_train, learning_rate=args.learning_rate, max_iter=args.max_iter)
    elif args.model == 'svm':
        model = train_linear_svc(X_train, y_train, learning_rate=args.learning_rate, max_iter=args.max_iter)
    
    # Create saved_models directory if it doesn't exist
    os.makedirs('saved_models', exist_ok=True)
    
    # Save model
    filename = f"{args.model}_{args.dataset}_{args.features}.pkl"
    filepath = os.path.join('saved_models', filename)
    model.save(filepath)
    
    print(f"Model trained and saved to {filepath}")
    
    # Load test data
    y_test = test_df["label"].values
    test_feature_path = f"data/processed/features/{args.dataset}/{args.features}_test.npz"
    X_test = sp.load_npz(test_feature_path)
    
    # Evaluate model
    metrics = evaluate_model(model, X_test, y_test)
    
    # Create metrics directory if it doesn't exist
    os.makedirs('metrics', exist_ok=True)
    
    # Save metrics
    metrics_filename = f"{args.model}_{args.dataset}_{args.features}_metrics.json"
    metrics_filepath = os.path.join('metrics', metrics_filename)
    with open(metrics_filepath, 'w') as f:
        json.dump(metrics, f, indent=4)
    
    print(f"Metrics saved to {metrics_filepath}")
    print("Evaluation Results:")
    for key, value in metrics.items():
        print(f"  {key.capitalize()}: {value:.4f}")

if __name__ == "__main__":
    main()