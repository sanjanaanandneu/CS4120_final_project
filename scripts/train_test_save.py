import argparse
import json
import logging
import os
import sys
from pathlib import Path

import numpy as np
import scipy.sparse as sp

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.data.preprocess import load_dataset_splits
from src.evaluation.metrics import evaluate_model, print_report
from src.models.logistic_regression import LogisticRegression
from src.models.svm import LinearSVC
from src.models.lstm import LSTMClassifier
from src.models.cnn import CNN
from src.models.rnn import RNN

logging.basicConfig(format="%(levelname)s: %(message)s", level=logging.INFO)
log = logging.getLogger(__name__)

SPARSE_FEATURES = {"tfidf", "word_ngram", "char_ngram"}
DENSE_FEATURES = {"bert_embeddings"}
SEQUENCE_FEATURES = {"word2vec_embeddings"}

VALID_FEATURES = {
    "lr":   SPARSE_FEATURES | DENSE_FEATURES,
    "svm":  SPARSE_FEATURES | DENSE_FEATURES,
    "lstm": SEQUENCE_FEATURES,
    "cnn":  SEQUENCE_FEATURES,
    "rnn":  SEQUENCE_FEATURES,
}


def main():
    parser = argparse.ArgumentParser(
        description="Train, evaluate, and save a model on a dataset."
    )
    parser.add_argument(
        "--model", required=True,
        choices=["lr", "svm", "lstm", "cnn", "rnn"],
        help="Model type",
    )
    parser.add_argument(
        "--dataset", required=True,
        choices=["hc3", "turingbench", "combined"],
        help="Dataset to use",
    )
    parser.add_argument(
        "--features", required=True,
        choices=["tfidf", "word_ngram", "char_ngram", "bert_embeddings", "word2vec_embeddings"],
        help="Feature type to use",
    )
    parser.add_argument("--learning_rate", type=float, default=0.01, help="Learning rate")
    parser.add_argument("--max_iter", type=int, default=1000, help="Max iterations (lr/svm only)")
    parser.add_argument("--epochs", type=int, default=10, help="Training epochs (neural models only)")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size (neural models only)")
    args = parser.parse_args()

    # Validate feature/model combination
    if args.features not in VALID_FEATURES[args.model]:
        log.error(
            "Feature type '%s' is not compatible with model '%s'. "
            "Valid features for '%s': %s",
            args.features, args.model, args.model,
            sorted(VALID_FEATURES[args.model]),
        )
        sys.exit(1)

    # Load labels
    train_df, test_df = load_dataset_splits(args.dataset)
    y_train = train_df["label"].values
    y_test = test_df["label"].values

    # Load features
    feat_dir = Path("data/processed/features") / args.dataset
    if args.features in SPARSE_FEATURES:
        X_train = sp.load_npz(feat_dir / f"{args.features}_train.npz")
        X_test = sp.load_npz(feat_dir / f"{args.features}_test.npz")
    else:
        X_train = np.load(feat_dir / f"{args.features}_train.npy")
        X_test = np.load(feat_dir / f"{args.features}_test.npy")

    # Instantiate model
    if args.model == "lr":
        model = LogisticRegression(learning_rate=args.learning_rate, max_iter=args.max_iter)
    elif args.model == "svm":
        model = LinearSVC(learning_rate=args.learning_rate, max_iter=args.max_iter)
    elif args.model == "lstm":
        model = LSTMClassifier(input_size=X_train.shape[2], hidden_size=128, num_layers=2)
    elif args.model == "cnn":
        model = CNN(embedding_dim=X_train.shape[2])
    elif args.model == "rnn":
        model = RNN(embedding_dim=X_train.shape[2])

    # Train
    if args.model in {"lr", "svm"}:
        model.fit(X_train, y_train)
    elif args.model == "lstm":
        model.fit(
            X_train, y_train,
            learning_rate=args.learning_rate,
            batch_size=args.batch_size,
            num_epochs=args.epochs,
        )
    else:
        model.fit(
            X_train, y_train,
            epochs=args.epochs,
            batch_size=args.batch_size,
            lr=args.learning_rate,
        )

    # Save model
    os.makedirs("saved_models", exist_ok=True)
    model_path = f"saved_models/{args.model}_{args.dataset}_{args.features}.pkl"
    model.save(model_path)
    log.info("Model saved → %s", model_path)

    # Evaluate
    metrics = evaluate_model(model, X_test, y_test)
    print_report(metrics, title=f"{args.model.upper()} | {args.dataset} | {args.features}")

    # Save metrics
    os.makedirs("metrics", exist_ok=True)
    metrics_path = f"metrics/{args.model}_{args.dataset}_{args.features}_metrics.json"
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=4)
    log.info("Metrics saved → %s", metrics_path)


if __name__ == "__main__":
    main()
