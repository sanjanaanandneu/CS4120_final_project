"""Benchmark fine-tuned model against TF-IDF/n-gram + LR/SVM baselines.

Usage (from project root):
    python scripts/run_comparison.py
    python scripts/run_comparison.py --checkpoint src/models/distilroberta-base

Requires pre-extracted features in ``data/processed/features/hc3/``
(produced by ``scripts/run_feature_extraction.py``) and a fine-tuned
checkpoint produced by ``scripts/run_finetune.py``.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import scipy.sparse as sp
import torch
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.data.preprocess import load_dataset_splits
from src.models.pretrained_classifier import PretrainedClassifier
from src.data.hc3_dataset import HC3Dataset
from src.models.logistic_regression import LogisticRegression
from src.models.svm import LinearSVC
from src.evaluation.metrics import (
    compute_metrics,
    compute_domain_metrics,
    build_results_table,
    print_report,
)

FEATURES_DIR = Path("data/processed/features/hc3")

# Baseline helpers

def _load_sparse(name: str) -> tuple[sp.csr_matrix, sp.csr_matrix]:
    train_path = FEATURES_DIR / f"{name}_train.npz"
    test_path = FEATURES_DIR / f"{name}_test.npz"
    if not train_path.exists() or not test_path.exists():
        raise FileNotFoundError(
            f"Feature files not found for '{name}'. "
            "Run scripts/run_feature_extraction.py first."
        )
    return sp.load_npz(str(train_path)), sp.load_npz(str(test_path))


def _load_dense(name: str) -> tuple[np.ndarray, np.ndarray]:
    train_path = FEATURES_DIR / f"{name}_train.npy"
    test_path = FEATURES_DIR / f"{name}_test.npy"
    if not train_path.exists() or not test_path.exists():
        raise FileNotFoundError(
            f"Embedding files not found for '{name}'. "
            "Run scripts/run_feature_extraction.py first."
        )
    return np.load(train_path), np.load(test_path)


def _run_baseline(
    feature_name: str,
    model_type: str,
    X_train,
    X_test,
    y_train: np.ndarray,
    y_test: np.ndarray,
) -> dict[str, float]:
    if model_type == "svm":
        print(f"\n  Training SVM on {feature_name} features …")
        model = LinearSVC()
        model.fit(X_train, y_train)
    else:
        print(f"\n  Training Logistic Regression on {feature_name} features …")
        model = LogisticRegression()
        model.fit(X_train, y_train)
    preds = model.predict(X_test)
    return compute_metrics(y_test, preds)


# ---------------------------------------------------------------------------
# Fine-tuned model inference
# ---------------------------------------------------------------------------

@torch.no_grad()
def _run_finetuned(
    checkpoint_dir: str | Path,
    test_df,
) -> tuple[np.ndarray, np.ndarray]:
    classifier = PretrainedClassifier.load(checkpoint_dir)
    classifier.eval()

    test_ds = HC3Dataset(test_df, classifier.tokenizer, max_length=512)
    test_loader = DataLoader(test_ds, batch_size=32, shuffle=False)

    all_preds: list[int] = []
    all_proba: list[np.ndarray] = []

    for batch in test_loader:
        input_ids = batch["input_ids"].to(classifier.device)
        attention_mask = batch["attention_mask"].to(classifier.device)
        proba = classifier.predict_proba(input_ids, attention_mask)
        all_proba.append(proba)
        all_preds.extend(proba.argmax(axis=1).tolist())

    return np.array(all_preds), np.vstack(all_proba)

# Main

def main() -> None:
    parser = argparse.ArgumentParser(description="Compare fine-tuned model vs. baselines")
    parser.add_argument(
        "--checkpoint",
        default=None,
        help="Path to fine-tuned checkpoint directory (skip if not available)",
    )
    args = parser.parse_args()

    train_df, test_df = load_dataset_splits("hc3")
    y_train = train_df["label"].values
    y_test = test_df["label"].values
    sources_test = test_df["source"].values

    all_results: dict[str, dict[str, float]] = {}

    sparse_features = [
        ("tfidf", "TF-IDF"),
        ("word_ngram", "Word N-gram"),
        ("char_ngram", "Char N-gram"),
    ]

    for feature_key, feature_label in sparse_features:
        try:
            X_train_sp, X_test_sp = _load_sparse(feature_key)
            for model_type, model_label in [("lr", "LR"), ("svm", "SVM")]:
                label = f"{feature_label} + {model_label}"
                all_results[label] = _run_baseline(
                    feature_label, model_type, X_train_sp, X_test_sp, y_train, y_test
                )
        except FileNotFoundError as e:
            print(f"  Skipping {feature_label} baselines: {e}")

    # Embeddings + LR and SVM
    for emb_name, emb_label in [
        ("distilbert_embeddings", "DistilBERT Embeddings"),
        ("bert_embeddings", "BERT Embeddings"),
    ]:
        try:
            X_train_emb, X_test_emb = _load_dense(emb_name)
            for model_type, model_label in [("lr", "LR"), ("svm", "SVM")]:
                label = f"{emb_label} + {model_label}"
                all_results[label] = _run_baseline(
                    emb_label, model_type, X_train_emb, X_test_emb, y_train, y_test
                )
            break  # use whichever embedding file exists
        except FileNotFoundError:
            continue

    # Fine-tuned model
    checkpoint = args.checkpoint
    if checkpoint is None:
        # Auto-detect the most recently modified checkpoint saved by run_finetune.py
        model_dirs = sorted(
            Path("models").glob("**/config.json"),
            key=lambda p: p.stat().st_mtime,
            reverse=True,
        )
        if model_dirs:
            checkpoint = str(model_dirs[0].parent)
            print(f"\n  Auto-detected checkpoint: {checkpoint}")

    if checkpoint and Path(checkpoint).exists():
        print(f"\n  Running fine-tuned model inference from {checkpoint} …")
        preds_ft, proba_ft = _run_finetuned(checkpoint, test_df)
        model_label = f"Fine-tuned ({Path(checkpoint).name})"
        metrics_ft = compute_metrics(y_test, preds_ft, proba_ft)
        domain_metrics_ft = compute_domain_metrics(y_test, preds_ft, sources_test, proba_ft)
        all_results[model_label] = metrics_ft
        print_report(metrics_ft, domain_metrics_ft, title=f"Fine-tuned Model — {Path(checkpoint).name}")
    else:
        print("\n  No fine-tuned checkpoint found. Run scripts/run_finetune.py first.")

    # Summary table
    if all_results:
        table = build_results_table(all_results)
        print("\n" + "=" * 60)
        print("  Model Comparison Summary")
        print("=" * 60)
        print(table.to_string(float_format="{:.4f}".format))
        print("=" * 60)


if __name__ == "__main__":
    main()
