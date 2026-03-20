"""Extract TF-IDF, n-gram, and transformer embedding features for both datasets.

Usage (from project root):
    python scripts/run_feature_extraction.py

All feature matrices are saved to ``data/processed/features/``.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import scipy.sparse as sp

# Ensure project root is on the path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.data.preprocess import load_dataset_splits
from src.features.tfidf import TFIDFExtractor
from src.features.ngrams import NgramExtractor
from src.features.embeddings import EmbeddingExtractor

FEATURES_DIR = Path("data/processed/features")


def save_sparse(matrix: sp.csr_matrix, path: Path) -> None:
    """Save a scipy sparse matrix to *path* as ``.npz``."""
    path.parent.mkdir(parents=True, exist_ok=True)
    sp.save_npz(str(path), matrix)
    print(f"  Saved sparse matrix {matrix.shape} → {path}")


def run_tfidf(
    dataset: str,
    train_texts: list[str],
    test_texts: list[str],
) -> None:
    print(f"\n--- TF-IDF: {dataset} ---")
    extractor = TFIDFExtractor(max_features=50_000, ngram_range=(1, 2))
    train_mat = extractor.fit_transform(train_texts)
    test_mat = extractor.transform(test_texts)

    save_sparse(train_mat, FEATURES_DIR / dataset / "tfidf_train.npz")
    save_sparse(test_mat, FEATURES_DIR / dataset / "tfidf_test.npz")
    extractor.save(FEATURES_DIR / dataset / "tfidf_vectorizer.joblib")


def run_ngrams(
    dataset: str,
    train_texts: list[str],
    test_texts: list[str],
) -> None:
    for analyzer, label in [("word", "word_ngram"), ("char_wb", "char_ngram")]:
        print(f"\n--- {label}: {dataset} ---")
        extractor = NgramExtractor(
            analyzer=analyzer,
            ngram_range=(2, 4) if analyzer == "char_wb" else (1, 3),
            max_features=50_000,
        )
        train_mat = extractor.fit_transform(train_texts)
        test_mat = extractor.transform(test_texts)

        save_sparse(train_mat, FEATURES_DIR / dataset / f"{label}_train.npz")
        save_sparse(test_mat, FEATURES_DIR / dataset / f"{label}_test.npz")
        extractor.save(FEATURES_DIR / dataset / f"{label}_vectorizer.joblib")


def run_embeddings(
    dataset: str,
    train_texts: list[str],
    test_texts: list[str],
) -> None:
    print(f"\n--- Embeddings: {dataset} ---")
    extractor = EmbeddingExtractor(
        model_name="bert-base-uncased",
        batch_size=32,
    )
    print("  Extracting train embeddings …")
    train_emb = extractor.extract(train_texts)
    print("  Extracting test embeddings …")
    test_emb = extractor.extract(test_texts)

    extractor.save_embeddings(train_emb, FEATURES_DIR / dataset / "bert_embeddings_train.npy")
    extractor.save_embeddings(test_emb, FEATURES_DIR / dataset / "bert_embeddings_test.npy")


def main() -> None:
    FEATURES_DIR.mkdir(parents=True, exist_ok=True)

    datasets = ["hc3"]

    for dataset in datasets:
        print("\n" + "=" * 60)
        print(f"Dataset: {dataset.upper()}")
        print("=" * 60)

        train_df, test_df = load_dataset_splits(dataset)
        train_texts = train_df["text"].tolist()
        test_texts = test_df["text"].tolist()

        run_tfidf(dataset, train_texts, test_texts)
        run_ngrams(dataset, train_texts, test_texts)
        run_embeddings(dataset, train_texts, test_texts)

    print("\n" + "=" * 60)
    print("Feature extraction complete. Saved files:")
    for f in sorted(FEATURES_DIR.rglob("*")):
        if f.is_file():
            size_mb = f.stat().st_size / 1_048_576
            print(f"  {f}  ({size_mb:.1f} MB)")
    print("=" * 60)


if __name__ == "__main__":
    main()
