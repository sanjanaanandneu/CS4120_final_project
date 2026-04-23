"""Extract features for one or more datasets and feature types.

Usage (from project root):
    python scripts/run_feature_extraction.py --datasets hc3 combined --features tfidf word2vec
    python scripts/run_feature_extraction.py --datasets hc3 --features all

All feature matrices are saved to ``data/processed/features/{dataset}/``.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import scipy.sparse as sp

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.data.preprocess import load_dataset_splits
from src.features.tfidf import TFIDFExtractor
from src.features.ngrams import NgramExtractor
from src.features.embeddings import EmbeddingExtractor
from src.features.word2vec import Word2VecExtractor

FEATURES_DIR = Path("data/processed/features")

VALID_DATASETS = ["hc3", "turingbench", "combined"]
VALID_FEATURES = ["tfidf", "ngrams", "bert_embeddings", "word2vec"]


def save_sparse(matrix: sp.csr_matrix, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    sp.save_npz(str(path), matrix)
    print(f"  Saved sparse matrix {matrix.shape} → {path}")


def run_tfidf(dataset: str, train_texts: list[str], test_texts: list[str]) -> None:
    print(f"\n--- TF-IDF: {dataset} ---")
    extractor = TFIDFExtractor(max_features=50_000, ngram_range=(1, 2))
    train_mat = extractor.fit_transform(train_texts)
    test_mat = extractor.transform(test_texts)
    save_sparse(train_mat, FEATURES_DIR / dataset / "tfidf_train.npz")
    save_sparse(test_mat, FEATURES_DIR / dataset / "tfidf_test.npz")


def run_ngrams(dataset: str, train_texts: list[str], test_texts: list[str]) -> None:
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


def run_embeddings(dataset: str, train_texts: list[str], test_texts: list[str]) -> None:
    print(f"\n--- BERT Embeddings: {dataset} ---")
    extractor = EmbeddingExtractor(model_name="bert-base-uncased", batch_size=32)
    print("  Extracting train embeddings …")
    train_emb = extractor.extract(train_texts)
    print("  Extracting test embeddings …")
    test_emb = extractor.extract(test_texts)
    extractor.save_embeddings(train_emb, FEATURES_DIR / dataset / "bert_embeddings_train.npy")
    extractor.save_embeddings(test_emb, FEATURES_DIR / dataset / "bert_embeddings_test.npy")


def run_word2vec(dataset: str, train_texts: list[str], test_texts: list[str]) -> None:
    print(f"\n--- Word2Vec: {dataset} ---")
    extractor = Word2VecExtractor(model_name="glove-wiki-gigaword-100")
    extractor.fit(train_texts)
    print("  Extracting train embeddings …")
    extractor.transform_and_save(
        train_texts, FEATURES_DIR / dataset / "word2vec_embeddings_train.npy", max_len=100
    )
    print("  Extracting test embeddings …")
    extractor.transform_and_save(
        test_texts, FEATURES_DIR / dataset / "word2vec_embeddings_test.npy", max_len=100
    )


FEATURE_RUNNERS = {
    "tfidf": run_tfidf,
    "ngrams": run_ngrams,
    "bert_embeddings": run_embeddings,
    "word2vec": run_word2vec,
}


def main() -> None:
    parser = argparse.ArgumentParser(description="Extract features for AI-text detection models")
    parser.add_argument(
        "--datasets",
        nargs="+",
        choices=VALID_DATASETS,
        default=["hc3"],
        metavar="DATASET",
        help=f"Datasets to process: {{{', '.join(VALID_DATASETS)}}} (default: hc3)",
    )
    parser.add_argument(
        "--features",
        nargs="+",
        choices=VALID_FEATURES + ["all"],
        default=["all"],
        metavar="FEATURE",
        help=f"Feature types to extract: {{{', '.join(VALID_FEATURES)}, all}} (default: all)",
    )
    args = parser.parse_args()

    datasets = args.datasets
    features = VALID_FEATURES if "all" in args.features else args.features

    FEATURES_DIR.mkdir(parents=True, exist_ok=True)

    for dataset in datasets:
        print("\n" + "=" * 60)
        print(f"Dataset: {dataset.upper()}")
        print("=" * 60)

        train_df, test_df = load_dataset_splits(dataset)
        train_texts = train_df["text"].tolist()
        test_texts = test_df["text"].tolist()

        out_dir = FEATURES_DIR / dataset
        out_dir.mkdir(parents=True, exist_ok=True)
        np.save(out_dir / "y_train.npy", train_df["label"].values)
        np.save(out_dir / "y_test.npy", test_df["label"].values)

        for feature in features:
            FEATURE_RUNNERS[feature](dataset, train_texts, test_texts)

    print("\n" + "=" * 60)
    print("Feature extraction complete. Saved files:")
    for f in sorted(FEATURES_DIR.rglob("*")):
        if f.is_file():
            size_mb = f.stat().st_size / 1_048_576
            print(f"  {f}  ({size_mb:.1f} MB)")
    print("=" * 60)


if __name__ == "__main__":
    main()
