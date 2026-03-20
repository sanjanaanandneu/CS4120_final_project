"""Character-level and word-level n-gram feature extraction."""

from __future__ import annotations

from pathlib import Path
from typing import Iterable, Literal

import joblib
import scipy.sparse as sp
from sklearn.feature_extraction.text import CountVectorizer


class NgramExtractor:
    """N-gram feature extractor backed by :class:`sklearn.feature_extraction.text.CountVectorizer`.

    Parameters
    ----------
    analyzer:
        ``"word"`` for word n-grams or ``"char_wb"`` for character n-grams
        (character n-grams bounded by word boundaries).
    ngram_range:
        ``(min_n, max_n)`` for the n-gram range.
    max_features:
        Maximum vocabulary size.
    min_df:
        Minimum document frequency for a term to be kept.
    max_df:
        Maximum document frequency threshold.
    """

    def __init__(
        self,
        analyzer: Literal["word", "char", "char_wb"] = "word",
        ngram_range: tuple[int, int] = (1, 3),
        max_features: int = 50_000,
        min_df: int = 2,
        max_df: float = 0.95,
    ) -> None:
        self.analyzer = analyzer
        self.vectorizer = CountVectorizer(
            analyzer=analyzer,
            ngram_range=ngram_range,
            max_features=max_features,
            min_df=min_df,
            max_df=max_df,
            binary=False,
        )
        self._fitted = False

    def fit_transform(self, train_texts: Iterable[str]) -> sp.csr_matrix:
        """Fit on *train_texts* and return the count matrix.

        Parameters
        ----------
        train_texts:
            Iterable of raw training strings.

        Returns
        -------
        scipy.sparse.csr_matrix of shape ``(n_samples, n_features)``
        """
        print(f"  Fitting {self.analyzer} n-gram extractor "
              f"(ngram_range={self.vectorizer.ngram_range}, "
              f"max_features={self.vectorizer.max_features}) …")
        matrix = self.vectorizer.fit_transform(train_texts)
        self._fitted = True
        print(f"  Vocabulary size: {len(self.vectorizer.vocabulary_)}")
        print(f"  Train matrix shape: {matrix.shape}")
        return matrix

    def transform(self, texts: Iterable[str]) -> sp.csr_matrix:
        """Transform *texts* using the already-fitted vectorizer.

        Parameters
        ----------
        texts:
            Iterable of raw strings.

        Returns
        -------
        scipy.sparse.csr_matrix
        """
        if not self._fitted:
            raise RuntimeError("Call fit_transform() before transform().")
        matrix = self.vectorizer.transform(texts)
        print(f"  Transform matrix shape: {matrix.shape}")
        return matrix

    def save(self, path: str | Path) -> None:
        """Persist the fitted vectorizer to *path* with joblib.

        Parameters
        ----------
        path:
            Destination file path.
        """
        if not self._fitted:
            raise RuntimeError("Nothing to save — vectorizer has not been fitted yet.")
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self.vectorizer, path)
        print(f"  Saved {self.analyzer} n-gram vectorizer → {path}")

    @classmethod
    def load(cls, path: str | Path) -> "NgramExtractor":
        """Load a previously saved vectorizer from *path*.

        Parameters
        ----------
        path:
            Path to a joblib file created by :meth:`save`.

        Returns
        -------
        NgramExtractor
        """
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"Vectorizer file not found: {path}")
        extractor = cls.__new__(cls)
        extractor.vectorizer = joblib.load(path)
        extractor.analyzer = extractor.vectorizer.analyzer
        extractor._fitted = True
        print(f"  Loaded n-gram vectorizer from {path}")
        return extractor
