"""TF-IDF feature extraction using scikit-learn's TfidfVectorizer."""

from __future__ import annotations

from pathlib import Path
from typing import Iterable

import joblib
import numpy as np
import scipy.sparse as sp
from sklearn.feature_extraction.text import TfidfVectorizer


class TFIDFExtractor:
    """Wrapper around :class:`sklearn.feature_extraction.text.TfidfVectorizer`.

    Parameters
    ----------
    max_features:
        Maximum number of vocabulary terms.
    ngram_range:
        ``(min_n, max_n)`` for n-gram extraction.
    max_df:
        Ignore terms with document frequency higher than this threshold.
    min_df:
        Ignore terms with document frequency lower than this threshold.
    """

    def __init__(
        self,
        max_features: int = 50_000,
        ngram_range: tuple[int, int] = (1, 2),
        max_df: float = 0.95,
        min_df: int = 2,
    ) -> None:
        self.vectorizer = TfidfVectorizer(
            max_features=max_features,
            ngram_range=ngram_range,
            max_df=max_df,
            min_df=min_df,
            sublinear_tf=True,
        )
        self._fitted = False

    def fit_transform(self, train_texts: Iterable[str]) -> sp.csr_matrix:
        """Fit the vectorizer on *train_texts* and return the feature matrix.

        Parameters
        ----------
        train_texts:
            Iterable of raw training strings.

        Returns
        -------
        scipy.sparse.csr_matrix of shape ``(n_samples, n_features)``
        """
        print(f"  Fitting TF-IDF vectorizer (max_features={self.vectorizer.max_features}, "
              f"ngram_range={self.vectorizer.ngram_range}) …")
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
            File path (e.g. ``data/processed/features/tfidf_vectorizer.joblib``).
        """
        if not self._fitted:
            raise RuntimeError("Nothing to save — vectorizer has not been fitted yet.")
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self.vectorizer, path)
        print(f"  Saved TF-IDF vectorizer → {path}")

    @classmethod
    def load(cls, path: str | Path) -> "TFIDFExtractor":
        """Load a previously saved vectorizer from *path*.

        Parameters
        ----------
        path:
            Path to a joblib file created by :meth:`save`.

        Returns
        -------
        TFIDFExtractor
            Instance with the loaded vectorizer ready for :meth:`transform`.
        """
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"Vectorizer file not found: {path}")
        extractor = cls.__new__(cls)
        extractor.vectorizer = joblib.load(path)
        extractor._fitted = True
        print(f"  Loaded TF-IDF vectorizer from {path}")
        return extractor
