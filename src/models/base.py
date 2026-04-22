"""Abstract base class for all models in this project."""

from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np


class BaseModel(ABC):
    """Common interface that every model must implement.

    Subclasses must implement:
        fit        -- train the model on (X, y)
        predict    -- return binary predictions for X
        save       -- persist the model to a filepath
        load       -- classmethod; load a model from a filepath
    """

    @abstractmethod
    def fit(self, X, y, **kwargs) -> None:
        """Train the model.

        Parameters
        ----------
        X:
            Feature matrix or list of raw texts, depending on the model.
        y:
            1-D array of binary labels (0 = human, 1 = AI).
        **kwargs:
            Model-specific hyperparameters (e.g. epochs, learning_rate).
        """

    @abstractmethod
    def predict(self, X) -> np.ndarray:
        """Return binary predictions.

        Parameters
        ----------
        X:
            Same format as the X passed to fit().

        Returns
        -------
        np.ndarray of shape (n_samples,) with values in {0, 1}.
        """

    @abstractmethod
    def save(self, filepath: str) -> None:
        """Persist the model to filepath.

        Parameters
        ----------
        filepath:
            Destination path. The exact file format is determined by each
            subclass (joblib pickle, torch checkpoint, HF directory, etc.).
        """

    @classmethod
    @abstractmethod
    def load(cls, filepath: str) -> "BaseModel":
        """Load a model from filepath.

        Parameters
        ----------
        filepath:
            Path previously written by save().

        Returns
        -------
        A fully initialised instance of the subclass.
        """
