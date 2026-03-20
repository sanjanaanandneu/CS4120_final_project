"""Transformer-based sentence embedding extraction via mean pooling."""

from __future__ import annotations

from pathlib import Path
from typing import Iterable

import numpy as np
import torch
from transformers import AutoModel, AutoTokenizer


class EmbeddingExtractor:
    """Extract fixed-size sentence embeddings from a pretrained transformer.

    Uses mean pooling over the last hidden state (ignoring padding tokens) to
    produce one vector per input text.

    Parameters
    ----------
    model_name:
        HuggingFace model identifier (e.g. ``"bert-base-uncased"``).
    batch_size:
        Number of texts to process per forward pass.
    max_length:
        Maximum token sequence length (inputs are truncated if longer).
    device:
        ``"cuda"``, ``"mps"``, ``"cpu"``, or ``None`` to auto-detect.
    """

    def __init__(
        self,
        model_name: str = "bert-base-uncased",
        batch_size: int = 32,
        max_length: int = 512,
        device: str | None = None,
    ) -> None:
        self.model_name = model_name
        self.batch_size = batch_size
        self.max_length = max_length

        if device is None:
            if torch.cuda.is_available():
                device = "cuda"
            elif torch.backends.mps.is_available():
                device = "mps"
            else:
                device = "cpu"
        self.device = device
        print(f"  EmbeddingExtractor using device: {self.device}")

        print(f"  Loading tokenizer and model '{model_name}' …")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name)
        self.model.eval()
        self.model.to(self.device)

    def _mean_pool(
        self,
        token_embeddings: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> torch.Tensor:
        """Mean-pool *token_embeddings* while masking padding tokens."""
        mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
        sum_embeddings = torch.sum(token_embeddings * mask_expanded, dim=1)
        sum_mask = torch.clamp(mask_expanded.sum(dim=1), min=1e-9)
        return sum_embeddings / sum_mask

    @torch.no_grad()
    def extract(self, texts: list[str]) -> np.ndarray:
        """Extract embeddings for a list of texts.

        Parameters
        ----------
        texts:
            List of raw strings.

        Returns
        -------
        np.ndarray of shape ``(len(texts), hidden_size)``
        """
        all_embeddings: list[np.ndarray] = []
        total = len(texts)

        for start in range(0, total, self.batch_size):
            batch = texts[start : start + self.batch_size]
            encoded = self.tokenizer(
                batch,
                padding=True,
                truncation=True,
                max_length=self.max_length,
                return_tensors="pt",
            )
            encoded = {k: v.to(self.device) for k, v in encoded.items()}

            outputs = self.model(**encoded)
            embeddings = self._mean_pool(
                outputs.last_hidden_state, encoded["attention_mask"]
            )
            all_embeddings.append(embeddings.cpu().numpy())

            done = min(start + self.batch_size, total)
            print(f"    Embedded {done}/{total} samples …", end="\r", flush=True)

        print()  # newline after progress
        result = np.vstack(all_embeddings)
        print(f"  Embeddings shape: {result.shape}")
        return result

    def save_embeddings(self, embeddings: np.ndarray, path: str | Path) -> None:
        """Save *embeddings* array to *path* as a ``.npy`` file.

        Parameters
        ----------
        embeddings:
            NumPy array to save.
        path:
            Destination file path (should end in ``.npy``).
        """
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        np.save(path, embeddings)
        print(f"  Saved embeddings {embeddings.shape} → {path}")

    @staticmethod
    def load_embeddings(path: str | Path) -> np.ndarray:
        """Load embeddings previously saved with :meth:`save_embeddings`.

        Parameters
        ----------
        path:
            Path to a ``.npy`` file.

        Returns
        -------
        np.ndarray
        """
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"Embeddings file not found: {path}")
        embeddings = np.load(path)
        print(f"  Loaded embeddings {embeddings.shape} from {path}")
        return embeddings
