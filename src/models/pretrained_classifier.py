"""Fine-tuning wrapper for HuggingFace sequence classification models."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from transformers import AutoModelForSequenceClassification, AutoTokenizer


def _resolve_device(device: str | None) -> str:
    if device is not None:
        return device
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


class PretrainedClassifier(nn.Module):
    """Sequence classifier built on top of a pretrained HuggingFace encoder.

    Wraps ``AutoModelForSequenceClassification`` (which attaches a linear
    classification head on the pooled output) and exposes convenience methods
    for saving/loading checkpoints and computing class probabilities.

    Parameters
    ----------
    model_name:
        HuggingFace model identifier, e.g. ``"distilroberta-base"``.
    num_labels:
        Number of output classes (2 for binary AI-detection).
    device:
        ``"cuda"``, ``"mps"``, ``"cpu"``, or ``None`` to auto-detect.
    """

    def __init__(
        self,
        model_name: str = "distilroberta-base",
        num_labels: int = 2,
        device: str | None = None,
    ) -> None:
        super().__init__()
        self.model_name = model_name
        self.num_labels = num_labels
        self.device = _resolve_device(device)

        print(f"  Loading tokenizer: {model_name}")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)

        print(f"  Loading model: {model_name}  (num_labels={num_labels})")
        self.model = AutoModelForSequenceClassification.from_pretrained(
            model_name,
            num_labels=num_labels,
        )
        self.model.to(self.device)
        print(f"  Device: {self.device}")

    # ------------------------------------------------------------------
    # Forward pass
    # ------------------------------------------------------------------

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        labels: torch.Tensor | None = None,
    ):
        """Run a forward pass through the underlying HF model.

        Parameters
        ----------
        input_ids, attention_mask:
            Token tensors from the tokenizer, shape ``(batch, seq_len)``.
        labels:
            Optional ground-truth class indices.  When provided the HF model
            returns a ``SequenceClassifierOutput`` with a ``loss`` field.

        Returns
        -------
        ``transformers.modeling_outputs.SequenceClassifierOutput``
        """
        return self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels,
        )

    # ------------------------------------------------------------------
    # Inference helpers
    # ------------------------------------------------------------------

    @torch.no_grad()
    def predict_proba(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> np.ndarray:
        """Return softmax class probabilities as a NumPy array.

        Parameters
        ----------
        input_ids, attention_mask:
            Token tensors, shape ``(batch, seq_len)``.

        Returns
        -------
        np.ndarray of shape ``(batch, num_labels)``
        """
        self.model.eval()
        outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
        probs = torch.softmax(outputs.logits, dim=-1)
        return probs.cpu().numpy()

    # ------------------------------------------------------------------
    # Checkpoint I/O
    # ------------------------------------------------------------------

    def save(self, checkpoint_dir: str | Path) -> None:
        """Save model weights and tokenizer to *checkpoint_dir*.

        Creates the directory if it does not exist.  Files written:
        - HuggingFace model config + weights (``config.json``, ``model.safetensors``)
        - Tokenizer files
        """
        checkpoint_dir = Path(checkpoint_dir)
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.model.save_pretrained(checkpoint_dir)
        self.tokenizer.save_pretrained(checkpoint_dir)
        print(f"  Checkpoint saved → {checkpoint_dir}")

    @classmethod
    def load_from_checkpoint(
        cls,
        checkpoint_dir: str | Path,
        device: str | None = None,
    ) -> "PretrainedClassifier":
        """Reload a classifier previously saved with :meth:`save`.

        Parameters
        ----------
        checkpoint_dir:
            Directory written by :meth:`save`.
        device:
            Target device override (auto-detected if ``None``).

        Returns
        -------
        PretrainedClassifier
        """
        checkpoint_dir = Path(checkpoint_dir)
        if not checkpoint_dir.exists():
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint_dir}")

        resolved_device = _resolve_device(device)

        tokenizer = AutoTokenizer.from_pretrained(checkpoint_dir)
        hf_model = AutoModelForSequenceClassification.from_pretrained(checkpoint_dir)

        # Build a shell instance without re-downloading from HF
        instance = cls.__new__(cls)
        nn.Module.__init__(instance)
        instance.model_name = str(checkpoint_dir)
        instance.num_labels = hf_model.config.num_labels
        instance.device = resolved_device
        instance.tokenizer = tokenizer
        instance.model = hf_model.to(resolved_device)

        print(f"  Loaded checkpoint from {checkpoint_dir}  (device={resolved_device})")
        return instance
