"""Fine-tuning wrapper for HuggingFace sequence classification models."""

from __future__ import annotations

import dataclasses
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset, random_split
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from src.models.base import BaseModel


def _resolve_device(device: str | None) -> str:
    if device is not None:
        return device
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


class _TextLabelDataset(Dataset):
    """Minimal Dataset that wraps pre-tokenized tensors and returns dict batches."""

    def __init__(
        self,
        input_ids: torch.Tensor,
        attention_masks: torch.Tensor,
        labels: torch.Tensor,
    ) -> None:
        self.input_ids = input_ids
        self.attention_masks = attention_masks
        self.labels = labels

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, idx: int) -> dict:
        return {
            "input_ids": self.input_ids[idx],
            "attention_mask": self.attention_masks[idx],
            "label": self.labels[idx],
        }


class PretrainedClassifier(nn.Module, BaseModel):
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
    # BaseModel interface
    # ------------------------------------------------------------------

    def predict(self, X: list[str], batch_size: int = 32) -> np.ndarray:
        """Tokenise *X* and return binary class predictions (0 or 1).

        Parameters
        ----------
        X:
            List of raw text strings.
        batch_size:
            Number of texts to process per forward pass.

        Returns
        -------
        np.ndarray of shape ``(len(X),)`` with integer labels.
        """
        all_preds: list[np.ndarray] = []
        for i in range(0, len(X), batch_size):
            batch = list(X[i : i + batch_size])
            encoded = self.tokenizer(
                batch,
                padding=True,
                truncation=True,
                max_length=512,
                return_tensors="pt",
            )
            input_ids = encoded["input_ids"].to(self.device)
            attention_mask = encoded["attention_mask"].to(self.device)
            probs = self.predict_proba(input_ids, attention_mask)
            all_preds.append(probs.argmax(axis=1))
        return np.concatenate(all_preds)

    def fit(self, X: list[str], y, **kwargs) -> None:
        """Fine-tune on raw texts *X* with labels *y*.

        Tokenises *X*, performs an 80/20 train/val split, and delegates to
        :class:`~src.training.trainer.Trainer`.

        Parameters
        ----------
        X:
            List of raw text strings (training set).
        y:
            Binary integer labels aligned with *X*.
        **kwargs:
            Passed through to :class:`~src.training.trainer.TrainerConfig`
            (e.g. ``num_epochs``, ``learning_rate``, ``output_dir``).
            ``batch_size`` (default 16) and ``max_length`` (default 512) are
            consumed here and not forwarded.
        """
        # Local import to avoid circular dependency (trainer.py imports this module).
        from src.training.trainer import Trainer, TrainerConfig

        batch_size = kwargs.pop("batch_size", 16)
        max_length = kwargs.pop("max_length", 512)

        # Tokenise all texts up front
        encoded = self.tokenizer(
            list(X),
            padding=True,
            truncation=True,
            max_length=max_length,
            return_tensors="pt",
        )
        dataset = _TextLabelDataset(
            encoded["input_ids"],
            encoded["attention_mask"],
            torch.tensor(list(y), dtype=torch.long),
        )

        # 80 / 20 train / val split
        n_val = max(1, int(0.2 * len(dataset)))
        n_train = len(dataset) - n_val
        train_ds, val_ds = random_split(dataset, [n_train, n_val])
        train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_ds, batch_size=batch_size)

        # Build TrainerConfig — only pass kwargs that are recognised fields
        valid_fields = {f.name for f in dataclasses.fields(TrainerConfig)}
        config = TrainerConfig(**{k: v for k, v in kwargs.items() if k in valid_fields})

        Trainer(self, config).train(train_loader, val_loader)

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
    def load(
        cls,
        filepath: str | Path,
        device: str | None = None,
    ) -> "PretrainedClassifier":
        """Reload a classifier previously saved with :meth:`save`.

        Parameters
        ----------
        filepath:
            Directory written by :meth:`save` (HuggingFace checkpoint directory).
        device:
            Target device override (auto-detected if ``None``).

        Returns
        -------
        PretrainedClassifier
        """
        checkpoint_dir = Path(filepath)
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
