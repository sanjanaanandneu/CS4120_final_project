"""PyTorch Dataset for tokenizing HC3 text samples."""

from __future__ import annotations

import torch
from torch.utils.data import Dataset
import pandas as pd
from transformers import PreTrainedTokenizerBase


class HC3Dataset(Dataset):
    """Tokenizes and wraps HC3 DataFrame rows for use with a DataLoader.

    Parameters
    ----------
    df:
        DataFrame with at least ``text`` and ``label`` columns.
    tokenizer:
        HuggingFace tokenizer instance (already loaded).
    max_length:
        Maximum token sequence length. Inputs are truncated/padded to this.
    """

    def __init__(
        self,
        df: pd.DataFrame,
        tokenizer: PreTrainedTokenizerBase,
        max_length: int = 512,
    ) -> None:
        self.texts: list[str] = df["text"].tolist()
        self.labels: list[int] = df["label"].tolist()
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self) -> int:
        return len(self.texts)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        encoding = self.tokenizer(
            self.texts[idx],
            padding="max_length",
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt",
        )
        return {
            "input_ids": encoding["input_ids"].squeeze(0),       # (seq_len,)
            "attention_mask": encoding["attention_mask"].squeeze(0),  # (seq_len,)
            "label": torch.tensor(self.labels[idx], dtype=torch.long),
        }
