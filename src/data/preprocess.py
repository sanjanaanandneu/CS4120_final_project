"""Shared text cleaning and dataset loading utilities."""

import re
from pathlib import Path
import pandas as pd

PROCESSED_DIR = Path(__file__).resolve().parents[2] / "data" / "processed"


def clean_text(text: str) -> str:
    """Normalize a text string.

    - Collapses all whitespace/newlines to single spaces.
    - Removes URL placeholders (URL_0, URL_1, …).
    - Strips leading/trailing whitespace.
    """
    text = text.strip()
    text = re.sub(r"URL_\d+", "", text)
    text = re.sub(r"\s+", " ", text)
    text = text.strip()
    return text


def _validate_and_summarize(df: pd.DataFrame, name: str) -> None:
    """Print basic stats and warn about nulls."""
    null_counts = df.isnull().sum()
    if null_counts.any():
        print(f"  [WARNING] Null values found in {name}:")
        print(f"  {null_counts[null_counts > 0].to_dict()}")

    print(f"  Label distribution:\n{df['label'].value_counts().to_string()}")
    print(f"  Avg text length (chars): {df['text'].str.len().mean():.0f}")
    print(f"  Total samples: {len(df)}")


def load_dataset_splits(dataset_name: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Load train and test CSV splits for *dataset_name* from ``data/processed/``.

    Parameters
    ----------
    dataset_name:
        One of ``"hc3"`` or ``"turingbench"`` (matches the CSV filename prefix).

    Returns
    -------
    train_df, test_df
    """
    train_path = PROCESSED_DIR / f"{dataset_name}_train.csv"
    test_path = PROCESSED_DIR / f"{dataset_name}_test.csv"

    if not train_path.exists():
        raise FileNotFoundError(f"Train split not found: {train_path}")
    if not test_path.exists():
        raise FileNotFoundError(f"Test split not found: {test_path}")

    print(f"\nLoading {dataset_name} …")
    train_df = pd.read_csv(train_path)
    test_df = pd.read_csv(test_path)

    print(f"  Train ({len(train_df)} rows):")
    _validate_and_summarize(train_df, f"{dataset_name} train")
    print(f"  Test ({len(test_df)} rows):")
    _validate_and_summarize(test_df, f"{dataset_name} test")

    return train_df, test_df
