"""TuringBench dataset loader and preprocessor.

Loads the TuringBench dataset from HuggingFace (``turingbench/TuringBench``),
converts multi-class labels to binary (0 = human, 1 = AI-generated), cleans
text, and saves stratified 80/20 train/test CSVs.

Falls back to downloading Parquet files directly via ``huggingface_hub`` if
the installed ``datasets`` version does not support the dataset script.
"""

from __future__ import annotations

import sys
from pathlib import Path
import pandas as pd
from sklearn.model_selection import train_test_split

# Allow running as a script from the project root
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.data.preprocess import clean_text, PROCESSED_DIR

DATASET_ID = "turingbench/TuringBench"
MIN_CHAR_LENGTH = 50
TEST_SIZE = 0.2
RANDOM_STATE = 42

# The text column name in the raw TuringBench dataset
TEXT_COL = "Generation"
LABEL_COL = "label"
HUMAN_LABEL = "Human"


def _load_via_datasets() -> pd.DataFrame:
    """Try loading with the ``datasets`` library."""
    from datasets import load_dataset  # type: ignore

    print(f"  Loading '{DATASET_ID}' via datasets library …")
    ds = load_dataset(DATASET_ID, split="train")
    df = ds.to_pandas()
    return df


def _load_via_huggingface_hub() -> pd.DataFrame:
    """Fallback: download Parquet files directly from HuggingFace Hub."""
    from huggingface_hub import hf_hub_download  # type: ignore
    import os

    print("  Falling back to direct Parquet download via huggingface_hub …")

    # TuringBench stores data as a single parquet file in the default config
    parquet_path = hf_hub_download(
        repo_id=DATASET_ID,
        filename="data/train-00000-of-00001.parquet",
        repo_type="dataset",
    )
    df = pd.read_parquet(parquet_path)
    return df


def _load_raw() -> pd.DataFrame:
    """Load raw TuringBench data, trying datasets first then Parquet fallback."""
    try:
        return _load_via_datasets()
    except Exception as e:
        print(f"  datasets library failed ({type(e).__name__}: {e})")
        return _load_via_huggingface_hub()


def preprocess_turingbench() -> tuple[pd.DataFrame, pd.DataFrame]:
    """Load, clean, binarise, and split TuringBench into train/test CSVs.

    Returns
    -------
    train_df, test_df
        DataFrames with columns ``text``, ``label``, ``source``.
    """
    df = _load_raw()
    print(f"  Raw samples: {len(df)}")
    print(f"  Columns: {list(df.columns)}")

    # Rename to standard column names
    df = df.rename(columns={TEXT_COL: "text", LABEL_COL: "source"})

    # Binarise: Human → 0, everything else → 1
    df["label"] = (df["source"] != HUMAN_LABEL).astype(int)

    # Clean text
    print("  Cleaning text …")
    df["text"] = df["text"].astype(str).apply(clean_text)

    # Filter short samples
    before = len(df)
    df = df[df["text"].str.len() >= MIN_CHAR_LENGTH].reset_index(drop=True)
    print(f"  Dropped {before - len(df)} samples shorter than {MIN_CHAR_LENGTH} chars")

    # Remove duplicates
    before = len(df)
    df = df.drop_duplicates(subset=["text"]).reset_index(drop=True)
    print(f"  Dropped {before - len(df)} duplicate texts")
    print(f"  Remaining samples: {len(df)}")

    # Keep only needed columns
    df = df[["text", "label", "source"]]

    print("\nLabel distribution:")
    print(df["label"].value_counts().rename({0: "human", 1: "ai"}).to_string())

    print("\nSamples per generator:")
    print(df["source"].value_counts().to_string())

    print(f"\nAvg text length (chars): {df['text'].str.len().mean():.0f}")

    # Stratified split
    train_df, test_df = train_test_split(
        df,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=df["label"],
    )
    train_df = train_df.reset_index(drop=True)
    test_df = test_df.reset_index(drop=True)

    # Save
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    train_path = PROCESSED_DIR / "turingbench_train.csv"
    test_path = PROCESSED_DIR / "turingbench_test.csv"
    train_df.to_csv(train_path, index=False)
    test_df.to_csv(test_path, index=False)
    print(f"\n  Saved train split ({len(train_df)} rows) → {train_path}")
    print(f"  Saved test split  ({len(test_df)} rows) → {test_path}")

    return train_df, test_df


if __name__ == "__main__":
    preprocess_turingbench()
