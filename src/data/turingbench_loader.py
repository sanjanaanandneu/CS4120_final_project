"""TuringBench dataset loader and preprocessor.

Reads the local TuringBench data from ``data/turingbench/AA/``,
converts multi-class labels to binary (0 = human, 1 = AI-generated), cleans
text, and saves stratified 80/20 train/test CSVs to ``data/processed/``.
"""

from __future__ import annotations

import sys
from pathlib import Path
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.data.preprocess import clean_text, PROCESSED_DIR

RAW_DIR = Path(__file__).resolve().parents[2] / "data" / "turingbench" / "AA"
MIN_CHAR_LENGTH = 50
TEXT_COL = "Generation"
LABEL_COL = "label"
HUMAN_LABEL = "human"


def preprocess_turingbench() -> tuple[pd.DataFrame, pd.DataFrame]:
    """Load, clean, binarise, and split TuringBench into train/test CSVs.

    Returns
    -------
    train_df, test_df
        DataFrames with columns ``text``, ``label``, ``source``.
    """
    if not RAW_DIR.exists():
        raise FileNotFoundError(f"TuringBench data not found at {RAW_DIR}")

    splits = {}
    for split in ("train", "test"):
        path = RAW_DIR / f"{split}.csv"
        if not path.exists():
            raise FileNotFoundError(f"Missing split file: {path}")
        splits[split] = pd.read_csv(path)

    train_df = splits["train"]
    test_df = splits["test"]

    print(f"  Raw train samples: {len(train_df)}")
    print(f"  Raw test samples:  {len(test_df)}")

    def process(df: pd.DataFrame) -> pd.DataFrame:
        df = df.rename(columns={TEXT_COL: "text", LABEL_COL: "source"})
        df["label"] = (df["source"] != HUMAN_LABEL).astype(int)
        df["text"] = df["text"].astype(str).apply(clean_text)
        df = df[df["text"].str.len() >= MIN_CHAR_LENGTH]
        df = df.drop_duplicates(subset=["text"])
        return df[["text", "label", "source"]].reset_index(drop=True)

    train_df = process(train_df)
    test_df = process(test_df)

    print(f"  Processed train samples: {len(train_df)}")
    print(f"  Processed test samples:  {len(test_df)}")
    print(f"\nLabel distribution (train):")
    print(train_df["label"].value_counts().rename({0: "human", 1: "ai"}).to_string())

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
