"""Download and preprocess the HC3 dataset, saving train/test CSVs.

Usage (from project root):
    python scripts/download_hc3.py
"""

from __future__ import annotations

import os
import re
import sys
from pathlib import Path

import pandas as pd
from datasets import load_dataset
from sklearn.model_selection import train_test_split

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

PROCESSED_DATA_PATH = "data/processed/"
TEST_SIZE = 0.2
MIN_CHAR_LENGTH = 20
RANDOM_STATE = 42


def clean_text(text: str) -> str:
    text = text.strip()
    text = re.sub(r"\s+", " ", text)
    text = re.sub(r"URL_\d+", "", text)
    text = text.strip()
    return text


def main() -> None:
    ds = load_dataset("Hello-SimpleAI/HC3", "all", split="train")

    rows = []
    for item in ds:
        question = item["question"]
        source = item["source"]

        for ans in item["human_answers"]:
            rows.append({"text": ans, "label": 0, "source": source, "question": question})
        for ans in item["chatgpt_answers"]:
            rows.append({"text": ans, "label": 1, "source": source, "question": question})

    df = pd.DataFrame(rows)
    print(f"  Flattened samples: {len(df)}")

    df["text"] = df["text"].apply(clean_text)

    before = len(df)
    df = df[df["text"].str.len() >= MIN_CHAR_LENGTH].reset_index(drop=True)
    print(f"  Dropped {before - len(df)} samples shorter than {MIN_CHAR_LENGTH} chars")

    before = len(df)
    df = df.drop_duplicates(subset=["text"]).reset_index(drop=True)
    print(f"  Dropped {before - len(df)} duplicate texts")
    print(f"  Remaining samples: {len(df)}")

    print("\nLabel distribution:")
    print(df["label"].value_counts().rename({0: "human", 1: "chatgpt"}))

    print("\nSamples per domain:")
    print(
        df.groupby("source")["label"]
        .value_counts()
        .unstack(fill_value=0)
        .rename(columns={0: "human", 1: "chatgpt"})
    )

    print(f"\nAvg text length (chars): {df['text'].str.len().mean():.0f}")

    df["strat_col"] = df["label"].astype(str) + "_" + df["source"]

    train_df, test_df = train_test_split(
        df,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=df["strat_col"],
    )

    train_df = train_df.drop(columns=["strat_col"]).reset_index(drop=True)
    test_df = test_df.drop(columns=["strat_col"]).reset_index(drop=True)

    os.makedirs(PROCESSED_DATA_PATH, exist_ok=True)
    train_df.to_csv(PROCESSED_DATA_PATH + "hc3_train.csv", index=False)
    test_df.to_csv(PROCESSED_DATA_PATH + "hc3_test.csv", index=False)

    print(f"\nSaved train ({len(train_df)}) → {PROCESSED_DATA_PATH}hc3_train.csv")
    print(f"Saved test  ({len(test_df)})  → {PROCESSED_DATA_PATH}hc3_test.csv")


if __name__ == "__main__":
    main()
