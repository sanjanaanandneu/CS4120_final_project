import re
from datasets import load_dataset
import pandas as pd
from sklearn.model_selection import train_test_split

PROCESSED_DATA_PATH = "data/processed/"
TEST_SIZE = 0.2
MIN_CHAR_LENGTH = 20
RANDOM_STATE = 42

ds = load_dataset("Hello-SimpleAI/HC3", "all", split="train")

rows = []
for item in ds:
    question = item["question"]
    source = item["source"]

    for ans in item["human_answers"]:
        rows.append({
            "text": ans,
            "label": 0,
            "source": source,
            "question": question,
        })
    for ans in item["chatgpt_answers"]:
        rows.append({
            "text": ans,
            "label": 1,
            "source": source,
            "question": question,
        })

df = pd.DataFrame(rows)
print(f"  Flattened samples: {len(df)}")

def clean_text(text: str) -> str:
    """Basic text normalization."""
    text = text.strip()
    # collapse multiple whitespace / newlines
    text = re.sub(r"\s+", " ", text)
    # remove URL placeholders from the dataset (URL_0, URL_1, etc.)
    text = re.sub(r"URL_\d+", "", text)
    # strip leading/trailing whitespace again after replacements
    text = text.strip()
    return text

df["text"] = df["text"].apply(clean_text)

before = len(df)
df = df[df["text"].str.len() >= MIN_CHAR_LENGTH].reset_index(drop=True)
print(f"  Dropped {before - len(df)} samples shorter than {MIN_CHAR_LENGTH} chars")
print(f"  Remaining samples: {len(df)}")

# Drop exact duplicates
before = len(df)
df = df.drop_duplicates(subset=["text"]).reset_index(drop=True)
print(f"  Dropped {before - len(df)} duplicate texts")

print("\nLabel distribution:")
print(df["label"].value_counts().rename({0: "human", 1: "chatgpt"}))

print("\nSamples per domain:")
print(df.groupby("source")["label"].value_counts().unstack(fill_value=0)
      .rename(columns={0: "human", 1: "chatgpt"}))

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

import os
os.makedirs(PROCESSED_DATA_PATH, exist_ok=True)

train_df.to_csv(PROCESSED_DATA_PATH + "hc3_train.csv", index=False)
test_df.to_csv(PROCESSED_DATA_PATH + "hc3_test.csv", index=False)

print(f"\nSaved train ({len(train_df)}) → {PROCESSED_DATA_PATH}hc3_train.csv")
print(f"Saved test  ({len(test_df)})  → {PROCESSED_DATA_PATH}hc3_test.csv")
