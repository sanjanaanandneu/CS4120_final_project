# CS4120 Final Project — AI Text Detection

Binary classification of human-written vs. AI-generated text. Compares neural networks against traditional feature extraction (TF-IDF, n-grams, BERT embeddings) using the HC3 dataset.

## Setup

```bash
pip install -r requirements.txt
```

## Usage

### 1. Generate HC3 data

```bash
python data/hc3_data.py
```

Saves `data/processed/hc3_train.csv` and `data/processed/hc3_test.csv`.

### 2. Run preprocessing

```bash
python scripts/run_preprocessing.py
```

Prints label distributions and average text lengths for the dataset.

### 3. Extract features

```bash
python scripts/run_feature_extraction.py
```

Saves all feature matrices to `data/processed/features/hc3/`:

| File | Type | Shape |
|------|------|-------|
| `tfidf_train.npz` / `tfidf_test.npz` | sparse | `(n, max_features)` |
| `word_ngram_train.npz` / `word_ngram_test.npz` | sparse | `(n, vocab_size)` |
| `char_ngram_train.npz` / `char_ngram_test.npz` | sparse | `(n, vocab_size)` |
| `bert_embeddings_train.npy` / `bert_embeddings_test.npy` | dense | `(n, 768)` |

> Note: BERT embedding extraction can take 10–30 minutes depending on hardware.

## Using Features in a Model

**Sparse features (TF-IDF / n-grams) → sklearn:**
```python
import scipy.sparse as sp
from src.data.preprocess import load_dataset_splits

X_train = sp.load_npz("data/processed/features/hc3/tfidf_train.npz")
X_test  = sp.load_npz("data/processed/features/hc3/tfidf_test.npz")

train_df, test_df = load_dataset_splits("hc3")
y_train = train_df["label"].values
y_test  = test_df["label"].values
```

**Dense embeddings → PyTorch:**
```python
import numpy as np, torch

X_train = torch.tensor(np.load("data/processed/features/hc3/bert_embeddings_train.npy"))
X_test  = torch.tensor(np.load("data/processed/features/hc3/bert_embeddings_test.npy"))
# shape: (n_samples, 768) — feed into nn.Linear(768, 2)
```

## Project Structure

```
CS4120_final_project/
├── data/
│   ├── hc3_data.py              # HC3 download + preprocessing
│   └── processed/               # generated CSVs and feature matrices
├── src/
│   ├── data/
│   │   ├── preprocess.py        # clean_text, load_dataset_splits
│   │   └── hc3_loader.py        # HC3 loader with domain filtering
│   └── features/
│       ├── tfidf.py             # TF-IDF vectorizer
│       ├── ngrams.py            # character + word n-gram counts
│       └── embeddings.py        # BERT mean-pooled embeddings
├── scripts/
│   ├── run_preprocessing.py
│   └── run_feature_extraction.py
└── requirements.txt
```

## Dataset

**HC3** (Human ChatGPT Comparison Corpus) — Q&A pairs from Reddit ELI5, open QA, finance, medicine, and Wikipedia. Labels: 0 = human, 1 = ChatGPT.
