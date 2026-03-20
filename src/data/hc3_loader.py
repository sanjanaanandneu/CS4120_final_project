"""HC3 dataset loader — thin wrapper around preprocess.load_dataset_splits."""

from typing import Optional
import pandas as pd

from src.data.preprocess import load_dataset_splits

HC3_DOMAINS = frozenset(
    ["reddit_eli5", "open_qa", "finance", "medicine", "wiki_csai"]
)


def load_hc3(
    domain: Optional[str] = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Load HC3 train/test splits, optionally filtered to a single domain.

    Parameters
    ----------
    domain:
        If given, keep only rows where ``source == domain``.
        Valid values: ``reddit_eli5``, ``open_qa``, ``finance``,
        ``medicine``, ``wiki_csai``.

    Returns
    -------
    train_df, test_df
    """
    train_df, test_df = load_dataset_splits("hc3")

    if domain is not None:
        if domain not in HC3_DOMAINS:
            raise ValueError(
                f"Unknown domain '{domain}'. Valid options: {sorted(HC3_DOMAINS)}"
            )
        print(f"  Filtering to domain: {domain}")
        train_df = train_df[train_df["source"] == domain].reset_index(drop=True)
        test_df = test_df[test_df["source"] == domain].reset_index(drop=True)
        print(f"  After filter — train: {len(train_df)}, test: {len(test_df)}")

    return train_df, test_df
