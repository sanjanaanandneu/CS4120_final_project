"""Evaluate a saved pretrained checkpoint on a dataset test split.

Usage (from project root):
    python scripts/eval_pretrained.py --checkpoint models/distilroberta-base_combined
    python scripts/eval_pretrained.py --checkpoint models/distilroberta-base_combined --dataset turingbench
    python scripts/eval_pretrained.py --checkpoint models/distilroberta-base_combined --dataset hc3 --max_length 64
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.data.preprocess import load_dataset_splits
from src.data.hc3_dataset import HC3Dataset
from src.models.pretrained_classifier import PretrainedClassifier
from src.evaluation.metrics import compute_metrics, compute_domain_metrics, print_report, plot_confusion_matrix


def _load_test_df(dataset: str):
    if dataset == "raid":
        from src.data.raid_loader import preprocess_raid
        import pandas as pd
        from src.data.preprocess import PROCESSED_DIR
        test_path = PROCESSED_DIR / "raid_test.csv"
        if test_path.exists():
            return pd.read_csv(test_path)
        _, test_df = preprocess_raid()
        return test_df
    _, test_df = load_dataset_splits(dataset)
    return test_df


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate a saved pretrained checkpoint")
    parser.add_argument("--checkpoint", required=True, help="Path to saved checkpoint directory")
    parser.add_argument("--dataset", default="turingbench", choices=["hc3", "turingbench", "combined", "raid"])
    parser.add_argument("--max_length", type=int, default=128)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--output_dir", default=None, help="Where to save plots (defaults to checkpoint dir)")
    args = parser.parse_args()

    test_df = _load_test_df(args.dataset)
    y_true = test_df["label"].values

    classifier = PretrainedClassifier.load_from_checkpoint(args.checkpoint)
    classifier.eval()

    test_ds = HC3Dataset(test_df, classifier.tokenizer, max_length=args.max_length)
    test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False)

    all_preds, all_proba = [], []
    with torch.no_grad():
        for batch in test_loader:
            input_ids = batch["input_ids"].to(classifier.device)
            attention_mask = batch["attention_mask"].to(classifier.device)
            proba = classifier.predict_proba(input_ids, attention_mask)
            all_proba.append(proba)
            all_preds.extend(proba.argmax(axis=1).tolist())

    preds = np.array(all_preds)
    proba = np.vstack(all_proba)

    metrics = compute_metrics(y_true, preds, proba)
    domain_metrics = (
        compute_domain_metrics(y_true, preds, test_df["source"].values, proba)
        if args.dataset == "hc3"
        else None
    )
    print_report(metrics, domain_metrics, title=f"{Path(args.checkpoint).name} — {args.dataset}")

    out_dir = Path(args.output_dir or args.checkpoint)
    plot_confusion_matrix(
        y_true, preds,
        save_path=out_dir / f"confusion_matrix_{args.dataset}.png",
        title=f"{Path(args.checkpoint).name} — {args.dataset}",
    )


if __name__ == "__main__":
    main()
