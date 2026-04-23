"""Fine-tune a pretrained HuggingFace model on the HC3 dataset.

Usage (from project root):
    python scripts/run_finetune.py
    python scripts/run_finetune.py --model distilbert-base-uncased --epochs 3
    python scripts/run_finetune.py --batch_size 8 --grad_accum 4 --output_dir models/my_run

Speed tips:
    --max_length 128   Reduces attention cost ~16x vs default 512 (recommended)
    --max_samples 10000  Cap training set size for quick experiments

All checkpoints are saved to ``--output_dir`` (best validation F1 wins).
Final test-set metrics are printed at the end.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.data.preprocess import load_dataset_splits
from src.data.hc3_dataset import HC3Dataset
from src.models.pretrained_classifier import PretrainedClassifier
from src.training.trainer import Trainer, TrainerConfig
from src.evaluation.metrics import compute_metrics, compute_domain_metrics, print_report


def _build_loaders(
    model_name: str,
    batch_size: int,
    max_length: int,
    dataset: str = "hc3",
    val_ratio: float = 0.1,
    max_samples: int | None = None,
) -> tuple[DataLoader, DataLoader, DataLoader, list[str]]:
    """Load dataset, carve a validation split, return three DataLoaders + test sources."""
    train_df, test_df = load_dataset_splits(dataset)

    # Optional subsample for quick experiments (stratified)
    if max_samples is not None and max_samples < len(train_df):
        train_df, _ = train_test_split(
            train_df,
            train_size=max_samples,
            stratify=train_df["label"],
            random_state=42,
        )
        train_df = train_df.reset_index(drop=True)
        print(f"  Subsampled training set to {len(train_df)} samples")

    # Stratified val split from train
    train_idx, val_idx = train_test_split(
        range(len(train_df)),
        test_size=val_ratio,
        stratify=train_df["label"],
        random_state=42,
    )
    val_df = train_df.iloc[val_idx].reset_index(drop=True)
    train_df = train_df.iloc[train_idx].reset_index(drop=True)

    print(
        f"\n  Split sizes — train: {len(train_df)}, "
        f"val: {len(val_df)}, test: {len(test_df)}"
    )

    # Build a temporary classifier just to get the tokenizer
    tokenizer = PretrainedClassifier(model_name=model_name).tokenizer

    train_ds = HC3Dataset(train_df, tokenizer, max_length=max_length)
    val_ds = HC3Dataset(val_df, tokenizer, max_length=max_length)
    test_ds = HC3Dataset(test_df, tokenizer, max_length=max_length)

    num_workers = 0  # tokenization is fast enough in-process

    train_loader = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True, num_workers=num_workers
    )
    val_loader = DataLoader(
        val_ds, batch_size=batch_size * 2, shuffle=False, num_workers=num_workers
    )
    test_loader = DataLoader(
        test_ds, batch_size=batch_size * 2, shuffle=False, num_workers=num_workers
    )

    test_sources = test_df["source"].tolist()
    return train_loader, val_loader, test_loader, test_sources


@torch.no_grad()
def _run_inference(
    classifier: PretrainedClassifier,
    loader: DataLoader,
) -> tuple[np.ndarray, np.ndarray]:
    """Return (predictions, probabilities) for an entire DataLoader."""
    classifier.eval()
    all_preds: list[int] = []
    all_proba: list[np.ndarray] = []

    for batch in loader:
        input_ids = batch["input_ids"].to(classifier.device)
        attention_mask = batch["attention_mask"].to(classifier.device)

        proba = classifier.predict_proba(input_ids, attention_mask)
        all_proba.append(proba)
        all_preds.extend(proba.argmax(axis=1).tolist())

    return np.array(all_preds), np.vstack(all_proba)



def main() -> None:
    parser = argparse.ArgumentParser(description="Fine-tune a HuggingFace model on HC3")
    parser.add_argument("--model", default="distilroberta-base", help="HuggingFace model ID")
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--grad_accum", type=int, default=2)
    parser.add_argument("--max_length", type=int, default=128,
                        help="Max token length")
    parser.add_argument("--max_samples", type=int, default=None,
                        help="Cap training set size (stratified subsample)")
    parser.add_argument("--val_ratio", type=float, default=0.1)
    parser.add_argument("--output_dir", default=None, help="Checkpoint directory (auto-named if omitted)")
    parser.add_argument("--patience", type=int, default=2)
    parser.add_argument("--dataset", default="hc3", choices=["hc3", "turingbench", "combined"],
                        help="Dataset to train on (default: hc3)")
    args = parser.parse_args()

    output_dir = args.output_dir or f"models/{args.model.replace('/', '_')}_{args.dataset}"

    # Data
    train_loader, val_loader, test_loader, test_sources = _build_loaders(
        model_name=args.model,
        batch_size=args.batch_size,
        max_length=args.max_length,
        dataset=args.dataset,
        val_ratio=args.val_ratio,
        max_samples=args.max_samples,
    )

    # Model
    classifier = PretrainedClassifier(model_name=args.model)

    # Trainer
    config = TrainerConfig(
        output_dir=output_dir,
        num_epochs=args.epochs,
        learning_rate=args.lr,
        gradient_accumulation_steps=args.grad_accum,
        early_stopping_patience=args.patience,
    )
    trainer = Trainer(classifier, config)
    trainer.train(train_loader, val_loader)

    # Reload best checkpoint for final evaluation
    print(f"\nReloading best checkpoint from {output_dir} …")
    best_classifier = PretrainedClassifier.load(output_dir)

    # Test evaluation
    print("Running inference on test set …")
    test_labels = [batch["label"].tolist() for batch in test_loader]
    test_labels_flat = [lbl for batch in test_labels for lbl in batch]

    preds, proba = _run_inference(best_classifier, test_loader)

    overall = compute_metrics(test_labels_flat, preds, proba)
    domain = compute_domain_metrics(test_labels_flat, preds, test_sources, proba)
    print_report(overall, domain, title=f"Test Results — {args.model}")


if __name__ == "__main__":
    main()
