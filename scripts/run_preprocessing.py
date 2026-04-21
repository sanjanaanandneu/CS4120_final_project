"""Run TuringBench preprocessing and print summary stats for both datasets.

Usage (from project root):
    python scripts/run_preprocessing.py
"""

from __future__ import annotations

import sys
from pathlib import Path

# Ensure project root is on the path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.data.preprocess import load_dataset_splits


def main() -> None:
    print("=" * 60)
    print("Summary stats — HC3")
    print("=" * 60)
    hc3_train, hc3_test = load_dataset_splits("hc3")

    print("\n" + "=" * 60)
    print("Preprocessing complete.")
    print(f"  HC3 — train: {len(hc3_train):>7,}  test: {len(hc3_test):>6,}")
    print("=" * 60)

    print("=" * 60)
    print("Summary stats — TuringBench")
    print("=" * 60)
    turingbench_train, turingbench_test = load_dataset_splits("turingbench")

    print("\n" + "=" * 60)
    print("Preprocessing complete.")
    print(f"  TuringBench — train: {len(turingbench_train):>7,}  test: {len(turingbench_test):>6,}")
    print("=" * 60)

    print("=" * 60)
    print("Summary stats — Combined")
    print("=" * 60)
    combined_train, combined_test = load_dataset_splits("combined")

    print("\n" + "=" * 60)
    print("Preprocessing complete.")
    print(f"  Combined — train: {len(combined_train):>7,}  test: {len(combined_test):>6,}")
    print("=" * 60)


if __name__ == "__main__":
    main()
