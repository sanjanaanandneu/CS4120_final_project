"""Evaluation utilities for the AI-text detection task.

Two layers of metrics are provided:

Custom (confusion-matrix-based)
    _compute_confusion_matrix, compute_accuracy, compute_precision,
    compute_recall, compute_f1
    Built from scratch using NumPy.  Useful for verification and as
    transparent reference implementations.

sklearn-based
    compute_metrics, compute_domain_metrics
    Delegate to scikit-learn for weighted/macro averaging and ROC-AUC.
    These are the primary metrics used in evaluation reports.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

HC3_DOMAINS = ["reddit_eli5", "open_qa", "finance", "medicine", "wiki_csai"]


# ---------------------------------------------------------------------------
# sklearn-based metrics
# ---------------------------------------------------------------------------

def compute_metrics(
    y_true: list[int] | np.ndarray,
    y_pred: list[int] | np.ndarray,
    y_proba: np.ndarray | None = None,
) -> dict[str, float]:
    """Compute classification metrics for a single evaluation set.

    Parameters
    ----------
    y_true:
        Ground-truth binary labels (0 = human, 1 = AI).
    y_pred:
        Predicted class indices.
    y_proba:
        Optional predicted probabilities, shape ``(n, 2)``.  Required for
        ROC-AUC; omit to skip that metric.

    Returns
    -------
    Dict with keys: ``accuracy``, ``f1_weighted``, ``f1_macro``,
    ``precision_weighted``, ``recall_weighted``, and optionally ``roc_auc``.
    """
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)

    metrics: dict[str, float] = {
        "accuracy": accuracy_score(y_true, y_pred),
        "f1_weighted": f1_score(y_true, y_pred, average="weighted"),
        "f1_macro": f1_score(y_true, y_pred, average="macro"),
        "precision_weighted": precision_score(y_true, y_pred, average="weighted", zero_division=0),
        "recall_weighted": recall_score(y_true, y_pred, average="weighted", zero_division=0),
    }

    if y_proba is not None:
        metrics["roc_auc"] = roc_auc_score(y_true, y_proba[:, 1])

    return metrics


def compute_domain_metrics(
    y_true: list[int] | np.ndarray,
    y_pred: list[int] | np.ndarray,
    sources: list[str] | np.ndarray,
    y_proba: np.ndarray | None = None,
) -> dict[str, dict[str, float]]:
    """Compute metrics broken down by domain/source.

    Parameters
    ----------
    y_true, y_pred, y_proba:
        Same as :func:`compute_metrics`.
    sources:
        Array of source strings aligned with *y_true* / *y_pred*
        (e.g. ``["reddit_eli5", "finance", ...]``).

    Returns
    -------
    Dict mapping each domain name to its own metrics dict.
    """
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    sources = np.asarray(sources)

    domain_results: dict[str, dict[str, float]] = {}
    for domain in HC3_DOMAINS:
        mask = sources == domain
        if mask.sum() == 0:
            continue
        proba_slice = y_proba[mask] if y_proba is not None else None
        domain_results[domain] = compute_metrics(
            y_true[mask], y_pred[mask], proba_slice
        )

    return domain_results


def print_report(
    metrics: dict[str, float],
    domain_metrics: dict[str, dict[str, float]] | None = None,
    title: str = "Evaluation Results",
) -> None:
    """Pretty-print an evaluation report to stdout.

    Parameters
    ----------
    metrics:
        Overall metrics dict from :func:`compute_metrics`.
    domain_metrics:
        Optional per-domain breakdown from :func:`compute_domain_metrics`.
    title:
        Header label for the report.
    """
    width = 60
    print(f"\n{'='*width}")
    print(f"  {title}")
    print(f"{'='*width}")
    print(f"  {'Accuracy':<25} {metrics['accuracy']:.4f}")
    print(f"  {'F1 (weighted)':<25} {metrics['f1_weighted']:.4f}")
    print(f"  {'F1 (macro)':<25} {metrics['f1_macro']:.4f}")
    print(f"  {'Precision (weighted)':<25} {metrics['precision_weighted']:.4f}")
    print(f"  {'Recall (weighted)':<25} {metrics['recall_weighted']:.4f}")
    if "roc_auc" in metrics:
        print(f"  {'ROC-AUC':<25} {metrics['roc_auc']:.4f}")

    if domain_metrics:
        print(f"\n  Per-domain F1 (weighted):")
        for domain, dm in domain_metrics.items():
            n_label = f"[{domain}]"
            print(f"    {n_label:<20} {dm['f1_weighted']:.4f}")

    print(f"{'='*width}")


def evaluate_model(
    model,
    X_test,
    y_true: list[int] | np.ndarray,
) -> dict[str, float]:
    """Call model.predict then return the full sklearn-based metrics dict.

    This is the canonical evaluation entry point.  Use the custom
    compute_precision / compute_recall / compute_f1 helpers when you need
    individual metric values derived from the confusion matrix directly.

    Parameters
    ----------
    model:
        Any model instance that exposes a predict(X) method.
    X_test:
        Feature matrix or raw texts matching the format expected by model.
    y_true:
        Ground-truth binary labels.

    Returns
    -------
    Dict with keys: ``accuracy``, ``f1_weighted``, ``f1_macro``,
    ``precision_weighted``, ``recall_weighted``.
    """
    y_pred = model.predict(X_test)
    return compute_metrics(y_true, y_pred)


def build_results_table(
    results: dict[str, dict[str, float]],
) -> pd.DataFrame:
    """Convert a mapping of model-name → metrics into a DataFrame table.

    Parameters
    ----------
    results:
        e.g. ``{"TF-IDF + LR": {...}, "DistilRoBERTa": {...}}``

    Returns
    -------
    DataFrame with models as rows and metric names as columns.
    """
    rows = []
    for model_name, m in results.items():
        row = {"model": model_name}
        row.update(m)
        rows.append(row)
    return pd.DataFrame(rows).set_index("model")
