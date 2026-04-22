"""Shared plotting utilities for model training visualisation.

All functions accept an optional save_path argument. When provided the figure
is saved to disk and closed; otherwise plt.show() is called instead.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.metrics import confusion_matrix


def _save_or_show(fig: plt.Figure, save_path: str | Path | None) -> None:
    if save_path is not None:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, bbox_inches="tight")
        plt.close(fig)
    else:
        plt.show()


def plot_training_curves(
    train_losses: list[float],
    val_losses: list[float] | None = None,
    train_accs: list[float] | None = None,
    val_accs: list[float] | None = None,
    save_path: str | Path | None = None,
) -> None:
    """Plot loss and accuracy curves over training epochs.

    Parameters
    ----------
    train_losses:
        Per-epoch training loss values.
    val_losses:
        Per-epoch validation loss values (optional).
    train_accs:
        Per-epoch training accuracy values (optional).
    val_accs:
        Per-epoch validation accuracy values (optional).
    save_path:
        If provided, save the figure here instead of calling plt.show().
    """
    has_acc = train_accs is not None
    n_panels = 2 if has_acc else 1
    fig, axes = plt.subplots(1, n_panels, figsize=(6 * n_panels, 5))
    if n_panels == 1:
        axes = [axes]

    epochs = range(1, len(train_losses) + 1)

    axes[0].plot(epochs, train_losses, marker="o", label="Training Loss")
    if val_losses is not None:
        axes[0].plot(epochs, val_losses, marker="o", label="Validation Loss")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Loss")
    axes[0].set_title("Training and Validation Loss")
    axes[0].legend()

    if has_acc:
        axes[1].plot(epochs, train_accs, marker="o", label="Training Accuracy")
        if val_accs is not None:
            axes[1].plot(epochs, val_accs, marker="o", label="Validation Accuracy")
        axes[1].set_xlabel("Epoch")
        axes[1].set_ylabel("Accuracy")
        axes[1].set_title("Training and Validation Accuracy")
        axes[1].legend()

    fig.tight_layout()
    _save_or_show(fig, save_path)


def plot_confusion_matrix(
    y_true,
    y_pred,
    labels: list[str] | None = None,
    save_path: str | Path | None = None,
) -> None:
    """Plot a labelled confusion matrix heatmap.

    Parameters
    ----------
    y_true:
        Ground-truth binary labels.
    y_pred:
        Predicted binary labels.
    labels:
        Display names for the two classes. Defaults to ["Human", "AI"].
    save_path:
        If provided, save the figure here instead of calling plt.show().
    """
    if labels is None:
        labels = ["Human", "AI"]

    cm = confusion_matrix(y_true, y_pred)
    fig, ax = plt.subplots(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax,
                xticklabels=labels, yticklabels=labels)
    ax.set_xlabel("Predicted Label")
    ax.set_ylabel("True Label")
    ax.set_title("Confusion Matrix")
    _save_or_show(fig, save_path)


def plot_prediction_distribution(
    y_pred_proba,
    y_true,
    save_path: str | Path | None = None,
) -> None:
    """Plot the distribution of predicted probabilities split by true class.

    Produces two side-by-side panels: a histogram of predicted probabilities
    per class and a box plot of predicted probability by true label.

    Parameters
    ----------
    y_pred_proba:
        1-D array of predicted probabilities for the positive (AI) class.
    y_true:
        Ground-truth binary labels aligned with y_pred_proba.
    save_path:
        If provided, save the figure here instead of calling plt.show().
    """
    y_pred_proba = np.asarray(y_pred_proba)
    y_true = np.asarray(y_true)

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    axes[0].hist(y_pred_proba[y_true == 0], bins=20, alpha=0.7, label="Human")
    axes[0].hist(y_pred_proba[y_true == 1], bins=20, alpha=0.7, label="AI")
    axes[0].set_xlabel("Predicted Probability")
    axes[0].set_ylabel("Frequency")
    axes[0].set_title("Distribution of Predicted Probabilities")
    axes[0].legend()

    sns.boxplot(x=y_true, y=y_pred_proba, ax=axes[1])
    axes[1].set_xlabel("True Label")
    axes[1].set_ylabel("Predicted Probability")
    axes[1].set_title("Predicted Probabilities by True Label")

    fig.tight_layout()
    _save_or_show(fig, save_path)
