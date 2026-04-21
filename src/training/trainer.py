"""Training loop for fine-tuning PretrainedClassifier on HC3."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np
import torch
from sklearn.metrics import f1_score
from torch.utils.data import DataLoader
from transformers import get_linear_schedule_with_warmup
from tqdm import tqdm

from src.models.pretrained_classifier import PretrainedClassifier


@dataclass
class TrainerConfig:
    """Hyperparameters and settings for the training loop.

    Attributes
    ----------
    output_dir:
        Directory where the best checkpoint will be written.
    num_epochs:
        Maximum number of full passes over the training set.
    learning_rate:
        Peak AdamW learning rate.
    weight_decay:
        L2 regularisation coefficient for AdamW.
    warmup_ratio:
        Fraction of total training steps used for linear LR warm-up.
    gradient_accumulation_steps:
        Accumulate gradients over this many batches before each optimizer step.
        Effective batch size = DataLoader batch size × this value.
    max_grad_norm:
        Gradient clipping threshold.
    early_stopping_patience:
        Stop training if validation F1 does not improve for this many epochs.
    use_amp:
        Enable automatic mixed precision (requires CUDA; ignored on MPS/CPU).
    """

    output_dir: str = "src/models/distilroberta"
    num_epochs: int = 5
    learning_rate: float = 2e-5
    weight_decay: float = 0.01
    warmup_ratio: float = 0.1
    gradient_accumulation_steps: int = 2
    max_grad_norm: float = 1.0
    early_stopping_patience: int = 2
    use_amp: bool = True


@dataclass
class EpochResult:
    epoch: int
    train_loss: float
    val_loss: float
    val_f1: float
    is_best: bool = False


class Trainer:
    """Fine-tunes a :class:`~src.models.pretrained_classifier.PretrainedClassifier`.

    Parameters
    ----------
    classifier:
        Model instance to train (already on the correct device).
    config:
        Training hyperparameters.
    """

    def __init__(
        self,
        classifier: PretrainedClassifier,
        config: TrainerConfig,
    ) -> None:
        self.classifier = classifier
        self.config = config
        self.device = classifier.device
        self.history: list[EpochResult] = []

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def train(
        self,
        train_loader: DataLoader,
        val_loader: DataLoader,
    ) -> list[EpochResult]:
        """Run the training loop.

        Parameters
        ----------
        train_loader, val_loader:
            DataLoaders that yield ``{input_ids, attention_mask, label}`` dicts.

        Returns
        -------
        List of :class:`EpochResult` (one per completed epoch).
        """
        cfg = self.config
        output_dir = Path(cfg.output_dir)

        # Optimizer
        optimizer = torch.optim.AdamW(
            self.classifier.parameters(),
            lr=cfg.learning_rate,
            weight_decay=cfg.weight_decay,
        )

        # LR Scheduler
        total_steps = (
            len(train_loader) // cfg.gradient_accumulation_steps * cfg.num_epochs
        )
        warmup_steps = int(total_steps * cfg.warmup_ratio)
        scheduler = get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=warmup_steps,
            num_training_steps=total_steps,
        )

        # AMP scaler (CUDA only)
        use_amp = cfg.use_amp and self.device == "cuda"
        scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

        best_val_f1 = -1.0
        no_improve_count = 0

        print(f"\n{'='*60}")
        print(f"Training {self.classifier.model_name}")
        print(f"  Epochs: {cfg.num_epochs} | LR: {cfg.learning_rate}")
        print(f"  Warmup steps: {warmup_steps}/{total_steps}")
        print(f"  Gradient accum: {cfg.gradient_accumulation_steps}")
        print(f"  AMP: {use_amp}")
        print(f"{'='*60}")

        for epoch in range(1, cfg.num_epochs + 1):
            train_loss = self._train_epoch(
                train_loader, optimizer, scheduler, scaler, use_amp
            )
            val_loss, val_f1 = self._eval_epoch(val_loader)

            is_best = val_f1 > best_val_f1
            if is_best:
                best_val_f1 = val_f1
                no_improve_count = 0
                self.classifier.save(output_dir)
            else:
                no_improve_count += 1

            result = EpochResult(
                epoch=epoch,
                train_loss=train_loss,
                val_loss=val_loss,
                val_f1=val_f1,
                is_best=is_best,
            )
            self.history.append(result)

            marker = " ← best" if is_best else ""
            print(
                f"  Epoch {epoch}/{cfg.num_epochs}"
                f"  train_loss={train_loss:.4f}"
                f"  val_loss={val_loss:.4f}"
                f"  val_f1={val_f1:.4f}{marker}"
            )

            if no_improve_count >= cfg.early_stopping_patience:
                print(
                    f"  Early stopping: no improvement for "
                    f"{cfg.early_stopping_patience} epochs."
                )
                break

        print(f"\nBest val F1: {best_val_f1:.4f}  (checkpoint → {output_dir})")
        return self.history

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _train_epoch(
        self,
        loader: DataLoader,
        optimizer: torch.optim.Optimizer,
        scheduler,
        scaler: torch.cuda.amp.GradScaler,
        use_amp: bool,
    ) -> float:
        self.classifier.train()
        total_loss = 0.0
        optimizer.zero_grad()

        pbar = tqdm(loader, desc="  train", leave=False, unit="batch")
        for step, batch in enumerate(pbar, start=1):
            input_ids = batch["input_ids"].to(self.device)
            attention_mask = batch["attention_mask"].to(self.device)
            labels = batch["label"].to(self.device)

            with torch.autocast(device_type=self.device.split(":")[0], enabled=use_amp):
                outputs = self.classifier(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    labels=labels,
                )
                loss = outputs.loss / self.config.gradient_accumulation_steps

            scaler.scale(loss).backward()

            if step % self.config.gradient_accumulation_steps == 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(
                    self.classifier.parameters(), self.config.max_grad_norm
                )
                scaler.step(optimizer)
                scaler.update()
                scheduler.step()
                optimizer.zero_grad()

            total_loss += outputs.loss.item()
            pbar.set_postfix(loss=f"{outputs.loss.item():.4f}")

        return total_loss / len(loader)

    @torch.no_grad()
    def _eval_epoch(self, loader: DataLoader) -> tuple[float, float]:
        self.classifier.eval()
        total_loss = 0.0
        all_preds: list[int] = []
        all_labels: list[int] = []

        for batch in tqdm(loader, desc="  val  ", leave=False, unit="batch"):
            input_ids = batch["input_ids"].to(self.device)
            attention_mask = batch["attention_mask"].to(self.device)
            labels = batch["label"].to(self.device)

            outputs = self.classifier(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels,
            )
            total_loss += outputs.loss.item()
            preds = outputs.logits.argmax(dim=-1).cpu().tolist()
            all_preds.extend(preds)
            all_labels.extend(labels.cpu().tolist())

        avg_loss = total_loss / len(loader)
        f1 = f1_score(all_labels, all_preds, average="weighted")
        return avg_loss, f1
