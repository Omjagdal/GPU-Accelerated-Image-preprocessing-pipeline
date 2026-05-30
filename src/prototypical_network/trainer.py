"""
Episodic training loop for Prototypical Networks.

Implements the full training pipeline with:
* Episodic N-way K-shot training with prototypical loss
* Learning rate scheduling (StepLR)
* Early stopping with configurable patience
* Best-model checkpointing
* Mixed-precision training (AMP) support
* Gradient clipping
* Comprehensive metric logging via :class:`~src.utils.logger.TrainingLogger`

Usage:
    >>> from src.prototypical_network.trainer import ProtoNetTrainer
    >>> trainer = ProtoNetTrainer(model, config, device="cuda")
    >>> history = trainer.train(train_loader, val_loader)
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.cuda.amp import GradScaler, autocast
from torch.optim import Adam
from torch.optim.lr_scheduler import StepLR
from torch.utils.data import DataLoader

from src.utils.logger import TrainingLogger


class ProtoNetTrainer:
    """Episodic trainer for Prototypical Networks.

    Manages the complete training lifecycle including optimisation,
    scheduling, early stopping, checkpointing, and logging.

    Attributes:
        model (nn.Module): The :class:`PrototypicalNetwork` to train.
        optimizer (Adam): Parameter optimizer.
        scheduler (StepLR): Learning rate scheduler.
        device (torch.device): Compute device.
        logger (TrainingLogger): Structured metric logger.
        best_val_acc (float): Best validation accuracy observed.
        best_model_path (Path): Path where the best model is saved.

    Args:
        model: A :class:`~src.prototypical_network.model.PrototypicalNetwork`.
        config: A configuration object with ``training`` and ``deployment``
            sub-configs (see :class:`~src.utils.config.ProjectConfig`).
        device: Target device string (``'cuda'``, ``'mps'``, ``'cpu'``).
        log_dir: Directory for log files.
        model_dir: Directory for model checkpoints.
    """

    def __init__(
        self,
        model: nn.Module,
        config: Any,
        device: str = "cpu",
        log_dir: str = "logs",
        model_dir: str = "models",
    ) -> None:
        self.model = model.to(device)
        self.config = config
        self.device = torch.device(device)

        # Training hyper-parameters
        tc = config.training
        self.epochs: int = tc.epochs
        self.patience: int = tc.patience
        self.min_delta: float = tc.min_delta
        self.gradient_clip_norm: float = tc.gradient_clip_norm
        self.mixed_precision: bool = tc.mixed_precision

        # Optimiser & scheduler
        self.optimizer = Adam(
            model.parameters(),
            lr=tc.lr,
            weight_decay=tc.weight_decay,
        )
        self.scheduler = StepLR(
            self.optimizer,
            step_size=tc.lr_step,
            gamma=tc.lr_gamma,
        )

        # Mixed-precision scaler
        self.scaler = GradScaler(enabled=self.mixed_precision and device == "cuda")

        # Logging
        self.logger = TrainingLogger(
            name="protonet_trainer",
            log_file=str(Path(log_dir) / "training.log"),
            csv_path=str(Path(log_dir) / "training_metrics.csv"),
        )

        # Checkpointing
        self.model_dir = Path(model_dir)
        self.model_dir.mkdir(parents=True, exist_ok=True)
        self.best_model_path = self.model_dir / "best_model.pth"
        self.best_val_acc: float = 0.0

        # Training history
        self.history: Dict[str, List[float]] = {
            "train_loss": [],
            "train_acc": [],
            "val_loss": [],
            "val_acc": [],
            "learning_rate": [],
        }

    # ------------------------------------------------------------------
    # Single episode
    # ------------------------------------------------------------------

    def _run_episode(
        self,
        episode: Dict[str, torch.Tensor],
        training: bool = True,
    ) -> Tuple[float, float]:
        """Run a single N-way K-shot episode.

        Args:
            episode: Dictionary with keys ``'support_images'``,
                ``'support_labels'``, ``'query_images'``, ``'query_labels'``.
            training: Whether to compute gradients and update parameters.

        Returns:
            A ``(loss, accuracy)`` tuple for this episode.
        """
        support_images = episode["support_images"].to(self.device)
        support_labels = episode["support_labels"].to(self.device)
        query_images = episode["query_images"].to(self.device)
        query_labels = episode["query_labels"].to(self.device)

        if training:
            self.model.train()
        else:
            self.model.eval()

        context = torch.enable_grad() if training else torch.no_grad()

        with context:
            with autocast(enabled=self.mixed_precision and str(self.device) == "cuda"):
                # Forward pass: get log-probabilities
                log_probs = self.model(
                    support_images, support_labels, query_images
                )  # (N_q, n_way)

                # Prototypical loss: NLL on log-softmax of negative distances
                loss = F.nll_loss(log_probs, query_labels)

            # Accuracy
            predictions = log_probs.argmax(dim=1)
            correct = (predictions == query_labels).sum().item()
            accuracy = correct / query_labels.size(0)

            if training:
                self.optimizer.zero_grad()
                if self.mixed_precision and str(self.device) == "cuda":
                    self.scaler.scale(loss).backward()
                    if self.gradient_clip_norm > 0:
                        self.scaler.unscale_(self.optimizer)
                        nn.utils.clip_grad_norm_(
                            self.model.parameters(),
                            self.gradient_clip_norm,
                        )
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                else:
                    loss.backward()
                    if self.gradient_clip_norm > 0:
                        nn.utils.clip_grad_norm_(
                            self.model.parameters(),
                            self.gradient_clip_norm,
                        )
                    self.optimizer.step()

        return loss.item(), accuracy

    # ------------------------------------------------------------------
    # Epoch routines
    # ------------------------------------------------------------------

    def train_epoch(
        self,
        dataloader: DataLoader,
        epoch: int,
    ) -> Tuple[float, float]:
        """Run all episodes in one training epoch.

        Args:
            dataloader: Episodic training data loader.
            epoch: Current epoch number (for logging).

        Returns:
            ``(mean_loss, mean_accuracy)`` over all episodes.
        """
        self.model.train()
        total_loss = 0.0
        total_acc = 0.0
        num_episodes = 0

        lr = self.optimizer.param_groups[0]["lr"]

        for episode_idx, episode in enumerate(dataloader):
            loss, acc = self._run_episode(episode, training=True)
            total_loss += loss
            total_acc += acc
            num_episodes += 1

            # Log every episode
            self.logger.log_episode(
                epoch=epoch,
                episode=episode_idx + 1,
                loss=loss,
                accuracy=acc,
                learning_rate=lr,
                phase="train",
            )

        mean_loss = total_loss / max(num_episodes, 1)
        mean_acc = total_acc / max(num_episodes, 1)
        return mean_loss, mean_acc

    def evaluate(
        self,
        dataloader: DataLoader,
        epoch: int = 0,
    ) -> Tuple[float, float]:
        """Evaluate the model on a validation/test set.

        Args:
            dataloader: Episodic validation/test data loader.
            epoch: Current epoch number (for logging).

        Returns:
            ``(mean_loss, mean_accuracy)`` over all episodes.
        """
        self.model.eval()
        total_loss = 0.0
        total_acc = 0.0
        num_episodes = 0

        with torch.no_grad():
            for episode_idx, episode in enumerate(dataloader):
                loss, acc = self._run_episode(episode, training=False)
                total_loss += loss
                total_acc += acc
                num_episodes += 1

                self.logger.log_episode(
                    epoch=epoch,
                    episode=episode_idx + 1,
                    loss=loss,
                    accuracy=acc,
                    phase="val",
                )

        mean_loss = total_loss / max(num_episodes, 1)
        mean_acc = total_acc / max(num_episodes, 1)
        return mean_loss, mean_acc

    # ------------------------------------------------------------------
    # Full training loop
    # ------------------------------------------------------------------

    def train(
        self,
        train_loader: DataLoader,
        val_loader: Optional[DataLoader] = None,
    ) -> Dict[str, List[float]]:
        """Run the complete training loop with early stopping.

        Args:
            train_loader: Episodic training data loader.
            val_loader: Optional episodic validation data loader.
                If ``None``, training accuracy is used for checkpointing.

        Returns:
            Dictionary of training history lists:
            ``{'train_loss', 'train_acc', 'val_loss', 'val_acc', 'learning_rate'}``.
        """
        self.logger.log_info(
            f"Starting training: {self.epochs} epochs on {self.device}"
        )
        self.logger.log_info(f"Model: {self.model}")

        patience_counter = 0

        for epoch in range(1, self.epochs + 1):
            epoch_start = time.time()

            # ----- Training epoch -----
            train_loss, train_acc = self.train_epoch(train_loader, epoch)
            self.history["train_loss"].append(train_loss)
            self.history["train_acc"].append(train_acc)

            # ----- Validation epoch -----
            val_loss, val_acc = 0.0, 0.0
            if val_loader is not None:
                val_loss, val_acc = self.evaluate(val_loader, epoch)
            self.history["val_loss"].append(val_loss)
            self.history["val_acc"].append(val_acc)

            # Current learning rate
            current_lr = self.optimizer.param_groups[0]["lr"]
            self.history["learning_rate"].append(current_lr)

            # Step the scheduler
            self.scheduler.step()

            # Epoch duration
            elapsed = time.time() - epoch_start

            # ----- Logging -----
            self.logger.log_epoch_summary(
                epoch=epoch,
                train_loss=train_loss,
                train_acc=train_acc,
                val_loss=val_loss if val_loader else None,
                val_acc=val_acc if val_loader else None,
                best_val_acc=self.best_val_acc,
                elapsed_seconds=elapsed,
            )

            # ----- Checkpointing & Early Stopping -----
            monitor_acc = val_acc if val_loader else train_acc
            if monitor_acc > self.best_val_acc + self.min_delta:
                self.best_val_acc = monitor_acc
                patience_counter = 0
                self._save_checkpoint(epoch, monitor_acc)
                self.logger.log_info(
                    f"✓ New best model saved (acc={monitor_acc:.4f})"
                )
            else:
                patience_counter += 1
                self.logger.log_info(
                    f"No improvement for {patience_counter}/{self.patience} epochs"
                )

            if patience_counter >= self.patience:
                self.logger.log_info(
                    f"Early stopping triggered at epoch {epoch}"
                )
                break

        self.logger.log_info(
            f"Training complete. Best accuracy: {self.best_val_acc:.4f}"
        )

        # Save final model
        self._save_checkpoint(epoch, monitor_acc, filename="final_model.pth")

        return self.history

    # ------------------------------------------------------------------
    # Checkpointing
    # ------------------------------------------------------------------

    def _save_checkpoint(
        self,
        epoch: int,
        accuracy: float,
        filename: str = "best_model.pth",
    ) -> None:
        """Save a model checkpoint.

        Args:
            epoch: Current epoch.
            accuracy: Accuracy at checkpoint time.
            filename: Checkpoint filename.
        """
        checkpoint = {
            "epoch": epoch,
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "scheduler_state_dict": self.scheduler.state_dict(),
            "best_val_acc": self.best_val_acc,
            "accuracy": accuracy,
            "config": self.config.to_dict() if hasattr(self.config, "to_dict") else {},
        }
        save_path = self.model_dir / filename
        torch.save(checkpoint, save_path)

    def load_checkpoint(self, path: str) -> None:
        """Load a model checkpoint.

        Args:
            path: Path to the ``.pth`` checkpoint file.
        """
        checkpoint = torch.load(path, map_location=self.device)
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        if "scheduler_state_dict" in checkpoint:
            self.scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
        self.best_val_acc = checkpoint.get("best_val_acc", 0.0)
        self.logger.log_info(
            f"Loaded checkpoint from {path} "
            f"(epoch {checkpoint.get('epoch', '?')}, "
            f"acc={checkpoint.get('accuracy', '?')})"
        )
