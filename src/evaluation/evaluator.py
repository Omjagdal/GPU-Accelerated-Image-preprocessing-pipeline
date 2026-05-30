"""
Few-Shot Evaluator with Visualisation Suite.

Combines metric computation with rich visualisations:
* Confusion matrix heatmaps
* Accuracy vs. K-shot line plots
* Training loss/accuracy curves
* t-SNE embedding visualisations

All plots are saved as high-resolution PNGs to the results directory.

Usage:
    >>> from src.evaluation.evaluator import FewShotEvaluator
    >>> evaluator = FewShotEvaluator(model, config, device="cuda")
    >>> results = evaluator.full_evaluation(test_loader, dataset)
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import matplotlib
matplotlib.use("Agg")  # Non-interactive backend for server environments
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import torch
import torch.nn as nn
from sklearn.manifold import TSNE

from src.evaluation.metrics import (
    compute_accuracy,
    compute_confusion_matrix,
    compute_precision_recall_f1,
    evaluate_k_shot,
    save_results_json,
)


# ---------------------------------------------------------------------------
# Plot styling
# ---------------------------------------------------------------------------

def _setup_plot_style() -> None:
    """Configure a publication-quality plot style."""
    plt.style.use("seaborn-v0_8-whitegrid")
    plt.rcParams.update({
        "figure.figsize": (10, 8),
        "figure.dpi": 150,
        "font.size": 12,
        "font.family": "sans-serif",
        "axes.titlesize": 14,
        "axes.labelsize": 12,
        "legend.fontsize": 10,
        "savefig.bbox": "tight",
        "savefig.pad_inches": 0.2,
    })


# ---------------------------------------------------------------------------
# Visualisation Functions
# ---------------------------------------------------------------------------

def plot_confusion_matrix(
    predictions: np.ndarray,
    labels: np.ndarray,
    class_names: List[str],
    save_path: str,
    title: str = "Confusion Matrix",
    normalize: bool = True,
) -> str:
    """Plot and save a confusion matrix heatmap.

    Args:
        predictions: Predicted class indices.
        labels: Ground-truth class indices.
        class_names: Class name labels.
        save_path: File path for the saved figure.
        title: Plot title.
        normalize: If True, show percentages; otherwise raw counts.

    Returns:
        The resolved save path as a string.
    """
    _setup_plot_style()

    cm_result = compute_confusion_matrix(
        predictions, labels,
        class_names=class_names,
        normalize="true" if normalize else None,
    )
    cm = cm_result["matrix"]

    fig, ax = plt.subplots(figsize=(8, 6))

    if normalize:
        fmt = ".1%"
        cm_display = cm
    else:
        fmt = "d"
        cm_display = cm.astype(int)

    sns.heatmap(
        cm_display,
        annot=True,
        fmt=fmt,
        cmap="Blues",
        xticklabels=class_names,
        yticklabels=class_names,
        ax=ax,
        cbar_kws={"shrink": 0.8},
        linewidths=0.5,
        linecolor="white",
        square=True,
    )

    ax.set_xlabel("Predicted Label", fontsize=13, fontweight="bold")
    ax.set_ylabel("True Label", fontsize=13, fontweight="bold")
    ax.set_title(title, fontsize=15, fontweight="bold", pad=15)

    # Rotate labels for readability
    ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha="right")
    ax.set_yticklabels(ax.get_yticklabels(), rotation=0)

    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path, dpi=150)
    plt.close(fig)

    print(f"Confusion matrix saved to {save_path}")
    return str(Path(save_path).resolve())


def plot_accuracy_vs_shots(
    k_shot_results: Dict[int, Dict[str, float]],
    save_path: str,
    title: str = "Accuracy vs. Number of Shots (K)",
) -> str:
    """Plot accuracy as a function of the number of support shots.

    Args:
        k_shot_results: Dictionary mapping K → metric dict with ``'accuracy'``.
        save_path: File path for the saved figure.
        title: Plot title.

    Returns:
        The resolved save path.
    """
    _setup_plot_style()

    k_values = sorted(k_shot_results.keys())
    accuracies = [k_shot_results[k]["accuracy"] for k in k_values]
    f1_scores = [k_shot_results[k].get("f1", 0) for k in k_values]

    fig, ax = plt.subplots(figsize=(10, 6))

    # Accuracy line
    ax.plot(
        k_values, accuracies,
        "o-", color="#2196F3", linewidth=2.5, markersize=10,
        label="Accuracy", zorder=5,
    )
    # F1 line
    ax.plot(
        k_values, f1_scores,
        "s--", color="#FF9800", linewidth=2, markersize=8,
        label="F1 Score", zorder=4,
    )

    # Annotations
    for k, acc in zip(k_values, accuracies):
        ax.annotate(
            f"{acc:.1%}",
            (k, acc),
            textcoords="offset points",
            xytext=(0, 12),
            ha="center",
            fontsize=11,
            fontweight="bold",
            color="#1565C0",
        )

    ax.set_xlabel("Number of Shots (K)", fontsize=13, fontweight="bold")
    ax.set_ylabel("Score", fontsize=13, fontweight="bold")
    ax.set_title(title, fontsize=15, fontweight="bold", pad=15)
    ax.set_xticks(k_values)
    ax.set_ylim(0, 1.05)
    ax.legend(loc="lower right", framealpha=0.9)
    ax.grid(True, alpha=0.3)

    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path, dpi=150)
    plt.close(fig)

    print(f"Accuracy vs shots plot saved to {save_path}")
    return str(Path(save_path).resolve())


def plot_training_curves(
    history: Dict[str, List[float]],
    save_path: str,
    title: str = "Training Curves",
) -> str:
    """Plot training and validation loss/accuracy curves.

    Args:
        history: Dictionary with ``'train_loss'``, ``'train_acc'``,
            ``'val_loss'``, ``'val_acc'`` lists.
        save_path: File path for the saved figure.
        title: Plot title.

    Returns:
        The resolved save path.
    """
    _setup_plot_style()

    epochs = list(range(1, len(history.get("train_loss", [])) + 1))
    if not epochs:
        print("No training history to plot.")
        return save_path

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    # ----- Loss subplot -----
    ax1.plot(
        epochs, history["train_loss"],
        "-", color="#E53935", linewidth=2, label="Train Loss",
    )
    if history.get("val_loss") and any(v > 0 for v in history["val_loss"]):
        ax1.plot(
            epochs, history["val_loss"],
            "--", color="#1E88E5", linewidth=2, label="Val Loss",
        )
    ax1.set_xlabel("Epoch", fontsize=12)
    ax1.set_ylabel("Loss", fontsize=12)
    ax1.set_title("Loss Curve", fontsize=13, fontweight="bold")
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # ----- Accuracy subplot -----
    ax2.plot(
        epochs, history["train_acc"],
        "-", color="#43A047", linewidth=2, label="Train Accuracy",
    )
    if history.get("val_acc") and any(v > 0 for v in history["val_acc"]):
        ax2.plot(
            epochs, history["val_acc"],
            "--", color="#FB8C00", linewidth=2, label="Val Accuracy",
        )
    ax2.set_xlabel("Epoch", fontsize=12)
    ax2.set_ylabel("Accuracy", fontsize=12)
    ax2.set_title("Accuracy Curve", fontsize=13, fontweight="bold")
    ax2.set_ylim(0, 1.05)
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    fig.suptitle(title, fontsize=15, fontweight="bold", y=1.02)
    fig.tight_layout()

    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path, dpi=150)
    plt.close(fig)

    print(f"Training curves saved to {save_path}")
    return str(Path(save_path).resolve())


def plot_embedding_tsne(
    model: nn.Module,
    dataset: Any,
    class_names: List[str],
    save_path: str,
    num_samples_per_class: int = 30,
    perplexity: float = 15.0,
    device: str = "cpu",
    title: str = "t-SNE Embedding Visualization",
) -> str:
    """Visualise learned embeddings using t-SNE dimensionality reduction.

    Args:
        model: Trained :class:`PrototypicalNetwork` or backbone.
        dataset: Dataset with ``class_to_indices`` attribute.
        class_names: List of class names.
        save_path: File path for the saved figure.
        num_samples_per_class: Number of samples per class to embed.
        perplexity: t-SNE perplexity parameter.
        device: Compute device.
        title: Plot title.

    Returns:
        The resolved save path.
    """
    import random

    _setup_plot_style()

    model.eval()
    embeddings_list: List[np.ndarray] = []
    labels_list: List[int] = []

    with torch.no_grad():
        for cls_label, indices in dataset.class_to_indices.items():
            n_samples = min(num_samples_per_class, len(indices))
            selected = random.sample(indices, n_samples)

            for idx in selected:
                img, _ = dataset[idx]
                img = img.unsqueeze(0).to(device)

                # Handle both PrototypicalNetwork and bare backbone
                if hasattr(model, "get_embedding"):
                    emb = model.get_embedding(img)
                elif hasattr(model, "backbone"):
                    emb = model.backbone(img)
                else:
                    emb = model(img)

                embeddings_list.append(emb.cpu().numpy().flatten())
                labels_list.append(cls_label)

    embeddings = np.array(embeddings_list)
    labels = np.array(labels_list)

    # Run t-SNE
    tsne = TSNE(
        n_components=2,
        perplexity=min(perplexity, len(embeddings) - 1),
        random_state=42,
        n_iter=1000,
    )
    embeddings_2d = tsne.fit_transform(embeddings)

    # Plot
    fig, ax = plt.subplots(figsize=(10, 8))

    colours = ["#E53935", "#1E88E5", "#43A047", "#FB8C00", "#8E24AA",
               "#00ACC1", "#F4511E", "#6D4C41"]

    for cls_idx, cls_name in enumerate(class_names):
        mask = labels == cls_idx
        if mask.sum() == 0:
            continue
        ax.scatter(
            embeddings_2d[mask, 0],
            embeddings_2d[mask, 1],
            c=colours[cls_idx % len(colours)],
            label=cls_name,
            alpha=0.7,
            s=60,
            edgecolors="white",
            linewidth=0.5,
        )

    ax.set_xlabel("t-SNE Dimension 1", fontsize=12)
    ax.set_ylabel("t-SNE Dimension 2", fontsize=12)
    ax.set_title(title, fontsize=15, fontweight="bold", pad=15)
    ax.legend(
        loc="best",
        framealpha=0.9,
        fontsize=11,
        markerscale=1.5,
    )
    ax.grid(True, alpha=0.2)

    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path, dpi=150)
    plt.close(fig)

    print(f"t-SNE embedding plot saved to {save_path}")
    return str(Path(save_path).resolve())


# ---------------------------------------------------------------------------
# FewShotEvaluator (orchestrator)
# ---------------------------------------------------------------------------

class FewShotEvaluator:
    """Orchestrates comprehensive few-shot evaluation and visualisation.

    Combines metric computation with visualisation generation into a
    single ``full_evaluation()`` call.

    Args:
        model: Trained :class:`PrototypicalNetwork`.
        config: Project configuration.
        device: Compute device.
    """

    def __init__(
        self,
        model: nn.Module,
        config: Any,
        device: str = "cpu",
    ) -> None:
        self.model = model
        self.config = config
        self.device = device
        self.class_names: List[str] = config.data.classes

        # Output directories
        self.results_dir = Path(config.results_dir)
        self.plots_dir = self.results_dir / "accuracy_plots"
        self.cm_dir = self.results_dir / "confusion_matrices"
        self.latency_dir = self.results_dir / "latency_reports"

        for d in (self.plots_dir, self.cm_dir, self.latency_dir):
            d.mkdir(parents=True, exist_ok=True)

    def full_evaluation(
        self,
        test_loader: Any,
        test_dataset: Any,
        training_history: Optional[Dict[str, List[float]]] = None,
    ) -> Dict[str, Any]:
        """Run comprehensive evaluation: metrics, K-shot comparison, and plots.

        Args:
            test_loader: Episodic test data loader.
            test_dataset: Test dataset for K-shot evaluation.
            training_history: Optional training history for loss/acc curves.

        Returns:
            Dictionary of all evaluation results.
        """
        print("\n" + "=" * 60)
        print("  Running Full Evaluation Suite")
        print("=" * 60)

        results: Dict[str, Any] = {}

        # 1. Episodic evaluation
        print("\n[1/5] Episodic evaluation on test set...")
        all_preds, all_labels = self._collect_predictions(test_loader)
        metrics = compute_precision_recall_f1(
            all_preds, all_labels,
            class_names=self.class_names,
        )
        results["episode_metrics"] = metrics
        print(f"  Accuracy: {metrics['accuracy']:.4f}")
        print(f"  F1 Score: {metrics['f1']:.4f}")
        print(metrics["classification_report"])

        # 2. Confusion matrix
        print("\n[2/5] Generating confusion matrix...")
        cm_path = str(self.cm_dir / "confusion_matrix.png")
        plot_confusion_matrix(
            all_preds, all_labels,
            self.class_names, cm_path,
        )
        results["confusion_matrix_path"] = cm_path

        # 3. K-shot comparison
        print("\n[3/5] K-shot performance comparison...")
        k_shot_results = evaluate_k_shot(
            self.model, test_dataset,
            k_values=[1, 5, 10],
            n_way=min(self.config.training.n_way, len(self.class_names)),
            q_query=min(self.config.training.q_query, 10),
            num_episodes=30,
            device=self.device,
            class_names=self.class_names,
        )
        results["k_shot_results"] = k_shot_results

        if k_shot_results:
            acc_path = str(self.plots_dir / "accuracy_vs_shots.png")
            plot_accuracy_vs_shots(k_shot_results, acc_path)
            results["accuracy_vs_shots_path"] = acc_path

        # 4. Training curves
        if training_history:
            print("\n[4/5] Plotting training curves...")
            curves_path = str(self.plots_dir / "training_curves.png")
            plot_training_curves(training_history, curves_path)
            results["training_curves_path"] = curves_path
        else:
            print("\n[4/5] Skipping training curves (no history provided)")

        # 5. t-SNE embedding visualization
        print("\n[5/5] Generating t-SNE embedding visualization...")
        try:
            tsne_path = str(self.plots_dir / "embedding_tsne.png")
            plot_embedding_tsne(
                self.model, test_dataset,
                self.class_names, tsne_path,
                device=self.device,
            )
            results["tsne_path"] = tsne_path
        except Exception as e:
            print(f"  t-SNE visualization skipped: {e}")

        # Save results JSON
        results_json_path = str(self.results_dir / "evaluation_results.json")
        save_results_json(
            {k: v for k, v in results.items() if not k.endswith("_path")},
            results_json_path,
        )

        print("\n" + "=" * 60)
        print("  Evaluation Complete")
        print("=" * 60)

        return results

    def _collect_predictions(
        self,
        dataloader: Any,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Collect all predictions from an episodic data loader.

        Returns:
            ``(predictions, labels)`` numpy arrays.
        """
        self.model.eval()
        all_preds: List[int] = []
        all_labels: List[int] = []

        with torch.no_grad():
            for episode in dataloader:
                support_images = episode["support_images"].to(self.device)
                support_labels = episode["support_labels"].to(self.device)
                query_images = episode["query_images"].to(self.device)
                query_labels = episode["query_labels"]

                log_probs = self.model(
                    support_images, support_labels, query_images,
                )
                preds = log_probs.argmax(dim=1).cpu()

                all_preds.extend(preds.tolist())
                all_labels.extend(query_labels.tolist())

        return np.array(all_preds), np.array(all_labels)
