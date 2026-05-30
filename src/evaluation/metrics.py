"""
Evaluation metrics for few-shot defect classification.

Provides comprehensive evaluation utilities:
* Per-episode and overall accuracy
* Precision, recall, and F1 scores (per-class and macro/weighted)
* Classification reports
* Confusion matrix computation
* K-shot performance comparison (1-shot, 5-shot, 10-shot)
* JSON/CSV result persistence

Usage:
    >>> from src.evaluation.metrics import compute_accuracy, compute_precision_recall_f1
    >>> acc = compute_accuracy(predictions, labels)
    >>> report = compute_precision_recall_f1(predictions, labels, class_names)
"""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix as sk_confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)


# ---------------------------------------------------------------------------
# Core Metrics
# ---------------------------------------------------------------------------

def compute_accuracy(
    predictions: torch.Tensor | np.ndarray,
    labels: torch.Tensor | np.ndarray,
) -> float:
    """Compute classification accuracy.

    Args:
        predictions: Predicted class indices ``(N,)``.
        labels: Ground-truth class indices ``(N,)``.

    Returns:
        Accuracy as a float in ``[0, 1]``.
    """
    preds = _to_numpy(predictions)
    lbls = _to_numpy(labels)
    return float(accuracy_score(lbls, preds))


def compute_precision_recall_f1(
    predictions: torch.Tensor | np.ndarray,
    labels: torch.Tensor | np.ndarray,
    class_names: Optional[List[str]] = None,
    average: str = "macro",
) -> Dict[str, Any]:
    """Compute precision, recall, and F1 score.

    Args:
        predictions: Predicted class indices ``(N,)``.
        labels: Ground-truth class indices ``(N,)``.
        class_names: Optional class name list for the report.
        average: Averaging method (``'macro'``, ``'weighted'``, ``'micro'``).

    Returns:
        Dictionary with keys:
        * ``'precision'``, ``'recall'``, ``'f1'`` – overall averages.
        * ``'per_class'`` – per-class breakdown (if class_names provided).
        * ``'classification_report'`` – full text report.
    """
    preds = _to_numpy(predictions)
    lbls = _to_numpy(labels)

    precision = float(precision_score(lbls, preds, average=average, zero_division=0))
    recall = float(recall_score(lbls, preds, average=average, zero_division=0))
    f1 = float(f1_score(lbls, preds, average=average, zero_division=0))

    result: Dict[str, Any] = {
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "accuracy": float(accuracy_score(lbls, preds)),
    }

    # Per-class metrics
    if class_names is not None:
        per_class_precision = precision_score(
            lbls, preds, average=None, zero_division=0
        )
        per_class_recall = recall_score(
            lbls, preds, average=None, zero_division=0
        )
        per_class_f1 = f1_score(
            lbls, preds, average=None, zero_division=0
        )

        per_class = {}
        for idx, name in enumerate(class_names):
            if idx < len(per_class_precision):
                per_class[name] = {
                    "precision": float(per_class_precision[idx]),
                    "recall": float(per_class_recall[idx]),
                    "f1": float(per_class_f1[idx]),
                }
        result["per_class"] = per_class

    # Full classification report
    report = classification_report(
        lbls, preds,
        target_names=class_names,
        zero_division=0,
    )
    result["classification_report"] = report

    return result


def compute_confusion_matrix(
    predictions: torch.Tensor | np.ndarray,
    labels: torch.Tensor | np.ndarray,
    class_names: Optional[List[str]] = None,
    normalize: Optional[str] = None,
) -> Dict[str, Any]:
    """Compute the confusion matrix.

    Args:
        predictions: Predicted class indices ``(N,)``.
        labels: Ground-truth class indices ``(N,)``.
        class_names: Class names for labelling.
        normalize: Normalisation mode (``'true'``, ``'pred'``, ``'all'``, or ``None``).

    Returns:
        Dictionary with:
        * ``'matrix'`` – 2D numpy array (n_classes × n_classes).
        * ``'class_names'`` – list of class names.
    """
    preds = _to_numpy(predictions)
    lbls = _to_numpy(labels)

    cm = sk_confusion_matrix(lbls, preds, normalize=normalize)

    return {
        "matrix": cm,
        "class_names": class_names or [str(i) for i in range(cm.shape[0])],
    }


# ---------------------------------------------------------------------------
# K-Shot Evaluation
# ---------------------------------------------------------------------------

def evaluate_k_shot(
    model: torch.nn.Module,
    dataset: Any,
    k_values: List[int] = None,
    n_way: int = 4,
    q_query: int = 15,
    num_episodes: int = 50,
    device: str = "cpu",
    class_names: Optional[List[str]] = None,
) -> Dict[int, Dict[str, float]]:
    """Compare model performance across different K-shot settings.

    Runs multiple evaluation episodes for each K value and reports
    mean accuracy, precision, recall, and F1.

    Args:
        model: Trained :class:`PrototypicalNetwork`.
        dataset: A :class:`DefectDataset` with ``class_to_indices``.
        k_values: List of shot values to evaluate (default: [1, 5, 10]).
        n_way: Number of classes per episode.
        q_query: Number of query images per class.
        num_episodes: Number of evaluation episodes per K value.
        device: Compute device.
        class_names: Class names for reporting.

    Returns:
        Dictionary mapping K value → metric dictionary.
    """
    import random

    if k_values is None:
        k_values = [1, 5, 10]

    model.eval()
    results: Dict[int, Dict[str, float]] = {}

    for k in k_values:
        all_preds: List[int] = []
        all_labels: List[int] = []

        # Check which classes have enough samples
        eligible_classes = [
            cls_label
            for cls_label, indices in dataset.class_to_indices.items()
            if len(indices) >= (k + q_query)
        ]

        if len(eligible_classes) < n_way:
            print(f"Skipping K={k}: not enough classes with ≥ {k + q_query} images")
            continue

        for _ in range(num_episodes):
            # Sample episode
            episode_classes = random.sample(eligible_classes, n_way)

            support_images = []
            support_labels = []
            query_images = []
            query_labels = []

            for local_label, cls_label in enumerate(episode_classes):
                indices = random.sample(
                    dataset.class_to_indices[cls_label],
                    k + q_query,
                )
                # Support
                for idx in indices[:k]:
                    img, _ = dataset[idx]
                    support_images.append(img)
                    support_labels.append(local_label)
                # Query
                for idx in indices[k:]:
                    img, _ = dataset[idx]
                    query_images.append(img)
                    query_labels.append(local_label)

            support_images_t = torch.stack(support_images).to(device)
            support_labels_t = torch.tensor(support_labels, dtype=torch.long).to(device)
            query_images_t = torch.stack(query_images).to(device)
            query_labels_t = torch.tensor(query_labels, dtype=torch.long)

            with torch.no_grad():
                log_probs = model(
                    support_images_t, support_labels_t, query_images_t
                )
                preds = log_probs.argmax(dim=1).cpu()

            all_preds.extend(preds.tolist())
            all_labels.extend(query_labels_t.tolist())

        # Compute metrics for this K
        metrics = compute_precision_recall_f1(
            np.array(all_preds),
            np.array(all_labels),
            class_names=class_names,
        )
        results[k] = {
            "k_shot": k,
            "accuracy": metrics["accuracy"],
            "precision": metrics["precision"],
            "recall": metrics["recall"],
            "f1": metrics["f1"],
            "num_episodes": num_episodes,
        }

        print(
            f"  K={k:>2d}-shot │ "
            f"Acc: {metrics['accuracy']:.4f} │ "
            f"F1: {metrics['f1']:.4f} │ "
            f"Prec: {metrics['precision']:.4f} │ "
            f"Rec: {metrics['recall']:.4f}"
        )

    return results


# ---------------------------------------------------------------------------
# Result Persistence
# ---------------------------------------------------------------------------

def save_results_json(results: Dict[str, Any], path: str) -> None:
    """Save evaluation results to a JSON file.

    Args:
        results: Dictionary of evaluation results.
        path: Output file path.
    """
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)

    # Convert numpy arrays to lists for JSON serialisation
    serialisable = _make_serialisable(results)

    with open(out, "w", encoding="utf-8") as f:
        json.dump(serialisable, f, indent=2)
    print(f"Results saved to {out}")


def save_results_csv(
    results: List[Dict[str, Any]],
    path: str,
    fieldnames: Optional[List[str]] = None,
) -> None:
    """Save evaluation results to a CSV file.

    Args:
        results: List of result dictionaries (one per row).
        path: Output file path.
        fieldnames: CSV column names. Auto-detected if ``None``.
    """
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)

    if fieldnames is None and results:
        fieldnames = list(results[0].keys())

    with open(out, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames or [])
        writer.writeheader()
        for row in results:
            writer.writerow({k: v for k, v in row.items() if k in fieldnames})
    print(f"Results saved to {out}")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _to_numpy(x: torch.Tensor | np.ndarray) -> np.ndarray:
    """Convert input to a numpy array."""
    if isinstance(x, torch.Tensor):
        return x.detach().cpu().numpy()
    return np.asarray(x)


def _make_serialisable(obj: Any) -> Any:
    """Recursively convert numpy types to Python natives for JSON."""
    if isinstance(obj, dict):
        return {k: _make_serialisable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_make_serialisable(v) for v in obj]
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    return obj
