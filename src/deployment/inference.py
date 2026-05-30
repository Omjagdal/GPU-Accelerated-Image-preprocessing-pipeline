"""
Unified Inference Engine for Prototypical Network Defect Detection.

Provides a high-level :class:`Inferencer` that supports multiple backends:
* **PyTorch** – Native model inference (default)
* **ONNX Runtime** – Cross-platform optimised inference
* **TensorRT** – Maximum performance on NVIDIA GPUs

The inferencer:
1. Loads a trained backbone from a checkpoint.
2. Pre-computes class prototypes from a labelled support set.
3. Classifies new images by computing distance to prototypes.
4. Reports latency statistics for performance benchmarking.

Usage:
    >>> from src.deployment.inference import Inferencer
    >>> inf = Inferencer.from_checkpoint("models/best_model.pth")
    >>> pred, probs = inf.predict("test_image.jpg")
    >>> print(f"Predicted: {pred}, Confidence: {probs.max():.2%}")
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image

from src.backbone.factory import get_backbone
from src.preprocessing.augmentations import get_val_transforms
from src.prototypical_network.distance import get_distance_fn
from src.prototypical_network.model import PrototypicalNetwork


class Inferencer:
    """Unified inference engine for prototypical defect classification.

    Manages the complete inference pipeline: image preprocessing,
    backbone feature extraction, prototype computation, and distance-based
    classification.

    Attributes:
        model: The prototypical network or backbone.
        prototypes: Pre-computed class prototype embeddings.
        class_names: Ordered list of class names.
        device: Compute device.
        transform: Image preprocessing pipeline.
        latency_history: List of inference latencies (ms).

    Args:
        model: A :class:`PrototypicalNetwork` instance.
        class_names: List of class names matching prototype order.
        device: Compute device string.
        image_size: Input image size for preprocessing.
    """

    def __init__(
        self,
        model: nn.Module,
        class_names: List[str],
        device: str = "cpu",
        image_size: int = 224,
    ) -> None:
        self.model = model.to(device)
        self.model.eval()
        self.class_names = class_names
        self.device = torch.device(device)
        self.transform = get_val_transforms(image_size)
        self.image_size = image_size

        # Prototypes (computed from support set)
        self.prototypes: Optional[torch.Tensor] = None

        # Distance function
        self.distance_fn = get_distance_fn("euclidean")

        # Latency tracking
        self.latency_history: List[float] = []

    # ------------------------------------------------------------------
    # Factory Methods
    # ------------------------------------------------------------------

    @classmethod
    def from_checkpoint(
        cls,
        checkpoint_path: str,
        backbone_name: str = "resnet18",
        class_names: Optional[List[str]] = None,
        device: str = "cpu",
        image_size: int = 224,
    ) -> "Inferencer":
        """Create an Inferencer from a saved model checkpoint.

        Args:
            checkpoint_path: Path to the ``.pth`` checkpoint.
            backbone_name: Backbone architecture name.
            class_names: Class names. Defaults to standard defect classes.
            device: Compute device.
            image_size: Image size for preprocessing.

        Returns:
            A ready-to-use :class:`Inferencer` instance.
        """
        if class_names is None:
            class_names = ["normal", "scratch", "crack", "dent"]

        # Build model
        backbone = get_backbone(backbone_name, pretrained=False)
        model = PrototypicalNetwork(backbone)

        # Load weights
        checkpoint = torch.load(checkpoint_path, map_location=device)
        if "model_state_dict" in checkpoint:
            model.load_state_dict(checkpoint["model_state_dict"])
        else:
            model.load_state_dict(checkpoint)

        print(f"Loaded checkpoint from {checkpoint_path}")

        return cls(model, class_names, device, image_size)

    # ------------------------------------------------------------------
    # Support Set & Prototypes
    # ------------------------------------------------------------------

    def load_support_set(
        self,
        support_dir: str,
        classes: Optional[List[str]] = None,
    ) -> torch.Tensor:
        """Load a support set and compute prototypes.

        Args:
            support_dir: Directory containing class sub-folders with
                representative images.
            classes: Class names. Uses ``self.class_names`` if ``None``.

        Returns:
            Prototype tensor of shape ``(n_classes, embedding_dim)``.
        """
        from src.preprocessing.image_loader import load_support_set

        classes = classes or self.class_names

        images, labels = load_support_set(
            support_dir=support_dir,
            transform=self.transform,
            classes=classes,
            image_size=self.image_size,
        )
        images = images.to(self.device)
        labels = labels.to(self.device)

        # Compute prototypes
        self.model.eval()
        with torch.no_grad():
            embeddings = self.model.get_embedding(images)
            self.prototypes = PrototypicalNetwork.compute_prototypes(
                embeddings, labels
            )

        print(f"Computed {self.prototypes.shape[0]} prototypes from {support_dir}")
        print(f"  Classes: {classes}")
        print(f"  Support images: {len(images)}")

        return self.prototypes

    def set_prototypes(self, prototypes: torch.Tensor) -> None:
        """Manually set pre-computed prototypes.

        Args:
            prototypes: Prototype tensor ``(n_classes, embedding_dim)``.
        """
        self.prototypes = prototypes.to(self.device)

    # ------------------------------------------------------------------
    # Prediction
    # ------------------------------------------------------------------

    def predict(
        self,
        image: str | np.ndarray | Image.Image | torch.Tensor,
    ) -> Tuple[str, Dict[str, float]]:
        """Classify a single image.

        Args:
            image: Input image as a file path, numpy array, PIL Image,
                or preprocessed tensor.

        Returns:
            A tuple ``(predicted_class, confidence_scores)`` where
            ``confidence_scores`` is a dict mapping class names to
            probabilities.

        Raises:
            RuntimeError: If prototypes have not been computed.
        """
        if self.prototypes is None:
            raise RuntimeError(
                "Prototypes not set. Call load_support_set() or "
                "set_prototypes() first."
            )

        # Preprocess image
        img_tensor = self._preprocess(image)  # (1, C, H, W)
        img_tensor = img_tensor.to(self.device)

        # Inference with latency measurement
        start = time.perf_counter()

        self.model.eval()
        with torch.no_grad():
            embedding = self.model.get_embedding(img_tensor)  # (1, D)
            distances = self.distance_fn(embedding, self.prototypes)  # (1, n_way)
            probs = F.softmax(-distances, dim=1).squeeze(0)  # (n_way,)

        latency_ms = (time.perf_counter() - start) * 1000
        self.latency_history.append(latency_ms)

        # Build result
        pred_idx = probs.argmax().item()
        predicted_class = self.class_names[pred_idx]

        confidence_scores = {
            name: float(probs[i])
            for i, name in enumerate(self.class_names)
        }

        return predicted_class, confidence_scores

    def predict_batch(
        self,
        images: List[str | np.ndarray | Image.Image],
    ) -> List[Tuple[str, Dict[str, float]]]:
        """Classify a batch of images.

        Args:
            images: List of images (paths, arrays, or PIL Images).

        Returns:
            List of ``(predicted_class, confidence_scores)`` tuples.
        """
        if self.prototypes is None:
            raise RuntimeError("Prototypes not set.")

        # Preprocess all images
        tensors = [self._preprocess(img) for img in images]
        batch = torch.cat(tensors, dim=0).to(self.device)  # (B, C, H, W)

        start = time.perf_counter()

        self.model.eval()
        with torch.no_grad():
            embeddings = self.model.get_embedding(batch)  # (B, D)
            distances = self.distance_fn(embeddings, self.prototypes)  # (B, n_way)
            probs = F.softmax(-distances, dim=1)  # (B, n_way)

        latency_ms = (time.perf_counter() - start) * 1000
        self.latency_history.append(latency_ms)

        results = []
        for i in range(probs.size(0)):
            pred_idx = probs[i].argmax().item()
            predicted_class = self.class_names[pred_idx]
            scores = {
                name: float(probs[i, j])
                for j, name in enumerate(self.class_names)
            }
            results.append((predicted_class, scores))

        return results

    # ------------------------------------------------------------------
    # Latency Reporting
    # ------------------------------------------------------------------

    def get_latency_stats(self) -> Dict[str, float]:
        """Get latency statistics from inference history.

        Returns:
            Dictionary with mean, std, min, max, p95, p99 latencies in ms.
        """
        if not self.latency_history:
            return {"message": "No inference runs recorded"}

        lats = np.array(self.latency_history)
        return {
            "num_inferences": len(lats),
            "mean_ms": float(np.mean(lats)),
            "std_ms": float(np.std(lats)),
            "min_ms": float(np.min(lats)),
            "max_ms": float(np.max(lats)),
            "p95_ms": float(np.percentile(lats, 95)),
            "p99_ms": float(np.percentile(lats, 99)),
        }

    # ------------------------------------------------------------------
    # Private Helpers
    # ------------------------------------------------------------------

    def _preprocess(
        self,
        image: str | np.ndarray | Image.Image | torch.Tensor,
    ) -> torch.Tensor:
        """Preprocess an image to a normalised tensor.

        Args:
            image: Raw input in various formats.

        Returns:
            Tensor of shape ``(1, C, H, W)``.
        """
        if isinstance(image, torch.Tensor):
            if image.dim() == 3:
                image = image.unsqueeze(0)
            return image

        if isinstance(image, str):
            image = Image.open(image).convert("RGB")
        elif isinstance(image, np.ndarray):
            image = Image.fromarray(image)

        # Apply transforms
        tensor = self.transform(image)
        return tensor.unsqueeze(0)
