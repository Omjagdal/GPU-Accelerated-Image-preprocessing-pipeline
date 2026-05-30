"""
Prototypical Network for Few-Shot Defect Classification.

Implements the core Prototypical Network architecture from
*Prototypical Networks for Few-Shot Learning* (Snell et al., NeurIPS 2017).

The network:
1. Embeds support and query images through a shared CNN backbone.
2. Computes **class prototypes** by averaging support embeddings per class.
3. Classifies queries by computing distances to each prototype and
   returning log-probabilities via softmax over negative distances.

Architecture::

    ┌─────────────┐       ┌───────────┐       ┌──────────────┐
    │ Support Set  │──────▶│  Backbone  │──────▶│ Prototypes   │
    └─────────────┘       │ (shared)   │       │ (mean embed) │
                          │            │       └──────┬───────┘
    ┌─────────────┐       │            │              │  distance
    │ Query Set    │──────▶│            │──────▶───────┼──────────▶ log-probs
    └─────────────┘       └───────────┘              │
                                                     ▼
                                              Classification

Usage:
    >>> from src.prototypical_network.model import PrototypicalNetwork
    >>> from src.backbone.factory import get_backbone
    >>> backbone = get_backbone("resnet18")
    >>> model = PrototypicalNetwork(backbone)
    >>> logits = model(support_images, support_labels, query_images)
"""

from __future__ import annotations

from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.prototypical_network.distance import euclidean_distance, get_distance_fn


class PrototypicalNetwork(nn.Module):
    """Prototypical Network for few-shot classification.

    Given a support set of labelled examples and a query set of unlabelled
    examples, the network:

    1. Embeds all images through a shared backbone encoder.
    2. Computes a **prototype** (centroid) for each class in the support set.
    3. Assigns each query to the nearest prototype using a distance metric.

    Attributes:
        backbone (nn.Module): CNN feature extractor producing embeddings.
        distance_fn: Callable computing pairwise distances.
        embedding_dim (int): Dimensionality of the embedding space.

    Args:
        backbone: A feature extraction backbone whose ``forward(x)``
            returns tensors of shape ``(B, embedding_dim)``.
        distance_type: Distance metric name (``'euclidean'``,
            ``'cosine'``, ``'mahalanobis'``).  Defaults to
            ``'euclidean'`` as recommended by the original paper.

    Example:
        >>> import torch
        >>> from src.backbone.resnet18 import ResNet18Backbone
        >>> backbone = ResNet18Backbone(pretrained=False)
        >>> model = PrototypicalNetwork(backbone, distance_type="euclidean")
        >>> support = torch.randn(20, 3, 224, 224)  # 4-way 5-shot
        >>> labels = torch.tensor([0]*5 + [1]*5 + [2]*5 + [3]*5)
        >>> query = torch.randn(8, 3, 224, 224)
        >>> log_probs = model(support, labels, query)
        >>> print(log_probs.shape)
        torch.Size([8, 4])
    """

    def __init__(
        self,
        backbone: nn.Module,
        distance_type: str = "euclidean",
    ) -> None:
        super().__init__()
        self.backbone = backbone
        self.distance_fn = get_distance_fn(distance_type)
        self.embedding_dim: int = getattr(backbone, "embedding_dim", 512)

    # ------------------------------------------------------------------
    # Prototype computation
    # ------------------------------------------------------------------

    @staticmethod
    def compute_prototypes(
        embeddings: torch.Tensor,
        labels: torch.Tensor,
    ) -> torch.Tensor:
        """Compute class prototypes by averaging embeddings per class.

        Args:
            embeddings: Support set embeddings of shape ``(N, D)``.
            labels: Corresponding integer labels of shape ``(N,)`` with
                values in ``[0, n_way)``.

        Returns:
            Prototype tensor of shape ``(n_way, D)`` where row *c*
            is the centroid of all embeddings with ``labels == c``.
        """
        unique_labels = torch.unique(labels)
        n_way = unique_labels.size(0)
        embedding_dim = embeddings.size(1)

        prototypes = torch.zeros(
            n_way, embedding_dim,
            device=embeddings.device,
            dtype=embeddings.dtype,
        )

        for idx, cls_label in enumerate(unique_labels):
            mask = labels == cls_label
            prototypes[idx] = embeddings[mask].mean(dim=0)

        return prototypes

    # ------------------------------------------------------------------
    # Forward pass
    # ------------------------------------------------------------------

    def forward(
        self,
        support_images: torch.Tensor,
        support_labels: torch.Tensor,
        query_images: torch.Tensor,
    ) -> torch.Tensor:
        """Run the full prototypical network forward pass.

        Steps:
            1. Embed support and query images through the backbone.
            2. Compute prototypes from support embeddings.
            3. Compute distances from each query to each prototype.
            4. Return log-probabilities (log-softmax of negative distances).

        Args:
            support_images: Support set tensor ``(n_way * k_shot, C, H, W)``.
            support_labels: Support labels ``(n_way * k_shot,)`` with
                episode-local class indices in ``[0, n_way)``.
            query_images: Query set tensor ``(n_query, C, H, W)``.

        Returns:
            Log-probability tensor of shape ``(n_query, n_way)`` where
            entry ``[i, c]`` is the log-probability that query *i*
            belongs to class *c*.
        """
        # 1. Embed support and query through the shared backbone
        support_embeddings = self.backbone(support_images)  # (N_s, D)
        query_embeddings = self.backbone(query_images)       # (N_q, D)

        # 2. Compute class prototypes
        prototypes = self.compute_prototypes(
            support_embeddings, support_labels
        )  # (n_way, D)

        # 3. Compute distances: (N_q, n_way)
        distances = self.distance_fn(query_embeddings, prototypes)

        # 4. Convert distances to log-probabilities
        # Negate distances so closer = higher probability
        log_probs = F.log_softmax(-distances, dim=1)  # (N_q, n_way)

        return log_probs

    # ------------------------------------------------------------------
    # Inference helpers
    # ------------------------------------------------------------------

    def compute_prototypes_from_images(
        self,
        support_images: torch.Tensor,
        support_labels: torch.Tensor,
    ) -> torch.Tensor:
        """Embed support images and return prototypes.

        This is useful during inference to pre-compute prototypes once
        and reuse them for multiple queries.

        Args:
            support_images: ``(N, C, H, W)`` support images.
            support_labels: ``(N,)`` class labels.

        Returns:
            Prototype tensor ``(n_way, D)``.
        """
        with torch.no_grad():
            embeddings = self.backbone(support_images)
        return self.compute_prototypes(embeddings, support_labels)

    def classify(
        self,
        query_images: torch.Tensor,
        prototypes: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Classify query images against pre-computed prototypes.

        Args:
            query_images: ``(N_q, C, H, W)`` query images.
            prototypes: ``(n_way, D)`` pre-computed class prototypes.

        Returns:
            A tuple ``(predictions, probabilities)`` where:

            * ``predictions`` has shape ``(N_q,)`` – predicted class indices.
            * ``probabilities`` has shape ``(N_q, n_way)`` – class
              probabilities (softmax of negative distances).
        """
        with torch.no_grad():
            query_embeddings = self.backbone(query_images)
            distances = self.distance_fn(query_embeddings, prototypes)
            probabilities = F.softmax(-distances, dim=1)
            predictions = torch.argmax(probabilities, dim=1)

        return predictions, probabilities

    # ------------------------------------------------------------------
    # Utility
    # ------------------------------------------------------------------

    def get_embedding(self, images: torch.Tensor) -> torch.Tensor:
        """Extract raw embeddings for a batch of images.

        Args:
            images: ``(B, C, H, W)`` image tensor.

        Returns:
            Embedding tensor ``(B, D)``.
        """
        return self.backbone(images)

    def __repr__(self) -> str:
        return (
            f"PrototypicalNetwork(\n"
            f"  backbone={self.backbone.__class__.__name__},\n"
            f"  embedding_dim={self.embedding_dim},\n"
            f"  distance={self.distance_fn.__name__}\n"
            f")"
        )
