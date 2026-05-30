"""
Distance functions for Prototypical Networks.

Provides GPU-compatible, batched distance computations between query
embeddings and class prototypes.  The default metric for Prototypical
Networks is **Euclidean distance** (Snell et al., NeurIPS 2017), but
cosine and Mahalanobis alternatives are included for experimentation.

All functions follow the signature::

    distance(x, y) → D

Where:
    * ``x`` has shape ``(N, D)`` – *N* query embeddings of dimension *D*.
    * ``y`` has shape ``(M, D)`` – *M* prototypes of dimension *D*.
    * ``D`` (output) has shape ``(N, M)`` – pairwise distances.

Usage:
    >>> from src.prototypical_network.distance import euclidean_distance
    >>> dists = euclidean_distance(query_embeddings, prototypes)
"""

from __future__ import annotations

import torch
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Euclidean Distance
# ---------------------------------------------------------------------------

def euclidean_distance(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """Compute batched squared Euclidean distance between two sets of vectors.

    Uses the expansion  ||x - y||² = ||x||² + ||y||² - 2·x·yᵀ  which is
    more numerically stable and efficient for GPU computation than naïve
    pairwise subtraction for large batches.

    Args:
        x: Query embeddings of shape ``(N, D)``.
        y: Prototype embeddings of shape ``(M, D)``.

    Returns:
        Distance matrix of shape ``(N, M)`` where entry ``[i, j]`` is
        the squared Euclidean distance between ``x[i]`` and ``y[j]``.
    """
    # ||x||²  →  (N, 1)
    x_sq = torch.sum(x ** 2, dim=1, keepdim=True)
    # ||y||²  →  (1, M)
    y_sq = torch.sum(y ** 2, dim=1, keepdim=True).t()
    # x · yᵀ  →  (N, M)
    xy = torch.mm(x, y.t())

    # ||x - y||² = ||x||² + ||y||² - 2·x·yᵀ
    dists = x_sq + y_sq - 2.0 * xy
    # Clamp small negatives caused by floating-point imprecision
    return torch.clamp(dists, min=0.0)


# ---------------------------------------------------------------------------
# Cosine Distance
# ---------------------------------------------------------------------------

def cosine_distance(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """Compute batched cosine distance between two sets of vectors.

    Cosine distance is defined as  ``1 - cosine_similarity(x, y)``.
    Values range from 0 (identical direction) to 2 (opposite direction).

    Args:
        x: Query embeddings of shape ``(N, D)``.
        y: Prototype embeddings of shape ``(M, D)``.

    Returns:
        Distance matrix of shape ``(N, M)``.
    """
    # L2-normalise along feature dimension
    x_norm = F.normalize(x, p=2, dim=1)  # (N, D)
    y_norm = F.normalize(y, p=2, dim=1)  # (M, D)

    # Cosine similarity → (N, M)
    similarity = torch.mm(x_norm, y_norm.t())

    # Convert to distance
    return 1.0 - similarity


# ---------------------------------------------------------------------------
# Mahalanobis Distance (simplified diagonal covariance)
# ---------------------------------------------------------------------------

def mahalanobis_distance(
    x: torch.Tensor,
    y: torch.Tensor,
    cov_diag: torch.Tensor | None = None,
) -> torch.Tensor:
    """Compute batched Mahalanobis distance with diagonal covariance.

    When *cov_diag* is ``None``, falls back to standard Euclidean
    distance.  With a per-dimension variance vector, each feature
    dimension is weighted inversely by its variance, emphasising
    discriminative features.

    Args:
        x: Query embeddings of shape ``(N, D)``.
        y: Prototype embeddings of shape ``(M, D)``.
        cov_diag: Optional diagonal covariance vector of shape ``(D,)``.
            If ``None``, standard Euclidean distance is returned.

    Returns:
        Distance matrix of shape ``(N, M)``.
    """
    if cov_diag is None:
        return euclidean_distance(x, y)

    # Inverse diagonal covariance (precision)
    # Add small epsilon for numerical stability
    precision = 1.0 / (cov_diag + 1e-8)  # (D,)

    # Weight each dimension: x_w = x * sqrt(precision)
    sqrt_prec = torch.sqrt(precision).unsqueeze(0)  # (1, D)
    x_weighted = x * sqrt_prec  # (N, D)
    y_weighted = y * sqrt_prec  # (M, D)

    return euclidean_distance(x_weighted, y_weighted)


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------

_DISTANCE_REGISTRY = {
    "euclidean": euclidean_distance,
    "cosine": cosine_distance,
    "mahalanobis": mahalanobis_distance,
}


def get_distance_fn(name: str = "euclidean"):
    """Return a distance function by name.

    Args:
        name: One of ``'euclidean'``, ``'cosine'``, ``'mahalanobis'``.

    Returns:
        A callable ``(x, y) → distances``.

    Raises:
        ValueError: If *name* is not registered.
    """
    key = name.lower().strip()
    if key not in _DISTANCE_REGISTRY:
        available = ", ".join(sorted(_DISTANCE_REGISTRY.keys()))
        raise ValueError(
            f"Unknown distance function '{name}'. Available: {available}"
        )
    return _DISTANCE_REGISTRY[key]
