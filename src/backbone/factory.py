"""
Backbone Factory for Prototypical Networks.

Provides a single entry-point :func:`get_backbone` that instantiates
a feature-extraction backbone by name.  This decouples the rest of the
codebase from specific backbone implementations and makes backbone
swapping a one-line configuration change.

Registered backbones
--------------------
* ``resnet18``     – :class:`~src.backbone.resnet18.ResNet18Backbone`
* ``resnet34``     – :class:`~src.backbone.resnet34.ResNet34Backbone`
* ``efficientnet`` – :class:`~src.backbone.efficientnet.EfficientNetBackbone`

Usage:
    >>> from src.backbone.factory import get_backbone
    >>> backbone = get_backbone("resnet18", pretrained=True)
    >>> print(backbone.embedding_dim)
    512
"""

from __future__ import annotations

from typing import Dict, Type

import torch.nn as nn

from src.backbone.resnet18 import ResNet18Backbone
from src.backbone.resnet34 import ResNet34Backbone
from src.backbone.efficientnet import EfficientNetBackbone


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------

_BACKBONE_REGISTRY: Dict[str, Type[nn.Module]] = {
    "resnet18": ResNet18Backbone,
    "resnet34": ResNet34Backbone,
    "efficientnet": EfficientNetBackbone,
    "efficientnet_b0": EfficientNetBackbone,  # alias
}


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def get_backbone(
    name: str,
    pretrained: bool = True,
    embedding_dim: int = 512,
    freeze_layers: bool = False,
) -> nn.Module:
    """Instantiate a feature-extraction backbone by name.

    Args:
        name: Backbone identifier (case-insensitive).  One of
            ``'resnet18'``, ``'resnet34'``, ``'efficientnet'``.
        pretrained: Load ImageNet pretrained weights.
        embedding_dim: Target embedding dimensionality.
        freeze_layers: Freeze early layers for transfer learning.

    Returns:
        An ``nn.Module`` whose ``forward(x)`` returns a tensor of
        shape ``(batch_size, embedding_dim)``.

    Raises:
        ValueError: If *name* is not found in the registry.
    """
    key = name.lower().strip()
    if key not in _BACKBONE_REGISTRY:
        available = ", ".join(sorted(_BACKBONE_REGISTRY.keys()))
        raise ValueError(
            f"Unknown backbone '{name}'. Available backbones: {available}"
        )

    backbone_cls = _BACKBONE_REGISTRY[key]
    backbone = backbone_cls(
        pretrained=pretrained,
        embedding_dim=embedding_dim,
        freeze_layers=freeze_layers,
    )
    return backbone


def list_backbones() -> list[str]:
    """Return a sorted list of registered backbone names."""
    return sorted(_BACKBONE_REGISTRY.keys())
