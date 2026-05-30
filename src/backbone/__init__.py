"""
Backbone Sub-package
====================

Provides feature-extraction backbones (ResNet-18, ResNet-50, etc.)
for embedding industrial defect images into a metric space.
"""

from src.backbone.factory import get_backbone

__all__ = [
    "get_backbone",
]
