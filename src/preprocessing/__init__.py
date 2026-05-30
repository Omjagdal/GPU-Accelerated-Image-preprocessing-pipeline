"""
Preprocessing Sub-package
=========================

Handles image loading, augmentation pipelines, and episodic
few-shot task generation for Prototypical Network training.
"""

from src.preprocessing.augmentations import (
    get_train_transforms,
    get_val_transforms,
    get_support_transforms,
)
from src.preprocessing.image_loader import (
    DefectDataset,
    EpisodicSampler,
    get_data_loaders,
    load_support_set,
)

__all__ = [
    "DefectDataset",
    "EpisodicSampler",
    "get_train_transforms",
    "get_val_transforms",
    "get_support_transforms",
    "get_data_loaders",
    "load_support_set",
]
