"""
PyTorch data loading utilities for episodic few-shot learning.

This module provides:

* :class:`DefectDataset` – A ``torch.utils.data.Dataset`` that recursively
  scans a directory tree for images and exposes per-class index lookups
  needed by the episodic sampler.
* :class:`EpisodicSampler` – A ``torch.utils.data.Sampler`` that yields
  batches of indices forming valid N-way K-shot episodes.
* :func:`collate_episodes` – A custom collate function that splits a raw
  batch into **support** and **query** tensors.
* :func:`get_data_loaders` – Factory that wires everything together and
  returns ready-to-use train / test ``DataLoader`` instances.
* :func:`load_support_set` – Loads a directory of labelled images into
  a ``(images, labels)`` tensor pair for inference.

Directory layout expected::

    root_dir/
    ├── normal/
    │   ├── img_0001.jpg
    │   └── ...
    ├── scratch/
    ├── crack/
    └── dent/

Usage:
    from src.preprocessing.image_loader import get_data_loaders
    from src.utils.config import get_config
    train_loader, test_loader = get_data_loaders(get_config())
"""

import os
import random
from functools import partial
from pathlib import Path
from typing import Any, Callable, Dict, Iterator, List, Optional, Tuple

import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset, Sampler

from src.preprocessing.augmentations import (
    get_support_transforms,
    get_train_transforms,
    get_val_transforms,
)

# Supported image file extensions (case-insensitive check)
_IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".tif", ".webp"}


# ===================================================================
#  Dataset
# ===================================================================

class DefectDataset(Dataset):
    """Dataset for loading industrial defect images from a folder structure.

    Each immediate sub-directory of *root_dir* is treated as a class.
    The sub-directories are sorted alphabetically; the sort index is used
    as the integer label.

    Attributes:
        root_dir: Absolute path to the dataset split root.
        classes: Sorted list of class names.
        class_to_idx: Mapping ``class_name → int label``.
        samples: List of ``(image_path, label)`` tuples.
        class_to_indices: Mapping ``label → [dataset indices]`` for
            efficient episodic sampling.
        transform: Optional torchvision transform pipeline.
    """

    def __init__(
        self,
        root_dir: str,
        transform: Optional[Callable] = None,
        classes: Optional[List[str]] = None,
    ) -> None:
        """Initialise the dataset by scanning *root_dir* for images.

        Args:
            root_dir: Path to the directory containing one sub-folder per
                class (e.g. ``data/train/``).
            transform: Optional ``torchvision.transforms`` pipeline applied
                to each loaded ``PIL.Image``.
            classes: If provided, only these class sub-folders are loaded
                (and in this order).  Otherwise all sub-directories are
                discovered and sorted alphabetically.
        """
        self.root_dir = Path(root_dir)
        self.transform = transform

        # ----- Discover or verify classes ----------------------------
        if classes is not None:
            self.classes = list(classes)
        else:
            self.classes = sorted(
                d.name
                for d in self.root_dir.iterdir()
                if d.is_dir() and not d.name.startswith(".")
            )

        self.class_to_idx: Dict[str, int] = {
            cls_name: idx for idx, cls_name in enumerate(self.classes)
        }

        # ----- Build sample list and class-to-indices mapping --------
        self.samples: List[Tuple[str, int]] = []
        self.class_to_indices: Dict[int, List[int]] = {
            idx: [] for idx in range(len(self.classes))
        }

        for cls_name in self.classes:
            cls_dir = self.root_dir / cls_name
            if not cls_dir.is_dir():
                continue  # skip missing classes gracefully
            label = self.class_to_idx[cls_name]
            for fpath in sorted(cls_dir.iterdir()):
                if fpath.suffix.lower() in _IMAGE_EXTENSIONS:
                    sample_idx = len(self.samples)
                    self.samples.append((str(fpath), label))
                    self.class_to_indices[label].append(sample_idx)

        if len(self.samples) == 0:
            raise FileNotFoundError(
                f"No images found in {self.root_dir}. "
                f"Expected sub-folders: {self.classes}"
            )

    # ----- Core Dataset API ------------------------------------------

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        """Load a single image and return ``(tensor, label)``.

        Args:
            idx: Dataset index.

        Returns:
            A tuple ``(image_tensor, integer_label)``.
        """
        img_path, label = self.samples[idx]
        image = Image.open(img_path).convert("RGB")
        if self.transform is not None:
            image = self.transform(image)
        return image, label

    def __len__(self) -> int:
        """Return the total number of images in the dataset."""
        return len(self.samples)

    # ----- Convenience -----------------------------------------------

    def summary(self) -> str:
        """Return a human-readable summary of the dataset contents."""
        lines = [
            f"DefectDataset @ {self.root_dir}",
            f"  Total samples : {len(self.samples)}",
            f"  Classes ({len(self.classes)}):  {', '.join(self.classes)}",
        ]
        for cls_name in self.classes:
            label = self.class_to_idx[cls_name]
            count = len(self.class_to_indices[label])
            lines.append(f"    {cls_name:>10s}: {count} images")
        return "\n".join(lines)


# ===================================================================
#  Episodic Sampler
# ===================================================================

class EpisodicSampler(Sampler):
    """Sampler that yields index batches forming N-way K-shot episodes.

    Each episode consists of:
    * ``n_way`` randomly chosen classes.
    * For each chosen class, ``k_shot + q_query`` randomly sampled
      image indices.

    The yielded batch therefore has ``n_way * (k_shot + q_query)``
    indices, arranged **class-contiguously**: all images for class 0
    first (support then query), then class 1, etc.  The
    :func:`collate_episodes` function then splits this into support
    and query tensors.

    Attributes:
        dataset: The underlying :class:`DefectDataset`.
        n_way: Number of classes per episode.
        k_shot: Number of support images per class.
        q_query: Number of query images per class.
        episodes: Total number of episodes to generate per epoch.
    """

    def __init__(
        self,
        dataset: DefectDataset,
        n_way: int,
        k_shot: int,
        q_query: int,
        episodes: int,
    ) -> None:
        """
        Args:
            dataset: A :class:`DefectDataset` providing ``class_to_indices``.
            n_way: Number of classes sampled per episode.
            k_shot: Number of support examples per class.
            q_query: Number of query examples per class.
            episodes: Number of episodes in one epoch.

        Raises:
            ValueError: If *n_way* exceeds the number of available classes
                or if any class has fewer than *k_shot + q_query* images.
        """
        super().__init__(dataset)
        self.dataset = dataset
        self.n_way = n_way
        self.k_shot = k_shot
        self.q_query = q_query
        self.episodes = episodes

        # Validation
        available_classes = [
            cls_label
            for cls_label, indices in dataset.class_to_indices.items()
            if len(indices) >= (k_shot + q_query)
        ]
        if len(available_classes) < n_way:
            raise ValueError(
                f"Need at least {n_way} classes with ≥ {k_shot + q_query} "
                f"images each, but only {len(available_classes)} classes "
                f"qualify: {available_classes}."
            )
        self._eligible_classes = available_classes

    def __iter__(self) -> Iterator[List[int]]:
        """Yield one batch of indices per episode.

        Yields:
            A list of ``n_way * (k_shot + q_query)`` dataset indices.
            The indices are ordered *class-contiguously*: for each of the
            ``n_way`` classes, the first ``k_shot`` indices are support
            samples and the next ``q_query`` indices are query samples.
        """
        for _ in range(self.episodes):
            # 1. Sample n_way classes (without replacement)
            episode_classes = random.sample(self._eligible_classes, self.n_way)

            batch_indices: List[int] = []
            for cls_label in episode_classes:
                # 2. Sample k_shot + q_query indices for this class
                cls_indices = self.dataset.class_to_indices[cls_label]
                selected = random.sample(cls_indices, self.k_shot + self.q_query)
                batch_indices.extend(selected)

            yield batch_indices

    def __len__(self) -> int:
        """Return the number of episodes per epoch."""
        return self.episodes


# ===================================================================
#  Collate function
# ===================================================================

def collate_episodes(
    batch: List[Tuple[torch.Tensor, int]],
    n_way: int,
    k_shot: int,
    q_query: int,
) -> Dict[str, torch.Tensor]:
    """Custom collate that splits a raw batch into support / query sets.

    Expects the batch to be ordered class-contiguously as produced by
    :class:`EpisodicSampler`.

    Args:
        batch: List of ``(image_tensor, label)`` tuples with length
            ``n_way * (k_shot + q_query)``.
        n_way: Number of classes in the episode.
        k_shot: Number of support images per class.
        q_query: Number of query images per class.

    Returns:
        A dictionary with keys:

        * ``"support_images"`` – ``(n_way * k_shot, C, H, W)``
        * ``"support_labels"`` – ``(n_way * k_shot,)``  *episode-local*
          labels in ``[0, n_way)``.
        * ``"query_images"``   – ``(n_way * q_query, C, H, W)``
        * ``"query_labels"``   – ``(n_way * q_query,)`` *episode-local*
          labels in ``[0, n_way)``.
    """
    images = torch.stack([item[0] for item in batch])   # (N, C, H, W)
    # Original dataset labels (we remap to episode-local labels below)

    samples_per_class = k_shot + q_query

    support_images: List[torch.Tensor] = []
    support_labels: List[int] = []
    query_images: List[torch.Tensor] = []
    query_labels: List[int] = []

    for cls_idx in range(n_way):
        start = cls_idx * samples_per_class
        # Support: first k_shot
        support_images.append(images[start: start + k_shot])
        support_labels.extend([cls_idx] * k_shot)
        # Query: next q_query
        query_images.append(images[start + k_shot: start + samples_per_class])
        query_labels.extend([cls_idx] * q_query)

    return {
        "support_images": torch.cat(support_images, dim=0),   # (n_way*k_shot, C, H, W)
        "support_labels": torch.tensor(support_labels, dtype=torch.long),
        "query_images": torch.cat(query_images, dim=0),       # (n_way*q_query, C, H, W)
        "query_labels": torch.tensor(query_labels, dtype=torch.long),
    }


# ===================================================================
#  Data-loader factory
# ===================================================================

def get_data_loaders(
    config: Any,
) -> Tuple[DataLoader, DataLoader]:
    """Create episodic train and test ``DataLoader`` instances.

    Uses the project :class:`~src.utils.config.Config` object to derive
    paths, class lists, and episodic hyper-parameters.

    Args:
        config: A :class:`~src.utils.config.Config` (or compatible) object
            exposing ``config.data``, ``config.training``, and
            ``config.project`` attributes.

    Returns:
        A ``(train_loader, test_loader)`` tuple of ``DataLoader``
        instances whose iteration yields dictionaries produced by
        :func:`collate_episodes`.
    """
    # ----- Unpack config ---------------------------------------------
    image_size = config.data.image_size
    classes = config.data.classes
    n_way = config.training.n_way
    k_shot = config.training.k_shot
    q_query = config.training.q_query
    episodes_per_epoch = config.training.episodes_per_epoch
    train_dir = str(config.project.train_dir)
    test_dir = str(config.project.test_dir)

    # ----- Datasets ---------------------------------------------------
    train_dataset = DefectDataset(
        root_dir=train_dir,
        transform=get_train_transforms(image_size),
        classes=classes,
    )
    test_dataset = DefectDataset(
        root_dir=test_dir,
        transform=get_val_transforms(image_size),
        classes=classes,
    )

    # ----- Episodic samplers ------------------------------------------
    train_sampler = EpisodicSampler(
        dataset=train_dataset,
        n_way=n_way,
        k_shot=k_shot,
        q_query=q_query,
        episodes=episodes_per_epoch,
    )
    # Fewer episodes for test evaluation
    test_episodes = max(episodes_per_epoch // 5, 10)
    test_sampler = EpisodicSampler(
        dataset=test_dataset,
        n_way=n_way,
        k_shot=k_shot,
        q_query=q_query,
        episodes=test_episodes,
    )

    # ----- Collate with frozen episodic params -----------------------
    train_collate = partial(collate_episodes, n_way=n_way, k_shot=k_shot, q_query=q_query)
    test_collate = partial(collate_episodes, n_way=n_way, k_shot=k_shot, q_query=q_query)

    # ----- Data loaders -----------------------------------------------
    train_loader = DataLoader(
        dataset=train_dataset,
        batch_sampler=train_sampler,
        collate_fn=train_collate,
        num_workers=4,
        pin_memory=True,
    )
    test_loader = DataLoader(
        dataset=test_dataset,
        batch_sampler=test_sampler,
        collate_fn=test_collate,
        num_workers=2,
        pin_memory=True,
    )

    print(f"[DataLoader] Train: {train_dataset.summary()}")
    print(f"[DataLoader] Test : {test_dataset.summary()}")

    return train_loader, test_loader


# ===================================================================
#  Inference helper – load a fixed support set
# ===================================================================

def load_support_set(
    support_dir: str,
    transform: Optional[Callable] = None,
    classes: Optional[List[str]] = None,
    image_size: int = 224,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Load all images from *support_dir* into a support-set tensor pair.

    This is used during inference: the support set is encoded once by the
    backbone, and the resulting prototypes are reused for every query.

    Args:
        support_dir: Path to the support directory (sub-folders per class).
        transform: Transform pipeline.  If ``None``, the default
            :func:`get_support_transforms` pipeline is used.
        classes: Ordered list of class names.  If ``None``, sub-folders
            are discovered alphabetically.
        image_size: Image size passed to default transforms if *transform*
            is not provided.

    Returns:
        A ``(images, labels)`` tuple where:

        * ``images`` has shape ``(N, C, H, W)``
        * ``labels`` has shape ``(N,)`` with integer class indices.
    """
    if transform is None:
        transform = get_support_transforms(image_size)

    dataset = DefectDataset(
        root_dir=support_dir,
        transform=transform,
        classes=classes,
    )

    images: List[torch.Tensor] = []
    labels: List[int] = []
    for idx in range(len(dataset)):
        img, lbl = dataset[idx]
        images.append(img)
        labels.append(lbl)

    return torch.stack(images), torch.tensor(labels, dtype=torch.long)
