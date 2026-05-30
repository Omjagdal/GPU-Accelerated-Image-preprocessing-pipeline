"""
Data augmentation pipelines for industrial defect detection.

Provides three augmentation tiers tailored for few-shot learning on
industrial surface images:

1. **Training transforms** – aggressive augmentations (flips, rotations,
   colour jitter, perspective warp, Gaussian blur, random erasing) to
   maximise diversity during episodic training.
2. **Validation / test transforms** – deterministic resize-and-normalise
   only, ensuring reproducible evaluation.
3. **Support-set transforms** – *light* augmentations that preserve the
   representative features of each class while adding slight variability
   so the prototype embeddings generalise better at inference time.

All pipelines finish with ImageNet-pretrained normalisation constants
(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]) to match
backbone pre-training statistics.

Usage:
    from src.preprocessing.augmentations import get_train_transforms
    transform = get_train_transforms(image_size=224)
    augmented = transform(pil_image)
"""

from typing import Optional

import torchvision.transforms as T


# ---------------------------------------------------------------------------
# ImageNet normalisation constants (used across all pipelines)
# ---------------------------------------------------------------------------
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def get_train_transforms(image_size: int = 224) -> T.Compose:
    """Return an aggressive augmentation pipeline for episodic training.

    The pipeline is designed to create diverse views of industrial surface
    images so the prototypical network learns invariant feature embeddings.

    Augmentation stages
    -------------------
    1. **Resize** to a fixed square to normalise input dimensions.
    2. **Geometric** – horizontal & vertical flips, rotation (±15°),
       affine translation/scale, and perspective warp.
    3. **Photometric** – colour jitter and Gaussian blur to simulate
       varying lighting and camera focus conditions.
    4. **Tensor conversion & normalisation** – standard ImageNet stats.
    5. **Random erasing** – applied *after* ``ToTensor`` because it
       operates on tensors directly; simulates partial occlusion.

    Args:
        image_size: Target spatial dimension for the square crop (H = W).

    Returns:
        A ``torchvision.transforms.Compose`` pipeline ready to accept a
        ``PIL.Image`` and return a normalised ``torch.Tensor``.
    """
    return T.Compose([
        # ----- Spatial normalisation -----
        T.Resize((image_size, image_size)),

        # ----- Geometric augmentations -----
        T.RandomHorizontalFlip(p=0.5),
        T.RandomVerticalFlip(p=0.3),
        T.RandomRotation(degrees=15),
        T.RandomAffine(
            degrees=0,
            translate=(0.1, 0.1),
            scale=(0.9, 1.1),
        ),
        T.RandomPerspective(distortion_scale=0.2, p=0.3),

        # ----- Photometric augmentations -----
        T.ColorJitter(
            brightness=0.3,
            contrast=0.3,
            saturation=0.2,
            hue=0.1,
        ),
        T.GaussianBlur(kernel_size=3, sigma=(0.1, 2.0)),

        # ----- Tensor conversion & normalisation -----
        T.ToTensor(),
        T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),

        # ----- Tensor-level augmentations -----
        # RandomErasing works on tensors, so it must come after ToTensor.
        T.RandomErasing(p=0.2, scale=(0.02, 0.15)),
    ])


def get_val_transforms(image_size: int = 224) -> T.Compose:
    """Return a deterministic transform pipeline for validation / testing.

    No stochastic augmentations are applied so that evaluation metrics
    are fully reproducible across runs.

    Args:
        image_size: Target spatial dimension for the square resize (H = W).

    Returns:
        A ``torchvision.transforms.Compose`` pipeline ready to accept a
        ``PIL.Image`` and return a normalised ``torch.Tensor``.
    """
    return T.Compose([
        T.Resize((image_size, image_size)),
        T.ToTensor(),
        T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])


def get_support_transforms(image_size: int = 224) -> T.Compose:
    """Return a *light* augmentation pipeline for support-set images.

    During few-shot inference the support images define the class
    prototypes.  We apply only minimal augmentations – small rotation and
    gentle colour jitter – to slightly broaden the prototype without
    distorting the characteristic appearance of each defect class.

    Args:
        image_size: Target spatial dimension for the square resize (H = W).

    Returns:
        A ``torchvision.transforms.Compose`` pipeline ready to accept a
        ``PIL.Image`` and return a normalised ``torch.Tensor``.
    """
    return T.Compose([
        # ----- Spatial normalisation -----
        T.Resize((image_size, image_size)),

        # ----- Minimal geometric augmentations -----
        T.RandomHorizontalFlip(p=0.3),
        T.RandomRotation(degrees=5),

        # ----- Subtle photometric variations -----
        T.ColorJitter(
            brightness=0.1,
            contrast=0.1,
            saturation=0.05,
            hue=0.02,
        ),

        # ----- Tensor conversion & normalisation -----
        T.ToTensor(),
        T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])


# ---------------------------------------------------------------------------
# Inverse normalisation (useful for visualisation & debugging)
# ---------------------------------------------------------------------------

def get_inverse_normalize() -> T.Normalize:
    """Return a ``Normalize`` transform that undoes ImageNet normalisation.

    Useful when you need to convert a normalised tensor back to the
    ``[0, 1]`` range for visualisation (e.g. with ``matplotlib``).

    Returns:
        A ``torchvision.transforms.Normalize`` that inverts the standard
        ImageNet mean/std normalisation.
    """
    inv_mean = [-m / s for m, s in zip(IMAGENET_MEAN, IMAGENET_STD)]
    inv_std = [1.0 / s for s in IMAGENET_STD]
    return T.Normalize(mean=inv_mean, std=inv_std)
