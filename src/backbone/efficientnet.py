"""
EfficientNet-B0 Feature Extractor Backbone for Prototypical Networks.

This module provides an EfficientNet-B0 based feature extractor that maps input
images of shape (B, 3, 224, 224) to compact embedding vectors of shape
(B, 512). EfficientNet-B0 uses compound scaling (depth, width, resolution)
and mobile inverted bottleneck blocks (MBConv) to achieve high accuracy with
significantly fewer parameters than ResNet counterparts.

Since EfficientNet-B0 natively produces 1280-dimensional features, a
learnable linear projection head maps the output to the standard
512-dimensional embedding space used by the prototypical network.

Architecture Overview:
    EfficientNet-B0 features → AdaptiveAvgPool2d → Flatten →
    Linear(1280→512) → ReLU → Dropout(0.2) → (B, 512) embedding

Key Advantages for Industrial Inspection:
    - Compound scaling captures multi-scale defect features efficiently
    - Squeeze-and-excitation blocks provide channel attention for
      texture-sensitive features (scratches, stains, anomalies)
    - Fewer parameters than ResNets → faster inference on edge devices
    - Learnable projection head allows embedding space adaptation

Typical Usage:
    >>> backbone = EfficientNetBackbone(pretrained=True, freeze_layers=True)
    >>> embeddings = backbone(images)  # images: (B, 3, 224, 224) → (B, 512)
"""

import torch
import torch.nn as nn
from torchvision import models


class EfficientNetBackbone(nn.Module):
    """EfficientNet-B0 feature extractor for prototypical networks.

    Extracts embedding vectors from input images using a pretrained
    EfficientNet-B0 backbone. The native 1280-dimensional output of
    EfficientNet-B0's feature maps is projected down to a configurable
    embedding dimension (default 512) via a learnable linear layer with
    ReLU activation and dropout regularization.

    The projection head serves two purposes:
        1. Dimensionality alignment with other backbones (ResNet-18/34)
           for consistent prototypical network operation.
        2. Learnable embedding space adaptation that can be fine-tuned
           for the target few-shot task.

    Attributes:
        features (nn.Sequential): EfficientNet-B0 convolutional feature
            extraction blocks (MBConv layers with squeeze-excitation).
        avgpool (nn.AdaptiveAvgPool2d): Global average pooling to reduce
            spatial dimensions to (1, 1).
        projection (nn.Sequential): Learnable projection head mapping
            1280-dim features to the target embedding dimension.
        embedding_dim (int): Dimensionality of the output embedding.

    Args:
        pretrained (bool): If True, load ImageNet pretrained weights.
            Defaults to True.
        embedding_dim (int): Output embedding dimensionality. Unlike
            ResNet backbones, this is configurable via the projection
            head. Defaults to 512.
        freeze_layers (bool): If True, freeze the first 6 (out of 9)
            feature blocks. EfficientNet-B0 has 9 feature blocks (0-8);
            freezing blocks 0-5 retains generic low-level features while
            allowing blocks 6-8 and the projection head to adapt.
            Defaults to False.

    Example:
        >>> import torch
        >>> backbone = EfficientNetBackbone(pretrained=True, embedding_dim=512)
        >>> images = torch.randn(8, 3, 224, 224)
        >>> embeddings = backbone(images)
        >>> print(embeddings.shape)
        torch.Size([8, 512])
    """

    # Native output dimensionality of EfficientNet-B0 feature maps.
    _EFFICIENTNET_B0_FEATURES_DIM: int = 1280

    def __init__(
        self,
        pretrained: bool = True,
        embedding_dim: int = 512,
        freeze_layers: bool = False,
    ) -> None:
        super().__init__()

        # Load EfficientNet-B0 with or without pretrained weights.
        weights = (
            models.EfficientNet_B0_Weights.DEFAULT if pretrained else None
        )
        efficientnet = models.efficientnet_b0(weights=weights)

        # Extract the convolutional feature backbone.
        # EfficientNet's `.features` attribute contains all MBConv blocks
        # organized as nn.Sequential with 9 sub-modules (indices 0-8).
        self.features = efficientnet.features

        # Global average pooling to collapse spatial dims: (B, 1280, H, W) → (B, 1280, 1, 1)
        self.avgpool = nn.AdaptiveAvgPool2d(1)

        # Projection head: 1280 → embedding_dim
        # - Linear layer performs the dimensionality reduction
        # - ReLU adds non-linearity to the projection
        # - Dropout (p=0.2) provides regularization, matching
        #   EfficientNet-B0's default dropout rate
        self.projection = nn.Sequential(
            nn.Linear(self._EFFICIENTNET_B0_FEATURES_DIM, embedding_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),
        )

        self.embedding_dim = embedding_dim

        if freeze_layers:
            # Freeze the first 6 feature blocks (indices 0-5).
            # EfficientNet-B0 has 9 blocks organized by increasing
            # receptive field and abstraction level:
            #   Blocks 0-2: Low-level features (edges, textures)
            #   Blocks 3-5: Mid-level features (patterns, shapes)
            #   Blocks 6-8: High-level features (object parts, semantics)
            # Freezing blocks 0-5 retains pretrained generic features
            # while allowing deeper blocks to adapt to industrial defects.
            for param in self.features[:6].parameters():
                param.requires_grad = False

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Extract feature embeddings from input images.

        Passes the input through the EfficientNet-B0 convolutional blocks,
        applies global average pooling, flattens, and projects to the
        target embedding dimension.

        Args:
            x (torch.Tensor): Input image batch of shape (B, 3, 224, 224).
                Expected to be normalized with ImageNet statistics:
                mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225].

        Returns:
            torch.Tensor: Feature embeddings of shape (B, embedding_dim).

        Raises:
            RuntimeError: If input tensor has incorrect shape or device
                incompatibility.
        """
        # Forward through convolutional feature blocks.
        # Output shape: (B, 1280, 7, 7) for 224×224 input.
        x = self.features(x)

        # Global average pooling: (B, 1280, 7, 7) → (B, 1280, 1, 1)
        x = self.avgpool(x)

        # Flatten: (B, 1280, 1, 1) → (B, 1280)
        x = x.view(x.size(0), -1)

        # Project to embedding space: (B, 1280) → (B, embedding_dim)
        x = self.projection(x)

        return x

    def get_trainable_params(self) -> int:
        """Count the number of trainable parameters.

        Useful for logging and verifying that layer freezing is working
        as expected. EfficientNet-B0 has ~5.3M parameters in total
        (significantly fewer than ResNet-18's ~11.2M).

        Returns:
            int: Number of parameters with requires_grad=True.
        """
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def get_total_params(self) -> int:
        """Count the total number of parameters (trainable + frozen).

        Returns:
            int: Total number of parameters in the backbone.
        """
        return sum(p.numel() for p in self.parameters())

    def __repr__(self) -> str:
        """Provide a concise summary of the backbone configuration."""
        trainable = self.get_trainable_params()
        total = self.get_total_params()
        return (
            f"EfficientNetBackbone("
            f"embedding_dim={self.embedding_dim}, "
            f"projection=1280→{self.embedding_dim}, "
            f"trainable_params={trainable:,}, "
            f"total_params={total:,}, "
            f"frozen={total - trainable:,})"
        )
