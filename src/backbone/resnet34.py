"""
ResNet-34 Feature Extractor Backbone for Prototypical Networks.

This module provides a ResNet-34 based feature extractor that maps input images
of shape (B, 3, 224, 224) to compact embedding vectors of shape (B, 512).
ResNet-34 is a deeper variant of ResNet-18, using 34 convolutional layers
with basic residual blocks (no bottleneck). It offers richer feature
representations while maintaining the same 512-dimensional output as ResNet-18,
making it a drop-in replacement for tasks that benefit from deeper networks.

Architecture Overview:
    ResNet-34 → Remove FC → AdaptiveAvgPool2d → Flatten → (B, 512) embedding

Key Differences from ResNet-18:
    - 34 layers vs. 18 layers
    - More residual blocks per stage: [3, 4, 6, 3] vs. [2, 2, 2, 2]
    - Same output dimensionality (512) → identical interface
    - Slightly higher computational cost but often better accuracy

Typical Usage:
    >>> backbone = ResNet34Backbone(pretrained=True, freeze_layers=True)
    >>> embeddings = backbone(images)  # images: (B, 3, 224, 224) → (B, 512)
"""

import torch
import torch.nn as nn
from torchvision import models


class ResNet34Backbone(nn.Module):
    """ResNet-34 feature extractor for prototypical networks.

    Extracts 512-dimensional embedding vectors from input images using a
    pretrained ResNet-34 architecture with the final fully-connected
    classification layer removed. Compared to ResNet-18, ResNet-34 uses
    more residual blocks per stage ([3, 4, 6, 3] vs [2, 2, 2, 2]),
    providing deeper feature hierarchies that can capture more complex
    visual patterns – particularly useful for subtle defect textures in
    industrial inspection.

    Attributes:
        features (nn.Sequential): All ResNet-34 layers up to and including
            the adaptive average pooling layer (excluding the final FC).
        embedding_dim (int): Dimensionality of the output embedding (512).

    Args:
        pretrained (bool): If True, load ImageNet pretrained weights.
            Strongly recommended for few-shot learning to leverage
            transferred feature representations. Defaults to True.
        embedding_dim (int): Output embedding dimensionality. For ResNet-34,
            this is inherently 512 and this parameter is primarily for
            interface consistency with other backbones. Defaults to 512.
        freeze_layers (bool): If True, freeze all parameters except those
            in layer3 and layer4 (the last two residual blocks). Early
            layers learn generic low-level features that transfer well
            across domains, while later layers adapt to task-specific
            patterns. Defaults to False.

    Example:
        >>> import torch
        >>> backbone = ResNet34Backbone(pretrained=True, freeze_layers=True)
        >>> images = torch.randn(8, 3, 224, 224)
        >>> embeddings = backbone(images)
        >>> print(embeddings.shape)
        torch.Size([8, 512])
    """

    def __init__(
        self,
        pretrained: bool = True,
        embedding_dim: int = 512,
        freeze_layers: bool = False,
    ) -> None:
        super().__init__()

        # Load the full ResNet-34 model with or without pretrained weights.
        # Using the new `weights` API (torchvision >= 0.13) for explicit
        # weight selection and deprecation-warning avoidance.
        weights = models.ResNet34_Weights.DEFAULT if pretrained else None
        resnet = models.resnet34(weights=weights)

        # Remove the final fully-connected layer by taking all children
        # except the last one. The remaining modules are:
        #   conv1 → bn1 → relu → maxpool → layer1 → layer2 → layer3 →
        #   layer4 → avgpool
        # The avgpool (AdaptiveAvgPool2d(1,1)) produces a (B, 512, 1, 1)
        # tensor which we flatten to (B, 512).
        self.features = nn.Sequential(*list(resnet.children())[:-1])

        # Store embedding dimensionality for interface consistency.
        # ResNet-34 (using BasicBlock) has a final channel count of 512.
        self.embedding_dim = embedding_dim

        if freeze_layers:
            # Freeze all layers except layer3 and layer4.
            # Rationale: In few-shot industrial vision, early convolutional
            # layers learn generic features (edges, textures, colors) that
            # are highly transferable across domains. The deeper layers
            # (layer3 with 6 blocks, layer4 with 3 blocks in ResNet-34)
            # learn more task-specific representations and benefit from
            # fine-tuning on the target domain (e.g., scratch, dent, or
            # stain detection patterns).
            for name, param in self.features.named_parameters():
                if "layer3" not in name and "layer4" not in name:
                    param.requires_grad = False

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Extract feature embeddings from input images.

        Passes the input through all convolutional blocks and the adaptive
        average pooling layer, then flattens the spatial dimensions to
        produce a 1D embedding vector per image.

        Args:
            x (torch.Tensor): Input image batch of shape (B, 3, 224, 224).
                Expected to be normalized with ImageNet statistics:
                mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225].

        Returns:
            torch.Tensor: Feature embeddings of shape (B, 512).

        Raises:
            RuntimeError: If input tensor has incorrect number of channels
                or spatial dimensions that cause shape mismatches.
        """
        # Forward through all feature extraction layers.
        # Output shape after features: (B, 512, 1, 1)
        x = self.features(x)

        # Flatten spatial dimensions: (B, 512, 1, 1) → (B, 512)
        # Using view with x.size(0) preserves the batch dimension and
        # collapses everything else into a single dimension.
        x = x.view(x.size(0), -1)

        return x

    def get_trainable_params(self) -> int:
        """Count the number of trainable parameters.

        Useful for logging and verifying that layer freezing is working
        as expected. ResNet-34 has ~21.3M total parameters.

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
            f"ResNet34Backbone("
            f"embedding_dim={self.embedding_dim}, "
            f"trainable_params={trainable:,}, "
            f"total_params={total:,}, "
            f"frozen={total - trainable:,})"
        )
