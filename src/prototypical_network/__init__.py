"""
Prototypical Network Sub-package
================================

Core few-shot classification logic: computes class prototypes from
support sets and classifies query images via Euclidean distance in
the learned embedding space.
"""

from src.prototypical_network.model import PrototypicalNetwork
from src.prototypical_network.trainer import ProtoNetTrainer
from src.prototypical_network.distance import (
    euclidean_distance,
    cosine_distance,
    get_distance_fn,
)

__all__ = [
    "PrototypicalNetwork",
    "ProtoNetTrainer",
    "euclidean_distance",
    "cosine_distance",
    "get_distance_fn",
]
