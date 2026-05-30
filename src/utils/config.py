"""
Central Configuration Module for FewShot-Industrial-Vision.

Provides structured, type-safe configuration using Python dataclasses
with full YAML serialization/deserialization support. Handles automatic
device detection (CUDA → MPS → CPU) and directory path resolution.

Usage:
    >>> from src.utils.config import get_default_config
    >>> config = get_default_config()
    >>> config.save("configs/experiment.yaml")
    >>> loaded = ProjectConfig.load("configs/experiment.yaml")
"""

from __future__ import annotations

import os
import platform
import socket
from dataclasses import dataclass, field, asdict
from datetime import datetime
from pathlib import Path
from typing import List, Optional, Dict, Any

import torch
import yaml


# ---------------------------------------------------------------------------
# Path Constants
# ---------------------------------------------------------------------------

# Resolve project root as two levels above this file (src/utils/config.py)
_PROJECT_ROOT: Path = Path(__file__).resolve().parents[2]

_DEFAULT_DATA_DIR: str = str(_PROJECT_ROOT / "data")
_DEFAULT_MODEL_DIR: str = str(_PROJECT_ROOT / "models")
_DEFAULT_RESULTS_DIR: str = str(_PROJECT_ROOT / "results")
_DEFAULT_LOG_DIR: str = str(_PROJECT_ROOT / "logs")
_DEFAULT_CONFIG_DIR: str = str(_PROJECT_ROOT / "configs")


# ---------------------------------------------------------------------------
# Helper: Device Detection
# ---------------------------------------------------------------------------

def _detect_device() -> str:
    """Detect the best available compute device.

    Priority order:
        1. NVIDIA CUDA GPU
        2. Apple Metal Performance Shaders (MPS)
        3. CPU fallback

    Returns:
        str: Device string compatible with ``torch.device()``.
    """
    if torch.cuda.is_available():
        return "cuda"
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


# ---------------------------------------------------------------------------
# Data Configuration
# ---------------------------------------------------------------------------

@dataclass
class DataConfig:
    """Configuration for dataset and image preprocessing.

    Attributes:
        image_size: Target spatial resolution (height == width).
        channels: Number of colour channels (1 = greyscale, 3 = RGB).
        classes: Ordered list of defect class labels.
        data_dir: Root directory containing raw / processed datasets.
        train_dir: Sub-directory for training splits.
        val_dir: Sub-directory for validation splits.
        test_dir: Sub-directory for test splits.
        augmentation_enabled: Whether to apply data augmentation during training.
        normalize_mean: Per-channel mean for normalisation (ImageNet defaults).
        normalize_std: Per-channel std for normalisation (ImageNet defaults).
    """

    image_size: int = 224
    channels: int = 3
    classes: List[str] = field(
        default_factory=lambda: ["normal", "scratch", "crack", "dent"]
    )
    data_dir: str = _DEFAULT_DATA_DIR
    train_dir: str = "train"
    val_dir: str = "val"
    test_dir: str = "test"
    augmentation_enabled: bool = True
    normalize_mean: List[float] = field(
        default_factory=lambda: [0.485, 0.456, 0.406]
    )
    normalize_std: List[float] = field(
        default_factory=lambda: [0.229, 0.224, 0.225]
    )

    @property
    def num_classes(self) -> int:
        """Return the number of defect classes."""
        return len(self.classes)

    @property
    def train_path(self) -> Path:
        """Fully resolved path to the training data."""
        return Path(self.data_dir) / self.train_dir

    @property
    def val_path(self) -> Path:
        """Fully resolved path to the validation data."""
        return Path(self.data_dir) / self.val_dir

    @property
    def test_path(self) -> Path:
        """Fully resolved path to the test data."""
        return Path(self.data_dir) / self.test_dir


# ---------------------------------------------------------------------------
# Training Configuration
# ---------------------------------------------------------------------------

@dataclass
class TrainingConfig:
    """Configuration for episodic (few-shot) training.

    Attributes:
        n_way: Number of classes sampled per episode.
        k_shot: Number of support examples per class.
        q_query: Number of query examples per class.
        episodes_per_epoch: Episodes to sample in one epoch.
        epochs: Maximum number of training epochs.
        lr: Initial learning rate.
        weight_decay: L2 regularisation coefficient.
        lr_scheduler: Type of learning rate scheduler.
        lr_step: Step size for StepLR scheduler.
        lr_gamma: Multiplicative decay factor for StepLR.
        patience: Early-stopping patience (epochs without improvement).
        min_delta: Minimum improvement to reset patience counter.
        gradient_clip_norm: Maximum gradient norm (0 to disable).
        seed: Global random seed for reproducibility.
        mixed_precision: Enable automatic mixed-precision training.
        num_workers: DataLoader worker processes.
        pin_memory: Pin host memory for faster GPU transfers.
    """

    n_way: int = 4
    k_shot: int = 5
    q_query: int = 15
    episodes_per_epoch: int = 100
    epochs: int = 200
    lr: float = 1e-3
    weight_decay: float = 1e-4
    lr_scheduler: str = "step"
    lr_step: int = 20
    lr_gamma: float = 0.5
    patience: int = 15
    min_delta: float = 1e-4
    gradient_clip_norm: float = 0.0
    seed: int = 42
    mixed_precision: bool = False
    num_workers: int = 4
    pin_memory: bool = True

    @property
    def episode_size(self) -> int:
        """Total images per episode (support + query)."""
        return self.n_way * (self.k_shot + self.q_query)


# ---------------------------------------------------------------------------
# Backbone Configuration
# ---------------------------------------------------------------------------

@dataclass
class BackboneConfig:
    """Configuration for the feature-extraction backbone.

    Attributes:
        name: Architecture identifier (e.g. ``'resnet18'``, ``'resnet50'``).
        embedding_dim: Dimensionality of the output embedding vector.
        pretrained: Load ImageNet-pretrained weights.
        freeze_layers: Freeze all backbone parameters (linear-probe mode).
        dropout_rate: Dropout probability before the final embedding layer.
        pool_type: Global pooling strategy (``'avg'`` or ``'max'``).
    """

    name: str = "resnet18"
    embedding_dim: int = 512
    pretrained: bool = True
    freeze_layers: bool = False
    dropout_rate: float = 0.0
    pool_type: str = "avg"


# ---------------------------------------------------------------------------
# Deployment Configuration
# ---------------------------------------------------------------------------

@dataclass
class DeploymentConfig:
    """Configuration for model export and production deployment.

    Attributes:
        onnx_opset: ONNX opset version for export.
        tensorrt_precision: TensorRT inference precision (``'fp32'``, ``'fp16'``, ``'int8'``).
        batch_size: Maximum batch size for inference optimisation.
        dynamic_axes: Enable dynamic batch-size axis in ONNX export.
        model_dir: Directory for serialised model artefacts.
        gradio_port: Port for the Gradio demo server.
        gradio_share: Create a public Gradio share link.
    """

    onnx_opset: int = 13
    tensorrt_precision: str = "fp16"
    batch_size: int = 1
    dynamic_axes: bool = True
    model_dir: str = _DEFAULT_MODEL_DIR
    gradio_port: int = 7860
    gradio_share: bool = False


# ---------------------------------------------------------------------------
# Project Configuration (top-level aggregate)
# ---------------------------------------------------------------------------

@dataclass
class ProjectConfig:
    """Root configuration aggregating all sub-configs.

    Provides serialisation helpers (``save`` / ``load``) and automatic
    device detection.

    Attributes:
        project_name: Human-readable project identifier.
        experiment_name: Name of the current experiment run.
        data: Dataset & preprocessing configuration.
        training: Training-loop configuration.
        backbone: Feature-extractor configuration.
        deployment: Export & serving configuration.
        device: Compute device (auto-detected if not set).
        project_root: Absolute path to the repository root.
        results_dir: Directory for evaluation outputs.
        log_dir: Directory for log files.
        config_dir: Directory for saved YAML configs.
    """

    project_name: str = "FewShot-Industrial-Vision"
    experiment_name: str = "default"
    data: DataConfig = field(default_factory=DataConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    backbone: BackboneConfig = field(default_factory=BackboneConfig)
    deployment: DeploymentConfig = field(default_factory=DeploymentConfig)
    device: str = field(default_factory=_detect_device)
    project_root: str = str(_PROJECT_ROOT)
    results_dir: str = _DEFAULT_RESULTS_DIR
    log_dir: str = _DEFAULT_LOG_DIR
    config_dir: str = _DEFAULT_CONFIG_DIR

    # ---- Serialisation ----

    def to_dict(self) -> Dict[str, Any]:
        """Convert the entire configuration tree to a plain dictionary."""
        return asdict(self)

    def save(self, path: str | Path) -> Path:
        """Persist the configuration as a YAML file.

        Args:
            path: Destination file path (directories created automatically).

        Returns:
            Path: Resolved path to the saved file.
        """
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        header = (
            f"# FewShot-Industrial-Vision Configuration\n"
            f"# Generated: {datetime.now().isoformat()}\n"
            f"# Host: {socket.gethostname()} | "
            f"Platform: {platform.system()} {platform.release()}\n"
            f"# Device: {self.device}\n\n"
        )
        with open(path, "w", encoding="utf-8") as fh:
            fh.write(header)
            yaml.dump(
                self.to_dict(),
                fh,
                default_flow_style=False,
                sort_keys=False,
                allow_unicode=True,
            )
        return path.resolve()

    @classmethod
    def load(cls, path: str | Path) -> "ProjectConfig":
        """Load a ``ProjectConfig`` from a YAML file.

        Args:
            path: Path to an existing YAML config file.

        Returns:
            ProjectConfig: Fully-hydrated configuration instance.

        Raises:
            FileNotFoundError: If *path* does not exist.
            yaml.YAMLError: If the file is not valid YAML.
        """
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"Config file not found: {path}")

        with open(path, "r", encoding="utf-8") as fh:
            raw: Dict[str, Any] = yaml.safe_load(fh)

        return cls(
            project_name=raw.get("project_name", "FewShot-Industrial-Vision"),
            experiment_name=raw.get("experiment_name", "default"),
            data=DataConfig(**raw.get("data", {})),
            training=TrainingConfig(**raw.get("training", {})),
            backbone=BackboneConfig(**raw.get("backbone", {})),
            deployment=DeploymentConfig(**raw.get("deployment", {})),
            device=raw.get("device", _detect_device()),
            project_root=raw.get("project_root", str(_PROJECT_ROOT)),
            results_dir=raw.get("results_dir", _DEFAULT_RESULTS_DIR),
            log_dir=raw.get("log_dir", _DEFAULT_LOG_DIR),
            config_dir=raw.get("config_dir", _DEFAULT_CONFIG_DIR),
        )

    # ---- Directory helpers ----

    def ensure_directories(self) -> None:
        """Create all configured directories if they do not exist."""
        for dir_path in (
            self.data.data_dir,
            self.deployment.model_dir,
            self.results_dir,
            self.log_dir,
            self.config_dir,
        ):
            Path(dir_path).mkdir(parents=True, exist_ok=True)

    # ---- Pretty printing ----

    def summary(self) -> str:
        """Return a human-readable summary of the configuration."""
        lines = [
            f"╔══════════════════════════════════════════════════╗",
            f"║  {self.project_name:^46}  ║",
            f"║  Experiment: {self.experiment_name:<35}  ║",
            f"╠══════════════════════════════════════════════════╣",
            f"║  Device : {self.device:<38}  ║",
            f"║  Classes: {', '.join(self.data.classes):<38}  ║",
            f"║  Image  : {self.data.image_size}×{self.data.image_size}×{self.data.channels:<29}  ║",
            f"║  Backbone: {self.backbone.name:<37}  ║",
            f"║  Embed  : {self.backbone.embedding_dim:<38}  ║",
            f"║  N-way  : {self.training.n_way:<38}  ║",
            f"║  K-shot : {self.training.k_shot:<38}  ║",
            f"║  Episodes: {self.training.episodes_per_epoch:<37}  ║",
            f"║  Epochs : {self.training.epochs:<38}  ║",
            f"║  LR     : {self.training.lr:<38}  ║",
            f"╚══════════════════════════════════════════════════╝",
        ]
        return "\n".join(lines)

    def __repr__(self) -> str:
        return self.summary()


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------

def get_default_config() -> ProjectConfig:
    """Create and return the default project configuration.

    This is the recommended entry-point for scripts and notebooks.

    Returns:
        ProjectConfig: A fresh ``ProjectConfig`` with all defaults applied.

    Example:
        >>> cfg = get_default_config()
        >>> print(cfg.device)
        'cuda'
    """
    return ProjectConfig()
