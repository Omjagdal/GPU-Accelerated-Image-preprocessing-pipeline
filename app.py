#!/usr/bin/env python3
"""
FewShot Industrial Defect Detection – Main Entry Point.

This script provides a unified CLI for all project operations:
* ``generate`` – Generate synthetic defect dataset
* ``train``    – Train the prototypical network
* ``evaluate`` – Run evaluation suite with visualisations
* ``export``   – Export model to ONNX (and optionally TensorRT)
* ``demo``     – Launch the Gradio interactive demo
* ``infer``    – Run inference on a single image

Usage:
    # Generate synthetic data
    python app.py generate

    # Train the model
    python app.py train --backbone resnet18 --epochs 50

    # Launch demo
    python app.py demo

    # Quick inference
    python app.py infer --image path/to/image.jpg
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Ensure project root is on the path
PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))


def cmd_generate(args: argparse.Namespace) -> None:
    """Generate synthetic defect dataset."""
    from src.preprocessing.preprocessing import generate_synthetic_dataset
    from src.utils.config import get_default_config

    config = get_default_config()
    output_dir = args.output_dir or config.data.data_dir

    generate_synthetic_dataset(
        output_dir=output_dir,
        train_count=args.train_count,
        test_count=args.test_count,
        support_count=args.support_count,
        image_size=args.image_size,
    )


def cmd_train(args: argparse.Namespace) -> None:
    """Train the prototypical network."""
    import torch

    from src.backbone.factory import get_backbone
    from src.preprocessing.image_loader import get_data_loaders
    from src.prototypical_network.model import PrototypicalNetwork
    from src.prototypical_network.trainer import ProtoNetTrainer
    from src.utils.config import get_default_config

    config = get_default_config()

    # Override config with CLI args
    if args.backbone:
        config.backbone.name = args.backbone
    if args.epochs:
        config.training.epochs = args.epochs
    if args.lr:
        config.training.lr = args.lr
    if args.k_shot:
        config.training.k_shot = args.k_shot

    # Set seed
    torch.manual_seed(config.training.seed)

    print(config.summary())

    # Build model
    backbone = get_backbone(
        config.backbone.name,
        pretrained=config.backbone.pretrained,
        embedding_dim=config.backbone.embedding_dim,
        freeze_layers=config.backbone.freeze_layers,
    )
    model = PrototypicalNetwork(backbone)
    print(f"\nModel: {model}")

    # Data loaders
    train_loader, test_loader = get_data_loaders(config)

    # Trainer
    trainer = ProtoNetTrainer(
        model=model,
        config=config,
        device=config.device,
        log_dir=str(Path(PROJECT_ROOT) / "logs"),
        model_dir=str(Path(PROJECT_ROOT) / "models"),
    )

    # Train
    history = trainer.train(train_loader, test_loader)

    # Save training history for evaluation
    import json
    history_path = Path(PROJECT_ROOT) / "results" / "training_history.json"
    history_path.parent.mkdir(parents=True, exist_ok=True)
    with open(history_path, "w") as f:
        json.dump(history, f, indent=2)
    print(f"Training history saved to {history_path}")


def cmd_evaluate(args: argparse.Namespace) -> None:
    """Run evaluation suite."""
    import json

    import torch

    from src.backbone.factory import get_backbone
    from src.evaluation.evaluator import FewShotEvaluator
    from src.preprocessing.augmentations import get_val_transforms
    from src.preprocessing.image_loader import DefectDataset, get_data_loaders
    from src.prototypical_network.model import PrototypicalNetwork
    from src.utils.config import get_default_config

    config = get_default_config()

    # Build model
    backbone = get_backbone(
        config.backbone.name,
        pretrained=True,
        embedding_dim=config.backbone.embedding_dim,
    )
    model = PrototypicalNetwork(backbone)

    # Load checkpoint
    model_path = args.model or str(Path(PROJECT_ROOT) / "models" / "best_model.pth")
    if Path(model_path).exists():
        checkpoint = torch.load(model_path, map_location=config.device)
        if "model_state_dict" in checkpoint:
            model.load_state_dict(checkpoint["model_state_dict"])
        print(f"Loaded model from {model_path}")
    else:
        print(f"⚠ No checkpoint at {model_path}, using pretrained backbone")

    model.to(config.device)

    # Data
    _, test_loader = get_data_loaders(config)

    test_dataset = DefectDataset(
        root_dir=str(config.data.test_path),
        transform=get_val_transforms(config.data.image_size),
        classes=config.data.classes,
    )

    # Load training history if available
    history = None
    history_path = Path(PROJECT_ROOT) / "results" / "training_history.json"
    if history_path.exists():
        with open(history_path, "r") as f:
            history = json.load(f)

    # Evaluate
    evaluator = FewShotEvaluator(model, config, device=config.device)
    results = evaluator.full_evaluation(
        test_loader, test_dataset, training_history=history,
    )


def cmd_export(args: argparse.Namespace) -> None:
    """Export model to ONNX and optionally TensorRT."""
    import torch

    from src.backbone.factory import get_backbone
    from src.deployment.exporter import ONNXExporter, TensorRTOptimizer
    from src.prototypical_network.model import PrototypicalNetwork
    from src.utils.config import get_default_config

    config = get_default_config()

    # Build model
    backbone = get_backbone(config.backbone.name, pretrained=True)
    model = PrototypicalNetwork(backbone)

    # Load checkpoint
    model_path = args.model or str(Path(PROJECT_ROOT) / "models" / "best_model.pth")
    if Path(model_path).exists():
        checkpoint = torch.load(model_path, map_location="cpu")
        if "model_state_dict" in checkpoint:
            model.load_state_dict(checkpoint["model_state_dict"])
        print(f"Loaded model from {model_path}")

    # Export backbone to ONNX (the backbone is what we deploy)
    onnx_path = str(Path(PROJECT_ROOT) / "models" / "backbone.onnx")
    exporter = ONNXExporter(
        model=model.backbone,
        device="cpu",
        input_shape=(3, config.data.image_size, config.data.image_size),
        opset_version=config.deployment.onnx_opset,
    )
    exporter.export(onnx_path)

    # Benchmark ONNX
    print("\nBenchmarking ONNX Runtime...")
    onnx_stats = exporter.benchmark_onnx(onnx_path)
    print(f"  Mean latency: {onnx_stats['mean_ms']:.2f}ms")
    print(f"  P95 latency:  {onnx_stats['p95_ms']:.2f}ms")

    # TensorRT (optional)
    if args.tensorrt:
        trt_opt = TensorRTOptimizer()
        if trt_opt.is_available:
            engine_path = str(Path(PROJECT_ROOT) / "models" / "backbone.engine")
            trt_opt.build_engine(
                onnx_path, engine_path,
                precision=config.deployment.tensorrt_precision,
            )


def cmd_demo(args: argparse.Namespace) -> None:
    """Launch Gradio demo."""
    from src.deployment.gradio_app import create_demo

    demo = create_demo(device=args.device)
    demo.launch(
        server_name=args.host,
        server_port=args.port,
        share=args.share,
    )


def cmd_infer(args: argparse.Namespace) -> None:
    """Run inference on a single image."""
    from src.deployment.inference import Inferencer
    from src.utils.config import get_default_config

    config = get_default_config()

    model_path = args.model or str(Path(PROJECT_ROOT) / "models" / "best_model.pth")
    support_dir = args.support_dir or str(Path(config.data.data_dir) / "support")

    inferencer = Inferencer.from_checkpoint(
        checkpoint_path=model_path,
        backbone_name=args.backbone or config.backbone.name,
        class_names=config.data.classes,
        device=config.device,
    )

    # Load support set
    inferencer.load_support_set(support_dir)

    # Predict
    pred_class, confidences = inferencer.predict(args.image)

    print(f"\n{'='*50}")
    print(f"  Image: {args.image}")
    print(f"  Prediction: {pred_class.upper()}")
    print(f"{'='*50}")
    for cls_name, conf in sorted(confidences.items(), key=lambda x: -x[1]):
        bar = "█" * int(conf * 30)
        print(f"  {cls_name:>10s}: {conf:6.1%} {bar}")
    print(f"{'='*50}")

    stats = inferencer.get_latency_stats()
    print(f"  Latency: {stats['mean_ms']:.1f}ms")


# ---------------------------------------------------------------------------
# CLI Setup
# ---------------------------------------------------------------------------

def main() -> None:
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(
        description="FewShot Industrial Defect Detection System",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # ----- generate -----
    gen_parser = subparsers.add_parser("generate", help="Generate synthetic dataset")
    gen_parser.add_argument("--output-dir", type=str, default=None)
    gen_parser.add_argument("--train-count", type=int, default=50)
    gen_parser.add_argument("--test-count", type=int, default=20)
    gen_parser.add_argument("--support-count", type=int, default=10)
    gen_parser.add_argument("--image-size", type=int, default=224)

    # ----- train -----
    train_parser = subparsers.add_parser("train", help="Train prototypical network")
    train_parser.add_argument("--backbone", type=str, default=None)
    train_parser.add_argument("--epochs", type=int, default=None)
    train_parser.add_argument("--lr", type=float, default=None)
    train_parser.add_argument("--k-shot", type=int, default=None)

    # ----- evaluate -----
    eval_parser = subparsers.add_parser("evaluate", help="Run evaluation suite")
    eval_parser.add_argument("--model", type=str, default=None)

    # ----- export -----
    export_parser = subparsers.add_parser("export", help="Export to ONNX/TensorRT")
    export_parser.add_argument("--model", type=str, default=None)
    export_parser.add_argument("--tensorrt", action="store_true")

    # ----- demo -----
    demo_parser = subparsers.add_parser("demo", help="Launch Gradio demo")
    demo_parser.add_argument("--host", type=str, default="0.0.0.0")
    demo_parser.add_argument("--port", type=int, default=7860)
    demo_parser.add_argument("--share", action="store_true")
    demo_parser.add_argument("--device", type=str, default=None)

    # ----- infer -----
    infer_parser = subparsers.add_parser("infer", help="Classify a single image")
    infer_parser.add_argument("--image", type=str, required=True)
    infer_parser.add_argument("--model", type=str, default=None)
    infer_parser.add_argument("--backbone", type=str, default=None)
    infer_parser.add_argument("--support-dir", type=str, default=None)

    args = parser.parse_args()

    if args.command is None:
        parser.print_help()
        return

    command_map = {
        "generate": cmd_generate,
        "train": cmd_train,
        "evaluate": cmd_evaluate,
        "export": cmd_export,
        "demo": cmd_demo,
        "infer": cmd_infer,
    }

    command_map[args.command](args)


if __name__ == "__main__":
    main()
