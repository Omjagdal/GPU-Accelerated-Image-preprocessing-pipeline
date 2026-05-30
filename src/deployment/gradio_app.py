"""
Gradio-based Interactive Demo for Prototypical Network Defect Detection.

Provides a polished web UI with:
* **Upload & Classify** – Upload an image to predict defect class
* **Support Set Viewer** – Display current support images per class
* **Backbone Selection** – Choose ResNet18 / ResNet34 / EfficientNet
* **Shot Setting** – Adjust K-shot (1/5/10) to see accuracy impact
* **Results Dashboard** – Accuracy metrics, confusion matrix, latency

Usage (standalone):
    >>> from src.deployment.gradio_app import create_demo
    >>> demo = create_demo()
    >>> demo.launch()

Or from the project root:
    $ python app.py
"""

from __future__ import annotations

import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))


def create_demo(
    model_path: Optional[str] = None,
    support_dir: Optional[str] = None,
    device: Optional[str] = None,
) -> Any:
    """Create and configure the Gradio demo application.

    Args:
        model_path: Path to trained model checkpoint.
            Defaults to ``models/best_model.pth``.
        support_dir: Path to support set directory.
            Defaults to ``data/support/``.
        device: Compute device. Auto-detected if ``None``.

    Returns:
        A configured ``gradio.Blocks`` application.
    """
    import gradio as gr
    import torch
    from PIL import Image

    from src.backbone.factory import get_backbone
    from src.deployment.inference import Inferencer
    from src.preprocessing.augmentations import get_val_transforms
    from src.prototypical_network.model import PrototypicalNetwork
    from src.utils.config import get_default_config

    # ------------------------------------------------------------------
    # Configuration
    # ------------------------------------------------------------------
    config = get_default_config()
    class_names = config.data.classes

    if device is None:
        device = config.device
    if model_path is None:
        model_path = str(Path(config.deployment.model_dir) / "best_model.pth")
    if support_dir is None:
        support_dir = str(Path(config.data.data_dir) / "support")

    # ------------------------------------------------------------------
    # State management
    # ------------------------------------------------------------------
    inferencer_state: Dict[str, Any] = {
        "inferencer": None,
        "backbone_name": "resnet18",
        "is_loaded": False,
    }

    def _load_model(backbone_name: str = "resnet18") -> str:
        """Load model and support set."""
        try:
            backbone = get_backbone(backbone_name, pretrained=True)
            model = PrototypicalNetwork(backbone)

            # Try loading trained weights
            if Path(model_path).exists():
                checkpoint = torch.load(model_path, map_location=device)
                if "model_state_dict" in checkpoint:
                    model.load_state_dict(checkpoint["model_state_dict"])
                else:
                    model.load_state_dict(checkpoint)
                status = f"✓ Loaded trained model ({backbone_name})"
            else:
                status = f"⚠ No checkpoint found. Using pretrained {backbone_name}"

            inferencer = Inferencer(
                model=model,
                class_names=class_names,
                device=device,
            )

            # Load support set
            if Path(support_dir).exists():
                inferencer.load_support_set(support_dir)
                status += f"\n✓ Support set loaded from {support_dir}"
            else:
                status += f"\n⚠ Support set not found at {support_dir}"
                status += "\nRun data generation first: python -c 'from src.preprocessing.preprocessing import generate_synthetic_dataset; generate_synthetic_dataset(\"data/\")'"

            inferencer_state["inferencer"] = inferencer
            inferencer_state["backbone_name"] = backbone_name
            inferencer_state["is_loaded"] = True

            return status

        except Exception as e:
            return f"✗ Error loading model: {str(e)}"

    # ------------------------------------------------------------------
    # Prediction function
    # ------------------------------------------------------------------

    def classify_image(
        image: np.ndarray | None,
        backbone_name: str,
    ) -> Tuple[Dict[str, float], str, str]:
        """Classify an uploaded image.

        Returns:
            (confidences_dict, prediction_text, latency_text)
        """
        if image is None:
            return {}, "Please upload an image", ""

        # Reload model if backbone changed
        if (
            not inferencer_state["is_loaded"]
            or inferencer_state["backbone_name"] != backbone_name
        ):
            status = _load_model(backbone_name)
            if "Error" in status:
                return {}, status, ""

        inferencer = inferencer_state["inferencer"]
        if inferencer is None or inferencer.prototypes is None:
            return (
                {},
                "Model not ready. Load support set first.",
                "",
            )

        try:
            # Convert numpy array to PIL Image
            pil_image = Image.fromarray(image)
            pred_class, confidences = inferencer.predict(pil_image)

            # Format results
            prediction_text = (
                f"🔍 **Predicted Class:** {pred_class.upper()}\n"
                f"📊 **Confidence:** {confidences[pred_class]:.1%}"
            )

            stats = inferencer.get_latency_stats()
            latency_text = (
                f"⚡ Latency: {stats.get('mean_ms', 0):.1f}ms "
                f"(last {stats.get('num_inferences', 0)} inferences)"
            )

            return confidences, prediction_text, latency_text

        except Exception as e:
            return {}, f"Error: {str(e)}", ""

    # ------------------------------------------------------------------
    # Support set gallery
    # ------------------------------------------------------------------

    def get_support_gallery() -> List[Tuple[str, str]]:
        """Get support set images for the gallery."""
        gallery = []
        support_path = Path(support_dir)
        if not support_path.exists():
            return gallery

        for cls_name in class_names:
            cls_dir = support_path / cls_name
            if cls_dir.exists():
                images = sorted(cls_dir.glob("*.jpg"))[:3]
                for img_path in images:
                    gallery.append((str(img_path), cls_name))

        return gallery

    # ------------------------------------------------------------------
    # Training trigger
    # ------------------------------------------------------------------

    def run_training(
        backbone_name: str,
        epochs: int,
        k_shot: int,
        lr: float,
    ) -> str:
        """Trigger model training with specified parameters."""
        try:
            from src.preprocessing.image_loader import get_data_loaders
            from src.prototypical_network.trainer import ProtoNetTrainer

            # Update config
            config.backbone.name = backbone_name
            config.training.epochs = int(epochs)
            config.training.k_shot = int(k_shot)
            config.training.lr = float(lr)

            # Build model
            backbone = get_backbone(backbone_name, pretrained=True)
            model = PrototypicalNetwork(backbone)

            # Create data loaders
            train_loader, test_loader = get_data_loaders(config)

            # Train
            trainer = ProtoNetTrainer(
                model=model,
                config=config,
                device=device,
                log_dir=str(Path(PROJECT_ROOT) / "logs"),
                model_dir=str(Path(PROJECT_ROOT) / "models"),
            )

            history = trainer.train(train_loader, test_loader)

            return (
                f"✓ Training complete!\n"
                f"  Best accuracy: {trainer.best_val_acc:.4f}\n"
                f"  Epochs run: {len(history['train_loss'])}\n"
                f"  Model saved to: models/best_model.pth"
            )

        except Exception as e:
            return f"✗ Training failed: {str(e)}"

    # ------------------------------------------------------------------
    # Build Gradio UI
    # ------------------------------------------------------------------

    # Custom CSS for premium look
    custom_css = """
    .gradio-container {
        font-family: 'Inter', 'Segoe UI', sans-serif !important;
    }
    .main-title {
        text-align: center;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-size: 2.2em !important;
        font-weight: 800 !important;
        margin-bottom: 5px !important;
    }
    .sub-title {
        text-align: center;
        color: #6b7280;
        font-size: 1.1em;
        margin-bottom: 20px;
    }
    """

    with gr.Blocks(
        title="FewShot Defect Detector",
        theme=gr.themes.Soft(
            primary_hue="indigo",
            secondary_hue="purple",
            neutral_hue="slate",
        ),
        css=custom_css,
    ) as demo:

        # Header
        gr.HTML(
            """
            <div style="text-align: center; padding: 20px;">
                <h1 class="main-title">🔬 FewShot Industrial Defect Detector</h1>
                <p class="sub-title">
                    GPU-Accelerated Prototypical Networks for Few-Shot Defect Classification
                </p>
            </div>
            """
        )

        with gr.Tabs() as tabs:

            # ---- Tab 1: Classification ----
            with gr.TabItem("🔍 Classify", id="classify"):
                with gr.Row():
                    with gr.Column(scale=1):
                        input_image = gr.Image(
                            label="Upload Defect Image",
                            type="numpy",
                            height=300,
                        )
                        backbone_dropdown = gr.Dropdown(
                            choices=["resnet18", "resnet34", "efficientnet"],
                            value="resnet18",
                            label="🧠 Backbone Architecture",
                        )
                        classify_btn = gr.Button(
                            "🔍 Classify Defect",
                            variant="primary",
                            size="lg",
                        )

                    with gr.Column(scale=1):
                        label_output = gr.Label(
                            label="📊 Classification Confidences",
                            num_top_classes=4,
                        )
                        prediction_text = gr.Markdown(
                            label="Prediction",
                            value="Upload an image to classify",
                        )
                        latency_text = gr.Markdown(
                            label="Performance",
                            value="",
                        )

                classify_btn.click(
                    fn=classify_image,
                    inputs=[input_image, backbone_dropdown],
                    outputs=[label_output, prediction_text, latency_text],
                )

            # ---- Tab 2: Support Set ----
            with gr.TabItem("📚 Support Set", id="support"):
                gr.Markdown(
                    "### Support Set Gallery\n"
                    "These representative images define each defect class. "
                    "The model computes prototypes by averaging their embeddings."
                )
                support_gallery = gr.Gallery(
                    label="Support Images",
                    columns=4,
                    height=400,
                    value=get_support_gallery,
                )
                refresh_btn = gr.Button("🔄 Refresh Gallery")
                refresh_btn.click(
                    fn=get_support_gallery,
                    outputs=[support_gallery],
                )

            # ---- Tab 3: Training ----
            with gr.TabItem("🏋️ Training", id="training"):
                gr.Markdown(
                    "### Train the Model\n"
                    "Configure and launch episodic training. "
                    "Ensure synthetic data has been generated first."
                )
                with gr.Row():
                    train_backbone = gr.Dropdown(
                        choices=["resnet18", "resnet34", "efficientnet"],
                        value="resnet18",
                        label="Backbone",
                    )
                    train_epochs = gr.Slider(
                        minimum=5, maximum=500, value=50, step=5,
                        label="Epochs",
                    )
                with gr.Row():
                    train_kshot = gr.Slider(
                        minimum=1, maximum=20, value=5, step=1,
                        label="K-Shot",
                    )
                    train_lr = gr.Number(
                        value=0.001,
                        label="Learning Rate",
                    )
                train_btn = gr.Button(
                    "🚀 Start Training",
                    variant="primary",
                    size="lg",
                )
                train_output = gr.Markdown(label="Training Status")

                train_btn.click(
                    fn=run_training,
                    inputs=[train_backbone, train_epochs, train_kshot, train_lr],
                    outputs=[train_output],
                )

            # ---- Tab 4: Model Info ----
            with gr.TabItem("ℹ️ About", id="about"):
                gr.Markdown(
                    """
                    ## Architecture Overview

                    This system uses **Prototypical Networks** for few-shot
                    learning, classifying industrial defects with only 5-10
                    examples per class.

                    ### How It Works
                    1. **Support Set** → Images representing each defect class
                    2. **Backbone CNN** → Extracts 512-dim feature embeddings
                    3. **Prototypes** → Class centroids in embedding space
                    4. **Classification** → Nearest-prototype assignment

                    ### Defect Classes
                    | Class | Description |
                    |-------|-------------|
                    | Normal | Clean metallic surface |
                    | Scratch | Linear surface abrasions |
                    | Crack | Jagged fracture patterns |
                    | Dent | Circular concavity deformations |

                    ### Backbone Options
                    | Backbone | Params | Embedding Dim |
                    |----------|--------|--------------|
                    | ResNet-18 | 11.2M | 512 |
                    | ResNet-34 | 21.3M | 512 |
                    | EfficientNet-B0 | 5.3M | 512 (projected) |
                    """
                )

        # Initial model load
        demo.load(
            fn=lambda: _load_model("resnet18"),
            outputs=[],
        )

    return demo


# ---------------------------------------------------------------------------
# CLI entry-point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    demo = create_demo()
    demo.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=False,
    )
