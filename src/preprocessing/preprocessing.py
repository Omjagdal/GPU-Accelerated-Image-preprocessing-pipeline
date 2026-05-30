"""
Synthetic industrial defect image generator.

Generates photorealistic metallic surface images with four defect
categories: **normal**, **scratch**, **crack**, and **dent**.  The
generator uses layered procedural techniques (multi-scale Gaussian noise,
directional brushing, random-walk cracks, radial dent gradients) to
produce training data when real-world labelled images are scarce – the
exact scenario few-shot learning is designed to address.

The generated dataset follows the directory layout expected by the rest
of the pipeline::

    <output_dir>/
    ├── train/
    │   ├── normal/   (N images)
    │   ├── scratch/
    │   ├── crack/
    │   └── dent/
    ├── test/
    │   └── ...
    └── support/
        └── ...

Usage:
    from src.preprocessing.preprocessing import generate_synthetic_dataset
    generate_synthetic_dataset("data/", train_count=50, test_count=20)
"""

import os
import random
from pathlib import Path
from typing import Callable, Dict, List, Tuple

import cv2
import numpy as np


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
_DEFECT_CLASSES: List[str] = ["normal", "scratch", "crack", "dent"]

# Base surface colour range (metallic grey tones)
_BASE_GREY_LOW = 140
_BASE_GREY_HIGH = 180


# ===================================================================
#  Base surface generation
# ===================================================================

def generate_base_surface(size: int = 224) -> np.ndarray:
    """Generate a realistic metallic / brushed-metal surface texture.

    The surface is built by layering several procedural effects:

    1. **Uniform grey base** with a subtle vertical brightness gradient
       that simulates directional ambient lighting.
    2. **Multi-scale Gaussian noise** – three octaves of noise at
       different blur radii are additively blended to create natural
       micro-texture variation reminiscent of rolled steel or aluminium.
    3. **Directional brushing lines** – faint horizontal streaks that
       emulate machining marks on a real metallic part.

    Args:
        size: Width and height of the output image in pixels.

    Returns:
        A ``numpy.ndarray`` of shape ``(size, size, 3)`` in BGR colour
        order with ``dtype=uint8``.
    """
    # --- Step 1: Base grey with vertical gradient ---------------------
    base_grey = np.random.randint(_BASE_GREY_LOW, _BASE_GREY_HIGH)
    surface = np.full((size, size), base_grey, dtype=np.float64)

    # Subtle vertical gradient (±8 intensity across height)
    gradient = np.linspace(-8, 8, size).reshape(-1, 1)
    surface += gradient

    # --- Step 2: Multi-scale Gaussian noise ---------------------------
    # Three octaves at different spatial frequencies create a natural
    # looking micro-texture that avoids the uniform feel of a flat fill.
    for sigma in (1, 3, 7):
        noise = np.random.randn(size, size) * 6.0
        blurred_noise = cv2.GaussianBlur(noise, (0, 0), sigmaX=sigma)
        surface += blurred_noise

    # --- Step 3: Directional brushing (horizontal streaks) ------------
    # Thin horizontal lines with very low opacity simulate machining.
    num_brush_lines = np.random.randint(20, 50)
    for _ in range(num_brush_lines):
        y = np.random.randint(0, size)
        thickness = np.random.randint(1, 3)
        intensity_shift = np.random.uniform(-4, 4)
        surface[max(0, y - thickness):y + thickness, :] += intensity_shift

    # --- Step 4: Final Gaussian smoothing for realism -----------------
    surface = cv2.GaussianBlur(surface, (3, 3), sigmaX=0.8)

    # Clip to valid range and convert to 3-channel BGR
    surface = np.clip(surface, 0, 255).astype(np.uint8)
    surface_bgr: np.ndarray = cv2.merge([surface, surface, surface])
    return surface_bgr


# ===================================================================
#  Individual defect generators
# ===================================================================

def generate_normal_image(size: int = 224) -> np.ndarray:
    """Generate a *clean* metallic surface with no defects.

    Minor natural variations (a very faint random speckle layer) are
    added so that the "normal" class is not unrealistically uniform.

    Args:
        size: Square image dimension in pixels.

    Returns:
        BGR ``uint8`` image of shape ``(size, size, 3)``.
    """
    surface = generate_base_surface(size)

    # Add minor speckle noise for natural look
    speckle = np.random.randn(size, size, 3) * 2.0
    surface = np.clip(surface.astype(np.float64) + speckle, 0, 255).astype(np.uint8)

    return surface


def generate_scratch_image(size: int = 224) -> np.ndarray:
    """Generate a metallic surface with **scratch** defects.

    Scratches are rendered as 1–3 thin bright lines at random angles.
    Each scratch has a soft-edge halo achieved by blending a blurred
    version of the scratch mask to simulate light scattering at the
    gouge edges.

    Args:
        size: Square image dimension in pixels.

    Returns:
        BGR ``uint8`` image of shape ``(size, size, 3)``.
    """
    surface = generate_base_surface(size)
    num_scratches = np.random.randint(1, 4)

    for _ in range(num_scratches):
        # Random start and end points – allow scratches to span most of
        # the image so they look realistic.
        margin = size // 8
        pt1 = (
            np.random.randint(margin, size - margin),
            np.random.randint(margin, size - margin),
        )
        pt2 = (
            np.random.randint(margin, size - margin),
            np.random.randint(margin, size - margin),
        )

        # Scratch colour: brighter than the surface (freshly exposed metal)
        scratch_brightness = np.random.randint(200, 240)
        scratch_colour = (scratch_brightness,) * 3
        thickness = np.random.randint(1, 3)

        # --- Draw the scratch core on a mask --------------------------
        scratch_mask = np.zeros((size, size, 3), dtype=np.uint8)
        cv2.line(scratch_mask, pt1, pt2, scratch_colour, thickness, lineType=cv2.LINE_AA)

        # --- Soft halo around the scratch edge ------------------------
        halo = cv2.GaussianBlur(scratch_mask, (5, 5), sigmaX=1.5)
        # Blend halo first (additive, low opacity), then scratch core
        surface = cv2.addWeighted(surface, 1.0, halo, 0.25, 0)
        surface = cv2.addWeighted(surface, 1.0, scratch_mask, 0.55, 0)

    return np.clip(surface, 0, 255).astype(np.uint8)


def generate_crack_image(size: int = 224) -> np.ndarray:
    """Generate a metallic surface with **crack** defects.

    Cracks are modelled as a jagged random-walk path with 1–3 branches.
    Each crack segment is dark (sub-surface shadow) and slightly noisy
    to emulate the irregularity of real fatigue cracks.

    Algorithm
    ---------
    1. Pick a random starting point near the centre.
    2. Perform a biased random walk to generate the main crack spine.
    3. At random points along the walk, spawn short branch cracks.
    4. Draw the paths with anti-aliased dark lines and add a slight
       Gaussian blur to soften the edges.

    Args:
        size: Square image dimension in pixels.

    Returns:
        BGR ``uint8`` image of shape ``(size, size, 3)``.
    """
    surface = generate_base_surface(size)

    def _random_walk_crack(
        start: Tuple[int, int],
        length: int,
        step_range: Tuple[int, int] = (3, 8),
    ) -> List[Tuple[int, int]]:
        """Generate a list of points via biased random walk."""
        points = [start]
        angle = np.random.uniform(0, 2 * np.pi)  # initial heading
        for _ in range(length):
            step = np.random.randint(*step_range)
            angle += np.random.uniform(-0.5, 0.5)  # slight direction change
            x = int(points[-1][0] + step * np.cos(angle))
            y = int(points[-1][1] + step * np.sin(angle))
            x = np.clip(x, 0, size - 1)
            y = np.clip(y, 0, size - 1)
            points.append((x, y))
        return points

    # --- Main crack spine ---------------------------------------------
    start = (
        np.random.randint(size // 4, 3 * size // 4),
        np.random.randint(size // 4, 3 * size // 4),
    )
    main_crack = _random_walk_crack(start, length=np.random.randint(15, 30))

    crack_mask = np.zeros((size, size, 3), dtype=np.uint8)
    crack_darkness = np.random.randint(30, 70)
    crack_colour = (crack_darkness,) * 3

    # Draw main crack
    for i in range(len(main_crack) - 1):
        thickness = np.random.choice([1, 1, 2])
        cv2.line(crack_mask, main_crack[i], main_crack[i + 1],
                 crack_colour, thickness, lineType=cv2.LINE_AA)

    # --- Branch cracks (1–3) ------------------------------------------
    num_branches = np.random.randint(1, 4)
    for _ in range(num_branches):
        branch_start_idx = np.random.randint(2, len(main_crack) - 2)
        branch_start = main_crack[branch_start_idx]
        branch = _random_walk_crack(branch_start, length=np.random.randint(5, 12),
                                     step_range=(2, 5))
        for i in range(len(branch) - 1):
            cv2.line(crack_mask, branch[i], branch[i + 1],
                     crack_colour, 1, lineType=cv2.LINE_AA)

    # --- Soften edges -------------------------------------------------
    crack_mask = cv2.GaussianBlur(crack_mask, (3, 3), sigmaX=0.8)

    # Overlay: where crack_mask is dark, darken the surface
    # We invert the overlay logic – the mask stores dark pixels, so we
    # subtract them from the surface.
    surface_f = surface.astype(np.float64)
    mask_f = crack_mask.astype(np.float64)
    # Scale: mask pixels range [0, crack_darkness]; we want a strong effect
    surface_f = surface_f - (255.0 - mask_f) * 0.15
    surface_f = np.clip(surface_f, 0, 255)
    return surface_f.astype(np.uint8)


def generate_dent_image(size: int = 224) -> np.ndarray:
    """Generate a metallic surface with **dent** (concavity) defects.

    Dents are rendered as 1–2 elliptical regions with a radial gradient:
    the centre is slightly lighter (stretched, reflective metal) while
    the rim is darker (shadow from the concave curvature).  A subtle
    shadow is cast on one side to enhance the 3-D impression.

    Args:
        size: Square image dimension in pixels.

    Returns:
        BGR ``uint8`` image of shape ``(size, size, 3)``.
    """
    surface = generate_base_surface(size)
    num_dents = np.random.randint(1, 3)

    for _ in range(num_dents):
        # Random centre and radii
        cx = np.random.randint(size // 5, 4 * size // 5)
        cy = np.random.randint(size // 5, 4 * size // 5)
        rx = np.random.randint(size // 10, size // 4)
        ry = np.random.randint(size // 10, size // 4)

        # Build a 2-D radial distance map from the centre
        yy, xx = np.mgrid[0:size, 0:size]
        dist = np.sqrt(((xx - cx) / max(rx, 1)) ** 2 + ((yy - cy) / max(ry, 1)) ** 2)

        # --- Concave dent profile: lighter centre, darker ring --------
        # Values < 1.0 are inside the ellipse
        dent_mask = np.clip(1.0 - dist, 0, 1)  # 1 at centre, 0 at rim

        # Intensity modulation: raise centre, lower edge
        centre_lift = 15.0 * dent_mask
        edge_darken = -20.0 * np.clip(dist - 0.5, 0, 0.5) * 2.0 * (dist < 1.2).astype(float)

        # --- Directional shadow (offset by a few pixels) ---------------
        shadow_offset_x = np.random.randint(3, 8)
        shadow_offset_y = np.random.randint(3, 8)
        dist_shadow = np.sqrt(
            ((xx - cx - shadow_offset_x) / max(rx, 1)) ** 2
            + ((yy - cy - shadow_offset_y) / max(ry, 1)) ** 2
        )
        shadow = -10.0 * np.clip(1.0 - dist_shadow, 0, 1)

        # Apply combined modulation
        surface_f = surface.astype(np.float64)
        modulation = centre_lift + edge_darken + shadow
        # Apply to all three channels equally
        for c in range(3):
            surface_f[:, :, c] += modulation
        surface = np.clip(surface_f, 0, 255).astype(np.uint8)

    return surface


# ===================================================================
#  Dataset generation orchestrator
# ===================================================================

# Mapping from class name → generator function
_GENERATORS: Dict[str, Callable[[int], np.ndarray]] = {
    "normal": generate_normal_image,
    "scratch": generate_scratch_image,
    "crack": generate_crack_image,
    "dent": generate_dent_image,
}


def generate_synthetic_dataset(
    output_dir: str,
    train_count: int = 50,
    test_count: int = 20,
    support_count: int = 10,
    image_size: int = 224,
) -> None:
    """Generate a complete synthetic dataset for all defect classes.

    Creates ``train/``, ``test/``, and ``support/`` splits under
    *output_dir*, each containing one sub-folder per class.  Images are
    saved as high-quality JPEG files.

    Args:
        output_dir: Root directory where the dataset will be created.
        train_count: Number of images **per class** in the training split.
        test_count: Number of images **per class** in the test split.
        support_count: Number of images **per class** in the support split.
        image_size: Square pixel dimension for generated images.

    Raises:
        OSError: If the output directory cannot be created.
    """
    output_path = Path(output_dir)
    splits: Dict[str, int] = {
        "train": train_count,
        "test": test_count,
        "support": support_count,
    }

    total_generated = 0

    for split_name, count in splits.items():
        for class_name in _DEFECT_CLASSES:
            class_dir = output_path / split_name / class_name
            class_dir.mkdir(parents=True, exist_ok=True)

            generator = _GENERATORS[class_name]
            for idx in range(count):
                image = generator(image_size)
                filename = f"{class_name}_{idx:04d}.jpg"
                filepath = class_dir / filename
                cv2.imwrite(
                    str(filepath),
                    image,
                    [cv2.IMWRITE_JPEG_QUALITY, 95],
                )
                total_generated += 1

    # ----- Print summary ------------------------------------------------
    print("=" * 60)
    print("  Synthetic Dataset Generation Complete")
    print("=" * 60)
    print(f"  Output directory : {output_path.resolve()}")
    print(f"  Image size       : {image_size}×{image_size}")
    print(f"  Classes          : {', '.join(_DEFECT_CLASSES)}")
    print("-" * 60)
    for split_name, count in splits.items():
        n_images = count * len(_DEFECT_CLASSES)
        print(f"  {split_name:>8s} : {count:>4d} per class × "
              f"{len(_DEFECT_CLASSES)} classes = {n_images:>5d} images")
    print("-" * 60)
    print(f"  Total images     : {total_generated}")
    print("=" * 60)


# ---------------------------------------------------------------------------
# CLI entry-point
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Generate a synthetic industrial defect dataset."
    )
    parser.add_argument(
        "--output-dir", type=str, default="data",
        help="Root directory for the generated dataset (default: data/).",
    )
    parser.add_argument("--train-count", type=int, default=50)
    parser.add_argument("--test-count", type=int, default=20)
    parser.add_argument("--support-count", type=int, default=10)
    parser.add_argument("--image-size", type=int, default=224)

    args = parser.parse_args()

    generate_synthetic_dataset(
        output_dir=args.output_dir,
        train_count=args.train_count,
        test_count=args.test_count,
        support_count=args.support_count,
        image_size=args.image_size,
    )
