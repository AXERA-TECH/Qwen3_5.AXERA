#!/usr/bin/env python3
"""Generate Pulsar2 calibration inputs for a fixed-shape Qwen3.5 vision ONNX model.

The vision ONNX model consumes raw, unnormalised image patches in
``[T, C, H*W, temporal_patch_size*patch_size*patch_size]`` layout.  Pulsar2
expects those patches to be supplied as U8 images, so this script serialises
each ``[H*W, tpp, C]`` patch grid to a JPEG and packages the generated files
into the tar archive referenced by ``config.json``.
"""

from __future__ import annotations

import argparse
import json
import os
import random
import sys
import tarfile
from pathlib import Path
from typing import Any


IMAGE_SUFFIXES = {".bmp", ".jpeg", ".jpg", ".png", ".webp"}
PATCH_SIZE = 16
TEMPORAL_PATCH_SIZE = 2
MERGE_SIZE = 2


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Create U8 patch-grid JPEGs and hidden_states.tar for Qwen3.5 "
            "Vision Encoder Pulsar2 calibration."
        )
    )
    parser.add_argument(
        "--input-dir",
        type=Path,
        required=True,
        help="Image directory to search recursively.",
    )
    parser.add_argument("--output-dir", type=Path, default=Path("calib_img"))
    parser.add_argument("--num-samples", type=int, default=8, help="Number of calibration samples to generate")
    parser.add_argument(
        "--image-size",
        type=int,
        nargs=2,
        default=(384, 384),
        metavar=("WIDTH", "HEIGHT"),
        help="Source image size used to derive grid_thw (default: 384 384)",
    )
    parser.add_argument("--seed", type=int, default=0, help="Deterministic source-image selection seed")
    parser.add_argument("--no-archive", action="store_true", help="Do not create hidden_states.tar")
    return parser.parse_args()


def build_export_image_processor(image_size: tuple[int, int]) -> Any:
    """Create the fixed Qwen3.5 image preprocessor used by the ONNX export path."""
    # Delay the transformers/torch import until after argument parsing and
    # source validation; even --help should remain fast on conversion hosts.
    transformers_src = os.environ.get("TRANSFORMERS_SRC")
    if transformers_src and transformers_src not in sys.path:
        sys.path.insert(0, transformers_src)

    from preprocess_qwen3_5_export import Qwen2VLImageProcessorExport

    image_width, image_height = image_size
    image_pixels = image_width * image_height

    return Qwen2VLImageProcessorExport(
        size={"shortest_edge": image_pixels, "longest_edge": image_pixels},
        patch_size=PATCH_SIZE,
        temporal_patch_size=TEMPORAL_PATCH_SIZE,
        merge_size=MERGE_SIZE,
        image_mean=[0.5, 0.5, 0.5],
        image_std=[0.5, 0.5, 0.5],
    )


def find_source_images(input_dir: Path) -> list[Path]:
    if not input_dir.is_dir():
        raise FileNotFoundError(f"input directory does not exist: {input_dir}")
    return sorted(
        path.resolve() for path in input_dir.rglob("*") if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES
    )


def select_sources(paths: list[Path], num_samples: int, seed: int) -> list[Path]:
    if not paths:
        raise ValueError("no supported source images found under --input-dir")
    if num_samples <= 0:
        raise ValueError("--num-samples must be positive")

    ordered = list(paths)
    random.Random(seed).shuffle(ordered)
    if len(ordered) < num_samples:
        print(
            f"warning: only {len(ordered)} unique source image(s) for {num_samples} samples; "
            "sources will be repeated"
        )
    return [ordered[index % len(ordered)] for index in range(num_samples)]


def make_patch_grid(
    source_path: Path,
    image_processor: Any,
    grid_thw: tuple[int, int, int],
) -> Any:
    """Return an unnormalised U8 tensor in the HWC layout stored as a JPEG."""
    import numpy as np
    from PIL import Image, ImageOps
    from transformers.image_utils import PILImageResampling

    grid_t, grid_h, grid_w = grid_thw
    if grid_t != 1:
        raise ValueError("image calibration only supports grid_thw T=1; use video-specific calibration for T > 1")

    target_size = (grid_w * image_processor.patch_size, grid_h * image_processor.patch_size)
    with Image.open(source_path) as source:
        image = ImageOps.exif_transpose(source).convert("RGB")
        image = image.resize(target_size, Image.Resampling.BICUBIC)

    patches, actual_grid_thw = image_processor._preprocess(
        [image],
        do_resize=True,
        resample=PILImageResampling.BICUBIC,
        do_rescale=False,
        do_normalize=False,
        do_convert_rgb=True,
    )
    actual_grid_thw = tuple(int(value) for value in actual_grid_thw)
    if actual_grid_thw != grid_thw:
        raise RuntimeError(
            f"preprocessing changed grid_thw for {source_path}: expected {grid_thw}, got {actual_grid_thw}"
        )

    expected_shape = (
        grid_t,
        grid_h * grid_w,
        image_processor.temporal_patch_size * image_processor.patch_size**2,
        3,
    )
    if tuple(patches.shape) != expected_shape:
        raise RuntimeError(f"unexpected patch-grid shape for {source_path}: expected {expected_shape}, got {patches.shape}")

    return np.rint(patches[0]).clip(0, 255).astype(np.uint8)


def main() -> None:
    args = parse_args()
    image_size = tuple(args.image_size)
    image_width, image_height = image_size
    resize_factor = PATCH_SIZE * MERGE_SIZE
    if image_width <= 0 or image_height <= 0:
        raise ValueError("--image-size values must be positive")
    if image_width % resize_factor or image_height % resize_factor:
        raise ValueError(
            f"--image-size WIDTH and HEIGHT must both be divisible by {resize_factor} "
            f"(patch_size={PATCH_SIZE}, merge_size={MERGE_SIZE})"
        )
    grid_thw = (1, image_height // PATCH_SIZE, image_width // PATCH_SIZE)
    print(f"image_size={image_size}, derived grid_thw={grid_thw}")

    source_paths = find_source_images(args.input_dir)
    selected_paths = select_sources(source_paths, args.num_samples, args.seed)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    import cv2

    image_processor = build_export_image_processor(image_size)
    generated_paths: list[Path] = []
    manifest_samples: list[dict[str, str]] = []
    patch_grid_shape: tuple[int, ...] | None = None
    for index, source_path in enumerate(selected_paths):
        patch_grid = make_patch_grid(source_path, image_processor, grid_thw)
        patch_grid_shape = tuple(patch_grid.shape)
        output_path = args.output_dir / f"h{index}.jpg"
        # OpenCV preserves the upstream BGR-on-disk convention. Pulsar2's
        # config.json declares tensor_format=BGR and restores the tensor order.
        if not cv2.imwrite(str(output_path), patch_grid):
            raise RuntimeError(f"failed to write calibration image: {output_path}")
        generated_paths.append(output_path)
        manifest_samples.append({"calibration_image": output_path.name, "source_image": str(source_path)})
        print(f"[{index + 1}/{args.num_samples}] {source_path} -> {output_path} ({patch_grid.shape})")

    manifest = {
        "image_size": list(image_size),
        "grid_thw": list(grid_thw),
        "patch_size": image_processor.patch_size,
        "temporal_patch_size": image_processor.temporal_patch_size,
        "patch_grid_shape": list(patch_grid_shape or ()),
        "samples": manifest_samples,
    }
    manifest_path = args.output_dir / "calib_manifest.json"
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    if not args.no_archive:
        archive_path = args.output_dir / "hidden_states.tar"
        with tarfile.open(archive_path, "w") as archive:
            for output_path in generated_paths:
                archive.add(output_path, arcname=output_path.name, recursive=False)
        print(f"created calibration archive: {archive_path} ({len(generated_paths)} images)")


if __name__ == "__main__":
    main()
