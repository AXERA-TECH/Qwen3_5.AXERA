import argparse
import os
import sys

import onnx
from onnx.shape_inference import infer_shapes
import onnxsim
import torch
from PIL import Image, ImageOps

TRANSFORMERS_SRC = os.environ.get("TRANSFORMERS_SRC", "/data/tmp/yongqiang/nfs/lhj/transformers/src")
if TRANSFORMERS_SRC not in sys.path:
    sys.path.insert(0, TRANSFORMERS_SRC)

from preprocess_qwen3_5_export import Qwen2VLImageProcessorExport, preprocess_image_to_nchw

try:
    from transformers import AutoProcessor
except ImportError:
    from transformers import AutoProcessor

from modeling_qwen3_5_export import Qwen3_5ForConditionalGenerationExport, compute_static_vision_tensors


def build_export_image_processor(processor, image_size):
    src = processor.image_processor
    image_width, image_height = image_size
    image_pixels = image_width * image_height

    return Qwen2VLImageProcessorExport(
        size={"shortest_edge": image_pixels, "longest_edge": image_pixels},
        patch_size=int(src.patch_size),
        temporal_patch_size=int(src.temporal_patch_size),
        merge_size=int(src.merge_size),
        image_mean=list(src.image_mean),
        image_std=list(src.image_std),
    )


def parse_args():
    parser = argparse.ArgumentParser(description="Export Qwen3.5 vision encoder to ONNX (fixed image size).")
    parser.add_argument(
        "--model-path",
        default="/data/tmp/yongqiang/nfs/lhj/Qwen/Qwen3.5-4B/",
        help="HuggingFace model directory",
    )
    parser.add_argument(
        "--image-size",
        nargs=2,
        type=int,
        default=(384, 384),
        metavar=("WIDTH", "HEIGHT"),
        help="Fixed input image size used to derive grid_thw (default: 384 384).",
    )
    parser.add_argument(
        "--image-path",
        default="",
        help="Optional image used as export input data; it is resized to --image-size.",
    )
    parser.add_argument(
        "--onnx-output",
        default="qwen3_5_4b_vision.onnx",
        help="Output ONNX path",
    )
    parser.add_argument(
        "--hidden-states",
        default="",
        help="Optional input tensor path (.pth) in [t, c, seq, tpp] layout. If empty, random tensor is used.",
    )
    parser.add_argument("--opset", type=int, default=16, help="ONNX opset version")
    parser.add_argument("--seed", type=int, default=0, help="Random seed for synthetic export input")
    return parser.parse_args()


def main():
    args = parse_args()

    image_size = tuple(args.image_size)
    image_width, image_height = image_size
    if image_width <= 0 or image_height <= 0:
        raise ValueError("--image-size values must be positive")

    processor = AutoProcessor.from_pretrained(args.model_path)
    src_image_processor = processor.image_processor
    patch_size = int(src_image_processor.patch_size)
    merge_size = int(src_image_processor.merge_size)
    resize_factor = patch_size * merge_size
    if image_width % resize_factor or image_height % resize_factor:
        raise ValueError(
            f"--image-size WIDTH and HEIGHT must both be divisible by {resize_factor} "
            f"(patch_size={patch_size}, merge_size={merge_size})"
        )

    grid_thw = torch.tensor(
        [1, image_height // patch_size, image_width // patch_size], dtype=torch.long
    ).reshape(1, 3)
    print(f"image_size={image_size}, derived grid_thw={grid_thw.reshape(-1).tolist()}")

    hidden_states_from_image = None
    if args.image_path:
        export_image_processor = build_export_image_processor(processor, image_size)
        with Image.open(args.image_path) as source_image:
            image = ImageOps.exif_transpose(source_image).convert("RGB")
            image = image.resize(image_size, Image.Resampling.BICUBIC)
        pixel_values_nchw, image_grid_thw = preprocess_image_to_nchw(image, export_image_processor)
        if not torch.equal(image_grid_thw, grid_thw):
            raise RuntimeError(
                f"image preprocessing produced grid_thw={image_grid_thw.reshape(-1).tolist()}, "
                f"expected {grid_thw.reshape(-1).tolist()} from --image-size"
            )
        hidden_states_from_image = pixel_values_nchw.to(torch.float32)

    print("[1/4] load export model")
    model = Qwen3_5ForConditionalGenerationExport.from_pretrained(
        args.model_path,
        torch_dtype=torch.float32,
        device_map="cpu",
    )
    model.eval()
    export_model = model.model.visual
    export_model.eval()

    print("[2/4] compute fixed-shape tensors in memory")
    static_tensors = compute_static_vision_tensors(export_model, grid_thw)
    export_model.set_static_tensors(static_tensors)
    export_model.forward = export_model.forward_export_nchw

    t, h, w = [int(v) for v in grid_thw.reshape(-1).tolist()]
    cfg = model.config.vision_config
    seq_len = h * w
    tpp = cfg.temporal_patch_size * cfg.patch_size * cfg.patch_size
    expected_shape = (t, cfg.in_channels, seq_len, tpp)

    if args.hidden_states:
        try:
            hidden_states = torch.load(args.hidden_states, map_location="cpu", weights_only=True)
        except TypeError:
            hidden_states = torch.load(args.hidden_states, map_location="cpu")
        hidden_states = hidden_states.to(torch.float32)
    elif hidden_states_from_image is not None:
        hidden_states = hidden_states_from_image
    else:
        torch.manual_seed(args.seed)
        hidden_states = torch.randn(expected_shape, dtype=torch.float32)

    if tuple(hidden_states.shape) != expected_shape:
        raise ValueError(
            f"hidden_states shape mismatch: expected {expected_shape}, got {tuple(hidden_states.shape)}"
        )

    print(f"[3/4] export onnx -> {args.onnx_output}")
    torch.onnx.export(
        export_model,
        (hidden_states,),
        args.onnx_output,
        input_names=["hidden_states"],
        output_names=["pooler_output"],
        opset_version=args.opset,
        dynamo=False,
    )

    print("[4/4] check onnx graph")
    onnx_model = onnx.load(args.onnx_output)
    onnx.checker.check_model(onnx_model)
    print("done")

    onnx_model = onnx.load(args.onnx_output)
    print("IR 版本:", onnx_model.ir_version)
    print("操作集:", onnx_model.opset_import)
    onnx_model = infer_shapes(onnx_model)
    # convert model
    model_simp, check = onnxsim.simplify(onnx_model)
    assert check, "Simplified ONNX model could not be validated"
    onnx.save(model_simp, args.onnx_output)
    print("onnx simpilfy successed, and model saved in {}".format(args.onnx_output))

if __name__ == "__main__":
    main()
