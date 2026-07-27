import argparse
import copy
import os
import sys

import torch
import torch.nn.functional as F
from PIL import Image, ImageOps

TRANSFORMERS_SRC = os.environ.get("TRANSFORMERS_SRC", "/data/tmp/yongqiang/nfs/lhj/transformers/src")
if TRANSFORMERS_SRC not in sys.path:
    sys.path.insert(0, TRANSFORMERS_SRC)

from modeling_qwen3_5_export import Qwen3_5ForConditionalGenerationONNX

try:
    from transformers import AutoProcessor, Qwen3_5ForConditionalGeneration
except ImportError:
    from transformers import AutoProcessor
    from transformers.models.qwen3_5.modeling_qwen3_5 import Qwen3_5ForConditionalGeneration


def parse_args():
    parser = argparse.ArgumentParser(description="Validate Qwen3.5 vision ONNX by replacing torch visual encoder.")
    parser.add_argument(
        "--model-path",
        default="/data/tmp/yongqiang/nfs/lhj/Qwen/Qwen3.5-2B/",
        help="HuggingFace model directory",
    )
    parser.add_argument("--onnx-path", default="qwen3_5_vision.onnx")
    parser.add_argument("--image-path", default="/home/lihongjie/AI-support/npu-codebase/RealWorld-04_384x384.png")
    parser.add_argument(
        "--image-size",
        nargs=2,
        type=int,
        default=(384, 384),
        metavar=("WIDTH", "HEIGHT"),
        help="Image size used by the fixed-shape ONNX model (default: 384 384).",
    )
    parser.add_argument("--prompt", default="这是哪里")
    parser.add_argument("--max-new-tokens", type=int, default=64, help="Generation length")
    parser.add_argument("--seed", type=int, default=0, help="Random seed")
    return parser.parse_args()


def clone_model_inputs(inputs):
    return {k: (v.clone() if torch.is_tensor(v) else copy.deepcopy(v)) for k, v in inputs.items()}


def hf_pixel_values_to_nchw(
    pixel_values_hf: torch.Tensor,
    image_grid_thw: torch.LongTensor,
    temporal_patch_size: int,
    patch_size: int,
) -> torch.Tensor:
    if pixel_values_hf.ndim != 2:
        raise ValueError(f"Expected HF pixel_values with shape [tokens, c*tpp], got {tuple(pixel_values_hf.shape)}")
    if image_grid_thw.ndim != 2 or image_grid_thw.shape[1] != 3:
        raise ValueError(f"Expected image_grid_thw shape [N, 3], got {tuple(image_grid_thw.shape)}")
    if image_grid_thw.shape[0] != 1:
        raise ValueError(
            "Current ONNX export/validation only supports exactly one image grid. "
            f"Got image_grid_thw with N={image_grid_thw.shape[0]}."
        )

    t, h, w = [int(v) for v in image_grid_thw[0].tolist()]
    seq_len = h * w
    expected_tokens = t * seq_len
    if pixel_values_hf.shape[0] != expected_tokens:
        raise ValueError(
            "HF pixel_values token count mismatch with image_grid_thw: "
            f"tokens={pixel_values_hf.shape[0]}, expected={expected_tokens} (t={t}, h={h}, w={w})."
        )

    tpp = int(temporal_patch_size) * int(patch_size) * int(patch_size)
    flat_dim = int(pixel_values_hf.shape[1])
    if flat_dim % tpp != 0:
        raise ValueError(
            f"HF pixel_values feature dim {flat_dim} is not divisible by temporal_patch_size*patch_size^2={tpp}."
        )

    in_channels = flat_dim // tpp
    pixel_values_nchw = pixel_values_hf.reshape(t, seq_len, in_channels, tpp).permute(0, 2, 1, 3).contiguous()
    return pixel_values_nchw


def main():
    args = parse_args()
    device = "cpu"
    torch.manual_seed(args.seed)

    print("[1/6] load processor")
    processor = AutoProcessor.from_pretrained(args.model_path)

    image_size = tuple(args.image_size)
    image_width, image_height = image_size
    patch_size = int(processor.image_processor.patch_size)
    merge_size = int(processor.image_processor.merge_size)
    resize_factor = patch_size * merge_size
    if image_width <= 0 or image_height <= 0:
        raise ValueError("--image-size values must be positive")
    if image_width % resize_factor or image_height % resize_factor:
        raise ValueError(
            f"--image-size WIDTH and HEIGHT must both be divisible by {resize_factor} "
            f"(patch_size={patch_size}, merge_size={merge_size})"
        )

    with Image.open(args.image_path) as source_image:
        image = ImageOps.exif_transpose(source_image).convert("RGB")
        image = image.resize(image_size, Image.Resampling.BICUBIC)
    expected_grid_thw = [1, image_height // patch_size, image_width // patch_size]
    print(f"image_size={image_size}, expected grid_thw={expected_grid_thw}")

    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": image},
                {"type": "text", "text": args.prompt},
            ],
        }
    ]
    inputs_torch = processor.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=True,
        return_dict=True,
        return_tensors="pt",
    )
    inputs_torch = inputs_torch.to(device)
    actual_grid_thw = inputs_torch["image_grid_thw"].reshape(-1).tolist()
    if actual_grid_thw != expected_grid_thw:
        raise ValueError(
            f"processor produced image_grid_thw={actual_grid_thw}, expected {expected_grid_thw} "
            "from --image-size"
        )

    # Keep HF-native vision layout for torch baseline; derive ONNX NCHW input by re-layout from HF pixel_values.
    pixel_values_nchw = hf_pixel_values_to_nchw(
        pixel_values_hf=inputs_torch["pixel_values"],
        image_grid_thw=inputs_torch["image_grid_thw"],
        temporal_patch_size=int(processor.image_processor.temporal_patch_size),
        patch_size=int(processor.image_processor.patch_size),
    )
    inputs_onnx = clone_model_inputs(inputs_torch)
    inputs_onnx["pixel_values"] = pixel_values_nchw.to(device)

    print(f"torch pixel_values shape={tuple(inputs_torch['pixel_values'].shape)}")
    print(f"onnx  pixel_values shape={tuple(inputs_onnx['pixel_values'].shape)}")

    num_image_tokens = int((inputs_torch["input_ids"] == processor.image_token_id).sum().item())
    expected_image_tokens = int(
        (inputs_torch["image_grid_thw"].prod(-1) // (processor.image_processor.merge_size**2)).sum().item()
    )
    if num_image_tokens != expected_image_tokens:
        raise ValueError(
            f"image token mismatch: input_ids has {num_image_tokens}, "
            f"but grid_thw implies {expected_image_tokens}. "
            "Please use the same preprocess settings as ONNX export."
        )

    print("[2/6] load torch baseline model")
    model_torch = Qwen3_5ForConditionalGeneration.from_pretrained(
        args.model_path,
        torch_dtype=torch.float32,
        device_map=device,
    ).eval()

    print("[3/6] load onnx-replaced model")
    model_onnx = Qwen3_5ForConditionalGenerationONNX.from_pretrained(
        args.model_path,
        torch_dtype=torch.float32,
        device_map=device,
    ).eval()
    model_onnx.model.visual.init_onnx_session(args.onnx_path, providers=["CPUExecutionProvider"])
    model_onnx.model.visual.forward = model_onnx.model.visual.forward_onnx_nchw
    onnx_input_shape = model_onnx.model.visual.session.get_inputs()[0].shape
    prepared_input_shape = tuple(inputs_onnx["pixel_values"].shape)
    if len(onnx_input_shape) == 4 and all(isinstance(value, int) for value in onnx_input_shape):
        expected_input_shape = tuple(int(value) for value in onnx_input_shape)
        if prepared_input_shape != expected_input_shape:
            raise ValueError(
                f"prepared hidden_states shape {prepared_input_shape} does not match ONNX input "
                f"shape {expected_input_shape}; use the same --image-size as ONNX export"
            )

    print("[4/6] compare vision features")
    with torch.no_grad():
        vision_torch = model_torch.get_image_features(
            pixel_values=inputs_torch["pixel_values"],
            image_grid_thw=inputs_torch["image_grid_thw"],
            return_dict=True,
        ).pooler_output[0].to(torch.float32)
        vision_onnx = model_onnx.get_image_features(
            pixel_values=inputs_onnx["pixel_values"],
            image_grid_thw=inputs_onnx["image_grid_thw"],
            return_dict=True,
        ).pooler_output[0].to(torch.float32)

    diff = (vision_torch - vision_onnx).abs()
    max_abs_diff = float(diff.max().item())
    mean_abs_diff = float(diff.mean().item())
    cosine = float(F.cosine_similarity(vision_torch.flatten().unsqueeze(0), vision_onnx.flatten().unsqueeze(0)).item())
    print(f"vision max_abs_diff={max_abs_diff:.8f}")
    print(f"vision mean_abs_diff={mean_abs_diff:.8f}")
    print(f"vision cosine={cosine:.8f}")

    print("[5/6] compare generation")
    gen_kwargs = {"max_new_tokens": args.max_new_tokens, "do_sample": False}
    with torch.no_grad():
        inputs_torch_gen = clone_model_inputs(inputs_torch)
        inputs_onnx_gen = clone_model_inputs(inputs_onnx)
        generated_torch = model_torch.generate(**inputs_torch_gen, **gen_kwargs)
        generated_onnx = model_onnx.generate(**inputs_onnx_gen, **gen_kwargs)

    trimmed_torch = [out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs_torch["input_ids"], generated_torch)]
    trimmed_onnx = [out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs_torch["input_ids"], generated_onnx)]
    same_ids = all(torch.equal(a, b) for a, b in zip(trimmed_torch, trimmed_onnx))
    text_torch = processor.batch_decode(
        trimmed_torch, skip_special_tokens=True, clean_up_tokenization_spaces=False
    )
    text_onnx = processor.batch_decode(
        trimmed_onnx, skip_special_tokens=True, clean_up_tokenization_spaces=False
    )
    same_text = text_torch == text_onnx

    print(f"same_generated_ids={same_ids}")
    print(f"same_generated_text={same_text}")

    print("[6/6] output")
    print("torch_text:", text_torch[0])
    print("onnx_text :", text_onnx[0])


if __name__ == "__main__":
    main()
