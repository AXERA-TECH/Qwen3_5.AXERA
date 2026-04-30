import argparse
import copy
import glob
import os
import sys

import onnxruntime as ort
import torch
import torch.nn.functional as F
from PIL import Image

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
    parser = argparse.ArgumentParser(description="Validate Qwen3.5 vision ONNX on video-frame inputs.")
    parser.add_argument(
        "--model-path",
        default="/data/tmp/yongqiang/nfs/lhj/Qwen/Qwen3.5-2B/",
        help="HuggingFace model directory",
    )
    parser.add_argument("--onnx-path", default="qwen3_5_vision.onnx")
    parser.add_argument("--video-frames-dir", default="video-test-03", help="Directory containing video frame images")
    parser.add_argument("--frame-pattern", default="*.jpg", help="Glob pattern for frame files")
    parser.add_argument("--prompt", default="请描述这个视频中的内容")
    parser.add_argument(
        "--static-tensors-path",
        default="vision_static_tensors.pth",
        help="Path to static tensor file saved during ONNX export (used to align video grid to static ONNX shape).",
    )
    parser.add_argument("--max-new-tokens", type=int, default=64, help="Generation length")
    parser.add_argument("--seed", type=int, default=0, help="Random seed")
    parser.add_argument(
        "--do-sample-frames",
        action="store_true",
        help="Enable internal processor frame sampling. By default, all provided frames are used.",
    )
    parser.add_argument(
        "--align-to-static-grid",
        action="store_true",
        default=True,
        help="Align frame count/resolution to static export grid from --static-tensors-path (default: enabled).",
    )
    parser.add_argument(
        "--no-align-to-static-grid",
        action="store_false",
        dest="align_to_static_grid",
        help="Disable static-grid alignment and use processor defaults directly.",
    )
    return parser.parse_args()


def clone_model_inputs(inputs):
    return {k: (v.clone() if torch.is_tensor(v) else copy.deepcopy(v)) for k, v in inputs.items()}


def hf_pixel_values_to_nchw(
    pixel_values_hf: torch.Tensor,
    grid_thw: torch.LongTensor,
    temporal_patch_size: int,
    patch_size: int,
) -> torch.Tensor:
    if pixel_values_hf.ndim != 2:
        raise ValueError(f"Expected HF pixel_values with shape [tokens, c*tpp], got {tuple(pixel_values_hf.shape)}")
    if grid_thw.ndim != 2 or grid_thw.shape[1] != 3:
        raise ValueError(f"Expected grid_thw shape [N, 3], got {tuple(grid_thw.shape)}")
    if grid_thw.shape[0] != 1:
        raise ValueError(
            "Current ONNX export/validation only supports exactly one video grid. "
            f"Got grid_thw with N={grid_thw.shape[0]}."
        )

    t, h, w = [int(v) for v in grid_thw[0].tolist()]
    seq_len = h * w
    expected_tokens = t * seq_len
    if pixel_values_hf.shape[0] != expected_tokens:
        raise ValueError(
            "HF pixel_values token count mismatch with grid_thw: "
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


def list_frame_paths(frame_dir: str, frame_pattern: str) -> list[str]:
    frame_paths = sorted(glob.glob(os.path.join(frame_dir, frame_pattern)))
    if not frame_paths:
        raise FileNotFoundError(f"No frames found in {frame_dir!r} with pattern {frame_pattern!r}")
    return frame_paths


def sample_frame_paths(frame_paths: list[str], target_frames: int) -> list[str]:
    if target_frames <= 0:
        raise ValueError(f"target_frames must be > 0, got {target_frames}")
    if len(frame_paths) == target_frames:
        return list(frame_paths)
    if len(frame_paths) > target_frames:
        if target_frames == 1:
            return [frame_paths[len(frame_paths) // 2]]
        last = len(frame_paths) - 1
        indices = [round(i * last / (target_frames - 1)) for i in range(target_frames)]
        return [frame_paths[i] for i in indices]
    return list(frame_paths) + [frame_paths[-1]] * (target_frames - len(frame_paths))


def load_frames(frame_paths: list[str], resize_to: tuple[int, int] | None = None) -> list[Image.Image]:
    resample_bicubic = Image.Resampling.BICUBIC if hasattr(Image, "Resampling") else Image.BICUBIC
    frames = []
    for p in frame_paths:
        img = Image.open(p).convert("RGB")
        if resize_to is not None and img.size != resize_to:
            img = img.resize(resize_to, resample_bicubic)
        frames.append(img)
    return frames


def _torch_load(path: str):
    try:
        return torch.load(path, map_location="cpu", weights_only=True)
    except TypeError:
        return torch.load(path, map_location="cpu")


def load_static_grid_thw(static_tensors_path: str) -> tuple[int, int, int] | None:
    if not static_tensors_path or not os.path.exists(static_tensors_path):
        return None
    static = _torch_load(static_tensors_path)
    if "grid_thw" not in static:
        return None
    grid = static["grid_thw"]
    if not torch.is_tensor(grid):
        grid = torch.tensor(grid, dtype=torch.long)
    grid = grid.to(dtype=torch.long).reshape(-1, 3)
    if grid.shape[0] != 1:
        return None
    t, h, w = [int(v) for v in grid[0].tolist()]
    return t, h, w


def get_onnx_hidden_states_shape(onnx_path: str) -> tuple[int, int, int, int] | None:
    session = ort.InferenceSession(onnx_path, providers=["CPUExecutionProvider"])
    input_shape = session.get_inputs()[0].shape
    if len(input_shape) != 4:
        return None
    if not all(isinstance(v, int) for v in input_shape):
        return None
    return tuple(int(v) for v in input_shape)


def main():
    args = parse_args()
    device = "cpu"
    torch.manual_seed(args.seed)

    print("[1/6] load processor and frames")
    processor = AutoProcessor.from_pretrained(args.model_path)
    frame_paths_all = list_frame_paths(args.video_frames_dir, args.frame_pattern)
    print(f"num_input_frames={len(frame_paths_all)}")
    print(f"first_frame={frame_paths_all[0]}")
    print(f"last_frame ={frame_paths_all[-1]}")

    video_kwargs = {"do_sample_frames": args.do_sample_frames}
    frame_paths_used = list(frame_paths_all)
    static_grid = load_static_grid_thw(args.static_tensors_path) if args.align_to_static_grid else None
    if static_grid is not None:
        temporal_patch_size = int(processor.video_processor.temporal_patch_size)
        patch_size = int(processor.video_processor.patch_size)
        target_t, target_h, target_w = static_grid
        target_raw_frames = target_t * temporal_patch_size
        target_size = (target_w * patch_size, target_h * patch_size)
        frame_paths_used = sample_frame_paths(frame_paths_all, target_raw_frames)
        frames = load_frames(frame_paths_used, resize_to=target_size)
        video_kwargs = {"do_sample_frames": False, "do_resize": False}
        print(
            "align_to_static_grid=True "
            f"grid_thw={list(static_grid)} raw_frames={target_raw_frames} "
            f"frame_size={target_size[0]}x{target_size[1]}"
        )
    else:
        frames = load_frames(frame_paths_used)
        if args.align_to_static_grid:
            print(
                "align_to_static_grid=True but static grid is unavailable; "
                "falling back to direct processor preprocessing."
            )
        else:
            print("align_to_static_grid=False")

    print(f"num_used_frames={len(frame_paths_used)}")
    print(f"used_first_frame={frame_paths_used[0]}")
    print(f"used_last_frame ={frame_paths_used[-1]}")

    messages = [
        {
            "role": "user",
            "content": [
                {"type": "video", "video": frames},
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
        **video_kwargs,
    )
    inputs_torch = inputs_torch.to(device)
    if "pixel_values_videos" not in inputs_torch or "video_grid_thw" not in inputs_torch:
        raise ValueError(f"Processor output missing video tensors. Available keys: {list(inputs_torch.keys())}")

    # Keep HF-native video layout for torch baseline; derive ONNX NCHW input by re-layout from HF pixel_values_videos.
    pixel_values_videos_nchw = hf_pixel_values_to_nchw(
        pixel_values_hf=inputs_torch["pixel_values_videos"],
        grid_thw=inputs_torch["video_grid_thw"],
        temporal_patch_size=int(processor.video_processor.temporal_patch_size),
        patch_size=int(processor.video_processor.patch_size),
    )
    inputs_onnx = clone_model_inputs(inputs_torch)
    inputs_onnx["pixel_values_videos"] = pixel_values_videos_nchw.to(device)

    print(f"torch pixel_values_videos shape={tuple(inputs_torch['pixel_values_videos'].shape)}")
    print(f"onnx  pixel_values_videos shape={tuple(inputs_onnx['pixel_values_videos'].shape)}")
    print(f"video_grid_thw={inputs_torch['video_grid_thw'].tolist()}")
    expected_hidden_shape = get_onnx_hidden_states_shape(args.onnx_path)
    if expected_hidden_shape is not None and tuple(inputs_onnx["pixel_values_videos"].shape) != expected_hidden_shape:
        raise ValueError(
            "Prepared ONNX hidden_states shape does not match model input shape: "
            f"prepared={tuple(inputs_onnx['pixel_values_videos'].shape)}, expected={expected_hidden_shape}. "
            "Use --align-to-static-grid with a matching --static-tensors-path or regenerate ONNX for this video grid."
        )

    num_video_tokens = int((inputs_torch["input_ids"] == processor.video_token_id).sum().item())
    expected_video_tokens = int(
        (inputs_torch["video_grid_thw"].prod(-1) // (processor.video_processor.merge_size**2)).sum().item()
    )
    if num_video_tokens != expected_video_tokens:
        raise ValueError(
            f"video token mismatch: input_ids has {num_video_tokens}, "
            f"but video_grid_thw implies {expected_video_tokens}. "
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

    print("[4/6] compare video features")
    with torch.no_grad():
        video_torch = model_torch.get_video_features(
            pixel_values_videos=inputs_torch["pixel_values_videos"],
            video_grid_thw=inputs_torch["video_grid_thw"],
            return_dict=True,
        ).pooler_output[0].to(torch.float32)
        video_onnx = model_onnx.get_video_features(
            pixel_values_videos=inputs_onnx["pixel_values_videos"],
            video_grid_thw=inputs_onnx["video_grid_thw"],
            return_dict=True,
        ).pooler_output[0].to(torch.float32)

    diff = (video_torch - video_onnx).abs()
    max_abs_diff = float(diff.max().item())
    mean_abs_diff = float(diff.mean().item())
    cosine = float(F.cosine_similarity(video_torch.flatten().unsqueeze(0), video_onnx.flatten().unsqueeze(0)).item())
    print(f"video max_abs_diff={max_abs_diff:.8f}")
    print(f"video mean_abs_diff={mean_abs_diff:.8f}")
    print(f"video cosine={cosine:.8f}")

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
