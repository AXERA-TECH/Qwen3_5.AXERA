import os
from itertools import cycle
from types import MethodType
from typing import Any, Dict, Iterator, List

import torch
from datasets import load_dataset
from gptqmodel import GPTQModel, QuantizeConfig


MODEL_ID = os.environ.get("MODEL_ID", "../../Qwen/Qwen3.5-0.8B/")
QUANT_PATH = os.environ.get("QUANT_PATH", "../../Qwen/Qwen3.5-0.8B-GPTQ-Int4-EN")  

WIKI_DATASET_ID = os.environ.get("WIKI_DATASET_ID", "wikimedia/wikipedia")
WIKI_EN_CONFIG = os.environ.get("WIKI_EN_CONFIG", "20231101.en")
# WIKI_ZH_CONFIG = os.environ.get("WIKI_ZH_CONFIG", "20231101.zh")
WIKI_ZH_CONFIG = WIKI_EN_CONFIG

COCO_CAPTION_DATA_FILES = os.environ.get("COCO_CAPTION_DATA_FILES", "val-00001-of-00013.parquet")
COCO_CAPTION_SEED = int(os.environ.get("COCO_CAPTION_SEED", "42"))

NUM_CALIB = int(os.environ.get("NUM_CALIB", "1024"))
MIN_TEXT_CHARS = int(os.environ.get("MIN_TEXT_CHARS", "64"))
MAX_TEXT_CHARS = int(os.environ.get("MAX_TEXT_CHARS", "2048"))
SHUFFLE_BUFFER = int(os.environ.get("SHUFFLE_BUFFER", "10000"))
BATCH_SIZE = int(os.environ.get("BATCH_SIZE", "1"))

NUM_COCO_CALIB = int(os.environ.get("NUM_COCO_CALIB", str(NUM_CALIB * 3 // 4)))
COCO_CAPTION_STRATEGY = os.environ.get("COCO_CAPTION_STRATEGY", "shortest")
ADD_GENERATION_PROMPT = os.environ.get("ADD_GENERATION_PROMPT", "1") not in {"0", "false", "False"}

MODEL_DEVICE = os.environ.get("MODEL_DEVICE", "cuda:0")
CALIBRATION_DEVICE = os.environ.get("CALIBRATION_DEVICE", MODEL_DEVICE)
PRECOMPUTE_INPUTS_DEVICE = os.environ.get("PRECOMPUTE_INPUTS_DEVICE", CALIBRATION_DEVICE)
RELEASE_PRECOMPUTE_MODULES = os.environ.get("RELEASE_PRECOMPUTE_MODULES", "1").strip().lower() not in {
    "0",
    "false",
    "no",
    "off",
}
CALIBRATION_CACHE_PATH = os.environ.get("CALIBRATION_CACHE_PATH", "").strip()
ENABLE_DYNAMIC_QUANT_CONFIG = os.environ.get("ENABLE_DYNAMIC_QUANT_CONFIG", "1").strip().lower() not in {
    "0",
    "false",
    "no",
    "off",
}
SKIP_LINEAR_ATTN_FIRST_N = int(os.environ.get("SKIP_LINEAR_ATTN_FIRST_N", "3"))

_PREPARED_CALIBRATION_DATASET: List[Dict[str, torch.Tensor]] | None = None

MODEL_INPUT_KEYS = {
    "input_ids",
    "attention_mask",
    "position_ids",
    "inputs_embeds",
    "pixel_values",
    "pixel_values_videos",
    "image_grid_thw",
    "video_grid_thw",
    "mm_token_type_ids",
}

# COCO_PROMPTS = [
#     "请只根据图片中清楚可见的内容，用一两句话客观描述。不要猜测人物关系、地点、时间、情绪或看不见的细节。",
#     "Describe only what is clearly visible in the image in one or two concise sentences. Do not infer relationships, location, time, emotions, or hidden details.",
#     "请简洁说明图片里的主要物体、场景和动作；不确定的内容不要编造。",
#     "Give a concise, grounded caption for the image. Avoid adding details that are not visually evident.",
#     "请描述画面中可以确认的主体和背景。如果无法从图像确定，请不要推断。",
#     "What can be directly observed in this image? Answer conservatively and avoid speculation.",
# ]

# 中英文混合量化效果不好
COCO_PROMPTS = [
    "Describe only what is clearly visible in the image in one or two concise sentences. Do not infer relationships, location, time, emotions, or hidden details.",
    "Give a concise, grounded caption for the image. Avoid adding details that are not visually evident.",
    "What can be directly observed in this image? Answer conservatively and avoid speculation.",
]


def _build_dynamic_quant_config() -> Dict[str, Dict[str, Any]] | None:
    if not ENABLE_DYNAMIC_QUANT_CONFIG:
        return None

    if SKIP_LINEAR_ATTN_FIRST_N <= 0:
        return None

    skip_layers = "|".join(str(idx) for idx in range(SKIP_LINEAR_ATTN_FIRST_N))
    return {
        f"-:^model\\.language_model\\.layers\\.(?:{skip_layers})"
        r"\.linear_attn\.(?:in_proj_qkv|in_proj_z|out_proj)$": {},
    }


def _has_calibration_cache() -> bool:
    return bool(CALIBRATION_CACHE_PATH) and os.path.exists(CALIBRATION_CACHE_PATH)


def _load_calibration_cache() -> List[Dict[str, torch.Tensor]] | None:
    if not _has_calibration_cache():
        return None

    print(f"loading prepared calibration cache: {CALIBRATION_CACHE_PATH}")
    try:
        cache = torch.load(CALIBRATION_CACHE_PATH, map_location="cpu", weights_only=False)
    except TypeError:
        cache = torch.load(CALIBRATION_CACHE_PATH, map_location="cpu")

    if isinstance(cache, dict) and "encoded_batches" in cache:
        encoded_batches = cache["encoded_batches"]
    else:
        encoded_batches = cache

    if not isinstance(encoded_batches, list) or not encoded_batches:
        raise ValueError(f"invalid calibration cache: {CALIBRATION_CACHE_PATH}")

    print(f"loaded {len(encoded_batches)} prepared calibration batches from cache")
    return encoded_batches


def _save_calibration_cache(encoded_batches: List[Dict[str, torch.Tensor]]) -> None:
    if not CALIBRATION_CACHE_PATH or _has_calibration_cache():
        return

    cache_dir = os.path.dirname(CALIBRATION_CACHE_PATH)
    if cache_dir:
        os.makedirs(cache_dir, exist_ok=True)

    tmp_path = f"{CALIBRATION_CACHE_PATH}.tmp"
    cache = {
        "metadata": {
            "model_id": MODEL_ID,
            "num_calib": NUM_CALIB,
            "num_coco_calib": NUM_COCO_CALIB,
            "max_text_chars": MAX_TEXT_CHARS,
            "add_generation_prompt": ADD_GENERATION_PROMPT,
            "coco_caption_data_files": COCO_CAPTION_DATA_FILES,
        },
        "encoded_batches": encoded_batches,
    }
    torch.save(cache, tmp_path)
    os.replace(tmp_path, CALIBRATION_CACHE_PATH)
    print(f"saved prepared calibration cache: {CALIBRATION_CACHE_PATH}")


def _build_cached_calibration_placeholder() -> List[Dict[str, Any]]:
    placeholder = {
        "messages": [
            {
                "role": "user",
                "content": [{"type": "text", "text": "cached calibration placeholder"}],
            }
        ]
    }
    return [placeholder for _ in range(max(NUM_CALIB, 256))]


def _load_wikipedia_stream(config_name: str, seed: int):
    dataset = load_dataset(
        WIKI_DATASET_ID,
        config_name,
        split="train",
        streaming=True,
    )
    return dataset.shuffle(seed=seed, buffer_size=SHUFFLE_BUFFER)


def _normalize_text(text: str) -> str:
    compact = " ".join(text.split()).strip()
    if MAX_TEXT_CHARS > 0:
        compact = compact[:MAX_TEXT_CHARS]
    return compact


def _build_text_calibration_dataset(num_samples: int) -> List[Dict[str, Any]]:
    if num_samples <= 0:
        return []

    zh_stream: Iterator[Dict[str, str]] = iter(_load_wikipedia_stream(WIKI_ZH_CONFIG, seed=42))
    en_stream: Iterator[Dict[str, str]] = iter(_load_wikipedia_stream(WIKI_EN_CONFIG, seed=43))

    streams = {"zh": zh_stream, "en": en_stream}
    calibration_dataset: List[Dict[str, Any]] = []

    for lang in cycle(("zh", "en")):
        if len(calibration_dataset) >= num_samples:
            break

        try:
            row = next(streams[lang])
        except StopIteration as exc:
            raise RuntimeError(f"Wikipedia stream for `{lang}` exhausted unexpectedly.") from exc

        text = _normalize_text(str(row.get("text", "")))
        if len(text) < MIN_TEXT_CHARS:
            continue

        # 纯文本样本也使用 Qwen3VL processor 需要的多模态 content 列表格式。
        calibration_dataset.append(
            {
                "messages": [
                    {
                        "role": "user",
                        "content": [{"type": "text", "text": text}],
                    }
                ]
            }
        )

        if len(calibration_dataset) % 64 == 0:
            print(f"collected {len(calibration_dataset)}/{num_samples} text samples from Wikipedia (zh+en)")

    return calibration_dataset


def _resolve_coco_data_files(data_files_spec: str):
    data_files = [path.strip() for path in data_files_spec.split(",") if path.strip()]
    if not data_files:
        raise ValueError("COCO_CAPTION_DATA_FILES must contain at least one parquet path.")
    return data_files[0] if len(data_files) == 1 else data_files


def _select_caption(answer: Any, sample_index: int) -> str:
    if isinstance(answer, list):
        captions = [str(item).strip() for item in answer if str(item).strip()]
        if not captions:
            return ""
        if COCO_CAPTION_STRATEGY == "first":
            return captions[0]
        if COCO_CAPTION_STRATEGY == "cycle":
            return captions[sample_index % len(captions)]
        if COCO_CAPTION_STRATEGY == "longest":
            return max(captions, key=len)
        return min(captions, key=len)
    return str(answer).strip()


def _coco_prompt(sample_index: int) -> str:
    return COCO_PROMPTS[sample_index % len(COCO_PROMPTS)]


def _build_coco_caption_calibration_dataset(num_samples: int) -> List[Dict[str, Any]]:
    if num_samples <= 0:
        return []

    dataset = load_dataset(
        "parquet",
        data_files=_resolve_coco_data_files(COCO_CAPTION_DATA_FILES),
        split="train",
    ).shuffle(seed=COCO_CAPTION_SEED)

    calibration_dataset: List[Dict[str, Any]] = []
    for idx, row in enumerate(dataset):
        if len(calibration_dataset) >= num_samples:
            break

        image = row.get("image")
        caption = _select_caption(row.get("answer", ""), len(calibration_dataset))
        question = _coco_prompt(len(calibration_dataset))
        if image is None or not question:
            continue

        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": image},
                    {"type": "text", "text": question},
                ],
            }
        ]
        if caption:
            messages.append(
                {
                    "role": "assistant",
                    "content": [{"type": "text", "text": caption}],
                }
            )

        calibration_dataset.append(
            {
                "messages": messages,
                "source": "coco_caption",
            }
        )

        if len(calibration_dataset) % 64 == 0:
            print(f"collected {len(calibration_dataset)}/{num_samples} COCO-Caption samples")

    return calibration_dataset


def build_calibration_dataset(num_samples: int) -> List[Dict[str, Any]]:
    coco_samples = min(max(NUM_COCO_CALIB, 0), num_samples)
    text_samples = num_samples - coco_samples
    text_dataset = _build_text_calibration_dataset(text_samples)
    coco_dataset = _build_coco_caption_calibration_dataset(coco_samples)

    calibration_dataset: List[Dict[str, Any]] = []
    text_iter = iter(text_dataset)
    coco_iter = iter(coco_dataset)

    while len(calibration_dataset) < num_samples:
        try:
            calibration_dataset.append(next(coco_iter))
        except StopIteration:
            pass

        if len(calibration_dataset) >= num_samples:
            break

        try:
            calibration_dataset.append(next(text_iter))
        except StopIteration:
            pass

        if len(calibration_dataset) >= len(text_dataset) + len(coco_dataset):
            break

    print(
        f"ready calibration mix: {len(text_dataset)} text samples, "
        f"{len(coco_dataset)} COCO-Caption samples"
    )
    return calibration_dataset


def _has_assistant_message(messages: List[Dict[str, Any]]) -> bool:
    return any(message.get("role") == "assistant" for message in messages)


def _module_device(module: torch.nn.Module) -> torch.device:
    for tensor in module.parameters(recurse=True):
        if tensor.device.type != "meta":
            return tensor.device
    for tensor in module.buffers(recurse=True):
        if tensor.device.type != "meta":
            return tensor.device
    return torch.device("cpu")


def _materialize_precompute_modules(qmodel) -> None:
    if getattr(qmodel, "_qwen3_5_precompute_modules_ready", False):
        return

    device = torch.device(PRECOMPUTE_INPUTS_DEVICE)
    base_model = qmodel.model.model
    print(f"materializing embed/visual modules on {device} for calibration precompute")
    qmodel.shell_module_materialize(base_model.get_input_embeddings(), device)
    qmodel.shell_module_materialize(base_model.visual, device)
    qmodel._qwen3_5_precompute_modules_ready = True


def _release_precompute_modules(qmodel) -> None:
    if not RELEASE_PRECOMPUTE_MODULES:
        return

    base_model = qmodel.model.model
    qmodel.shell_module_materialize(base_model.get_input_embeddings(), torch.device("cpu"))
    qmodel.shell_module_materialize(base_model.visual, torch.device("cpu"))
    qmodel._qwen3_5_precompute_modules_ready = False


def _precompute_inputs_embeds(qmodel, encoded: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    model = qmodel.model
    base_model = model.model
    embed_tokens = base_model.get_input_embeddings()
    text_device = _module_device(embed_tokens)

    input_ids = encoded["input_ids"].to(text_device)
    attention_mask = encoded.get("attention_mask")
    if attention_mask is not None:
        attention_mask = attention_mask.to(text_device)

    with torch.no_grad():
        inputs_embeds = embed_tokens(input_ids)

        image_grid_thw = encoded.get("image_grid_thw")
        if encoded.get("pixel_values") is not None and image_grid_thw is not None:
            visual_device = _module_device(base_model.visual)
            pixel_values = encoded["pixel_values"].to(visual_device)
            image_grid_for_visual = image_grid_thw.to(visual_device)
            image_outputs = base_model.get_image_features(
                pixel_values=pixel_values,
                image_grid_thw=image_grid_for_visual,
                return_dict=True,
            )
            image_embeds = torch.cat(image_outputs.pooler_output, dim=0).to(inputs_embeds.device, inputs_embeds.dtype)
            image_mask, _ = base_model.get_placeholder_mask(
                input_ids,
                inputs_embeds=inputs_embeds,
                image_features=image_embeds,
            )
            inputs_embeds = inputs_embeds.masked_scatter(image_mask, image_embeds)

        position_ids = None
        mm_token_type_ids = encoded.get("mm_token_type_ids")
        if mm_token_type_ids is not None and image_grid_thw is not None:
            position_ids = base_model.compute_3d_position_ids(
                input_ids=input_ids,
                inputs_embeds=inputs_embeds,
                image_grid_thw=image_grid_thw.to(text_device),
                attention_mask=attention_mask,
                mm_token_type_ids=mm_token_type_ids.to(text_device),
            )

    batch = {
        "input_ids": input_ids.detach().cpu(),
        "inputs_embeds": inputs_embeds.detach().cpu(),
    }
    if attention_mask is not None:
        batch["attention_mask"] = attention_mask.detach().cpu()
    if position_ids is not None:
        batch["position_ids"] = position_ids.detach().cpu()
    return batch


def _patch_forward_for_precomputed_inputs(qmodel) -> None:
    original_forward = qmodel.model.forward

    def patched_forward(*args, **kwargs):
        if kwargs.get("inputs_embeds") is not None and kwargs.get("input_ids") is not None:
            kwargs = dict(kwargs)
            kwargs.pop("input_ids")
        return original_forward(*args, **kwargs)

    qmodel.model.forward = patched_forward


def _prepare_mixed_calibration_dataset(
    qmodel,
    calibration_dataset,
    calibration_dataset_concat_size=None,
    calibration_dataset_sort="desc",
    batch_size=1,
    calibration_data_min_length=10,
    calibration_concat_separator=None,
):
    global _PREPARED_CALIBRATION_DATASET

    del calibration_dataset_concat_size, calibration_concat_separator

    if _PREPARED_CALIBRATION_DATASET is not None:
        print("reusing prepared calibration batches from memory")
        return _PREPARED_CALIBRATION_DATASET

    cached_batches = _load_calibration_cache()
    if cached_batches is not None:
        _PREPARED_CALIBRATION_DATASET = cached_batches
        return cached_batches

    if batch_size != 1:
        print("multimodal calibration keeps one sample per batch; ignoring batch_size > 1")

    processor = getattr(qmodel, "processor", None) or getattr(qmodel, "tokenizer", None)
    if processor is None:
        raise RuntimeError("Qwen3.5 multimodal calibration requires model.processor or model.tokenizer.")

    _materialize_precompute_modules(qmodel)

    encoded_batches: List[Dict[str, torch.Tensor]] = []
    skipped = 0
    image_batches = 0

    for idx, example in enumerate(calibration_dataset):
        messages = example.get("messages")
        if messages is None:
            text = str(example.get("text", ""))
            messages = [{"role": "user", "content": [{"type": "text", "text": text}]}]

        encoded = processor.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=ADD_GENERATION_PROMPT and not _has_assistant_message(messages),
            return_dict=True,
            return_tensors="pt",
        )

        input_ids = encoded.get("input_ids")
        if input_ids is None or input_ids.shape[-1] <= calibration_data_min_length:
            skipped += 1
            continue

        encoded_tensors = {
            key: value.detach()
            for key, value in encoded.items()
            if key in MODEL_INPUT_KEYS and torch.is_tensor(value)
        }
        if "pixel_values" in encoded_tensors:
            image_batches += 1

        batch = _precompute_inputs_embeds(qmodel, encoded_tensors)
        encoded_batches.append(batch)

        if len(encoded_batches) % 64 == 0:
            print(f"encoded {len(encoded_batches)} multimodal/text calibration batches")

    if skipped:
        print(f"skipped {skipped} calibration samples shorter than {calibration_data_min_length} tokens")

    if calibration_dataset_sort == "asc":
        encoded_batches.sort(key=lambda item: item["input_ids"].shape[-1])
    elif calibration_dataset_sort == "desc":
        encoded_batches.sort(key=lambda item: item["input_ids"].shape[-1], reverse=True)

    total_tokens = sum(batch["attention_mask"].sum().item() for batch in encoded_batches if "attention_mask" in batch)
    print(
        f"prepared {len(encoded_batches)} one-sample calibration batches "
        f"({image_batches} with images precomputed into inputs_embeds, {total_tokens} non-padded tokens)"
    )
    _save_calibration_cache(encoded_batches)
    _PREPARED_CALIBRATION_DATASET = encoded_batches
    _release_precompute_modules(qmodel)
    return encoded_batches


calibration_dataset = (
    _build_cached_calibration_placeholder()
    if _has_calibration_cache()
    else build_calibration_dataset(NUM_CALIB)
)
print(f"ready to quantize with {len(calibration_dataset)} mixed calibration samples")

dynamic_quant_config = _build_dynamic_quant_config()
if dynamic_quant_config:
    print(f"dynamic quant skip rules: {dynamic_quant_config}")
elif not ENABLE_DYNAMIC_QUANT_CONFIG:
    print("dynamic quant skip rules disabled by ENABLE_DYNAMIC_QUANT_CONFIG")

quant_config = QuantizeConfig(
    bits=4,
    group_size=128,
    desc_act=False,
    static_groups=True,
    sym=True,
    mse=2.5,
    calibration_data_device=CALIBRATION_DEVICE,
    dynamic=dynamic_quant_config,
)

model = GPTQModel.load(MODEL_ID, quant_config, device=MODEL_DEVICE)
_patch_forward_for_precomputed_inputs(model)
model.prepare_dataset = MethodType(_prepare_mixed_calibration_dataset, model)
model.quantize(calibration_dataset, batch_size=BATCH_SIZE)
model.save(QUANT_PATH)
