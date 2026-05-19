import os
import base64
import gc
from io import BytesIO
from types import MethodType
from typing import Any, Dict, Iterator, List, Optional

import torch
from datasets import load_dataset
from gptqmodel import GPTQModel, QuantizeConfig
from PIL import Image


def _env_bool(name: str, default: str) -> bool:
    return os.environ.get(name, default).strip().lower() not in {"0", "false", "no", "off"}


def _split_csv(value: str) -> List[str]:
    return [item.strip().lower() for item in value.split(",") if item.strip()]


def _normalize_language_pattern(pattern: List[str], default: List[str]) -> List[str]:
    supported_languages = {"zh", "en"}
    normalized = [lang for lang in pattern if lang in supported_languages]
    return normalized or default


MODEL_ID = os.environ.get("MODEL_ID", "../../Qwen/Qwen3.5-0.8B/")
QUANT_PATH = os.environ.get("QUANT_PATH", "../../Qwen/Qwen3.5-0.8B-GPTQ-Int4-ZH")

TARGET_LANGUAGE = os.environ.get("TARGET_LANGUAGE", "zh").strip().lower()
PREFER_CHINESE = TARGET_LANGUAGE in {"zh", "cn", "chinese", "中文"}

WIKI_DATASET_ID = os.environ.get("WIKI_DATASET_ID", "wikimedia/wikipedia")
WIKI_DATA_SOURCE = os.environ.get("WIKI_DATA_SOURCE", "modelscope").strip().lower()
WIKI_EN_CONFIG = os.environ.get("WIKI_EN_CONFIG", "20231101.en")
WIKI_ZH_CONFIG = os.environ.get("WIKI_ZH_CONFIG", "20231101.zh")
DATASETS_CACHE_DIR = os.environ.get("DATASETS_CACHE_DIR", "").strip() or None
MODELSCOPE_DATASET_CACHE_DIR = os.environ.get("MODELSCOPE_DATASET_CACHE_DIR", "").strip() or None

COCO_CAPTION_DATA_FILES = os.environ.get("COCO_CAPTION_DATA_FILES", "val-00001-of-00013.parquet")
COCO_CAPTION_SEED = int(os.environ.get("COCO_CAPTION_SEED", "42"))
COCO_IMAGE_FIELD = os.environ.get("COCO_IMAGE_FIELD", "image").strip()
COCO_EN_ANSWER_FIELD = os.environ.get("COCO_EN_ANSWER_FIELD", "answer").strip()

VQA_DATA_SOURCE = os.environ.get("VQA_DATA_SOURCE", "modelscope").strip().lower()
VQA_DATASET_ID = os.environ.get("VQA_DATASET_ID", "moonshotai/WorldVQA").strip()
VQA_CONFIG = os.environ.get("VQA_CONFIG", "").strip()
VQA_DATA_FILES = os.environ.get("VQA_DATA_FILES", "").strip()
VQA_SPLIT = os.environ.get("VQA_SPLIT", "train").strip()
VQA_SEED = int(os.environ.get("VQA_SEED", "43"))
VQA_IMAGE_FIELD = os.environ.get("VQA_IMAGE_FIELD", "image").strip()
VQA_QUESTION_FIELD = os.environ.get("VQA_QUESTION_FIELD", "question").strip()
VQA_ANSWER_FIELD = os.environ.get("VQA_ANSWER_FIELD", "answer").strip()
VQA_LANG_FIELD = os.environ.get("VQA_LANG_FIELD", "language").strip()
VQA_ZH_QUESTION_FIELD = os.environ.get("VQA_ZH_QUESTION_FIELD", "").strip()
VQA_ZH_ANSWER_FIELD = os.environ.get("VQA_ZH_ANSWER_FIELD", "").strip()

NUM_CALIB = int(os.environ.get("NUM_CALIB", "1024"))
MIN_TEXT_CHARS = int(os.environ.get("MIN_TEXT_CHARS", "64"))
MAX_TEXT_CHARS = int(os.environ.get("MAX_TEXT_CHARS", "2048"))
SHUFFLE_BUFFER = int(os.environ.get("SHUFFLE_BUFFER", "10000"))
BATCH_SIZE = int(os.environ.get("BATCH_SIZE", "1"))

DEFAULT_NUM_VQA_CALIB = NUM_CALIB * 5 // 8
NUM_VQA_CALIB = int(os.environ.get("NUM_VQA_CALIB", str(DEFAULT_NUM_VQA_CALIB)))
DEFAULT_NUM_COCO_CALIB = NUM_CALIB // 4
NUM_COCO_CALIB = int(os.environ.get("NUM_COCO_CALIB", str(DEFAULT_NUM_COCO_CALIB)))
DEFAULT_NUM_LANGUAGE_CALIB = min(max(32, NUM_CALIB // 16), max(1, NUM_CALIB // 8))
NUM_LANGUAGE_CALIB = int(os.environ.get("NUM_LANGUAGE_CALIB", str(DEFAULT_NUM_LANGUAGE_CALIB)))
COCO_CAPTION_STRATEGY = os.environ.get("COCO_CAPTION_STRATEGY", "grounded").strip().lower()
ADD_GENERATION_PROMPT = _env_bool("ADD_GENERATION_PROMPT", "1")
CALIBRATION_ENABLE_THINKING = _env_bool("CALIBRATION_ENABLE_THINKING", "0")
NO_THINKING_CALIB_RATIO = float(os.environ.get("NO_THINKING_CALIB_RATIO", "0.75"))
TEXT_LANGUAGE_PATTERN = _normalize_language_pattern(
    _split_csv(os.environ.get("TEXT_LANGUAGE_PATTERN", "zh,zh,en" if PREFER_CHINESE else "zh,en")),
    ["zh", "zh", "en"] if PREFER_CHINESE else ["zh", "en"],
)
COCO_PROMPT_PATTERN = _normalize_language_pattern(
    _split_csv(os.environ.get("COCO_PROMPT_PATTERN", "zh,zh,en" if PREFER_CHINESE else "zh,en")),
    ["zh", "zh", "en"] if PREFER_CHINESE else ["zh", "en"],
)
VQA_PROMPT_PATTERN = _normalize_language_pattern(
    _split_csv(os.environ.get("VQA_PROMPT_PATTERN", "zh,zh,en" if PREFER_CHINESE else "zh,en")),
    ["zh", "zh", "en"] if PREFER_CHINESE else ["zh", "en"],
)
LANGUAGE_FOLLOWING_PATTERN = _normalize_language_pattern(
    _split_csv(os.environ.get("LANGUAGE_FOLLOWING_PATTERN", "zh,zh,en" if PREFER_CHINESE else "zh,en")),
    ["zh", "zh", "en"] if PREFER_CHINESE else ["zh", "en"],
)
TEXT_AS_ASSISTANT = _env_bool("TEXT_AS_ASSISTANT", "1")
TEXT_PROMPT_CHARS = int(os.environ.get("TEXT_PROMPT_CHARS", "384"))
REQUIRE_CJK_FOR_ZH = _env_bool("REQUIRE_CJK_FOR_ZH", "1")
MIN_CJK_RATIO = float(os.environ.get("MIN_CJK_RATIO", "0.20"))
COCO_TRANSLATION_MODEL_ID = os.environ.get("COCO_TRANSLATION_MODEL_ID", "tencent/Hunyuan-MT-7B")
COCO_ZH_ANSWER_FIELD = os.environ.get("COCO_ZH_ANSWER_FIELD", "").strip()
COCO_TRANSLATION_DEVICE = os.environ.get("COCO_TRANSLATION_DEVICE", "cuda:1")
COCO_TRANSLATION_BATCH_SIZE = int(os.environ.get("COCO_TRANSLATION_BATCH_SIZE", "4"))
COCO_TRANSLATION_MAX_NEW_TOKENS = int(os.environ.get("COCO_TRANSLATION_MAX_NEW_TOKENS", "128"))
COCO_TRANSLATION_TEMPERATURE = float(os.environ.get("COCO_TRANSLATION_TEMPERATURE", "0.7"))
COCO_TRANSLATION_TOP_P = float(os.environ.get("COCO_TRANSLATION_TOP_P", "0.6"))
COCO_TRANSLATION_TOP_K = int(os.environ.get("COCO_TRANSLATION_TOP_K", "20"))
COCO_TRANSLATION_REPETITION_PENALTY = float(os.environ.get("COCO_TRANSLATION_REPETITION_PENALTY", "1.05"))
COCO_TRANSLATION_ADD_GENERATION_PROMPT = _env_bool("COCO_TRANSLATION_ADD_GENERATION_PROMPT", "0")
TRANSLATION_PROMPT_TEMPLATE = os.environ.get(
    "TRANSLATION_PROMPT_TEMPLATE",
    "Translate the following segment into Chinese, without additional explanation.\n\n{}",
)
COCO_TRANSLATION_PROMPT_TEMPLATE = os.environ.get("COCO_TRANSLATION_PROMPT_TEMPLATE", TRANSLATION_PROMPT_TEMPLATE)
VQA_TRANSLATION_PROMPT_TEMPLATE = os.environ.get("VQA_TRANSLATION_PROMPT_TEMPLATE", TRANSLATION_PROMPT_TEMPLATE)
STRICT_CALIBRATION_CACHE = _env_bool("STRICT_CALIBRATION_CACHE", "1")
OVERWRITE_CALIBRATION_CACHE = _env_bool("OVERWRITE_CALIBRATION_CACHE", "0")
MAX_TEXT_SCAN_MULTIPLIER = int(os.environ.get("MAX_TEXT_SCAN_MULTIPLIER", "200"))

MODEL_DEVICE = os.environ.get("MODEL_DEVICE", "cuda:0")
CALIBRATION_DEVICE = os.environ.get("CALIBRATION_DEVICE", MODEL_DEVICE)
PRECOMPUTE_INPUTS_DEVICE = os.environ.get("PRECOMPUTE_INPUTS_DEVICE", CALIBRATION_DEVICE)
RELEASE_PRECOMPUTE_MODULES = _env_bool("RELEASE_PRECOMPUTE_MODULES", "1")
RELEASE_TRANSLATION_MODEL = _env_bool("RELEASE_TRANSLATION_MODEL", "1")
CALIBRATION_CACHE_PATH = os.environ.get("CALIBRATION_CACHE_PATH", "").strip()
ENABLE_DYNAMIC_QUANT_CONFIG = _env_bool("ENABLE_DYNAMIC_QUANT_CONFIG", "1")
SKIP_LINEAR_ATTN_FIRST_N = int(os.environ.get("SKIP_LINEAR_ATTN_FIRST_N", "4"))

_PREPARED_CALIBRATION_DATASET: List[Dict[str, torch.Tensor]] | None = None
_CALIBRATION_CACHE_LOAD_ATTEMPTED = False
_LOADED_CALIBRATION_CACHE: List[Dict[str, torch.Tensor]] | None = None
_CALIBRATION_CACHE_INVALID = False
_CAPTION_TRANSLATION_TOKENIZER = None
_CAPTION_TRANSLATION_MODEL = None

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

COCO_PROMPTS_ZH = [
    "描述图片内容。请用中文回答，只描述清楚可见的主体、场景和动作，不要猜测人物关系或情绪。",
    "请用中文回答：这张图片里有什么？不确定的细节请说明不确定。",
    "请只根据图片中清楚可见的内容，用一两句话中文客观描述。不要猜测人物关系、地点、时间、情绪或看不见的细节。",
    "请用中文简洁说明图片里的主要物体、场景和动作；不确定的内容不要编造。",
    "请描述画面中可以确认的主体和背景，并保持全程中文回答。如果无法从图像确定，请不要推断。",
]

COCO_PROMPTS_EN = [
    "Describe the image. Answer in English and only mention clearly visible subjects, scene, and actions.",
    "What is in this image? Answer in English. If a detail is uncertain, say it is uncertain.",
    "Describe only what is clearly visible in the image in one or two concise sentences. Do not infer relationships, location, time, emotions, or hidden details.",
    "Give a concise, grounded caption for the image. Avoid adding details that are not visually evident.",
    "What can be directly observed in this image? Answer conservatively and avoid speculation.",
]

TEXT_CONTINUE_PROMPTS = {
    "zh": "请继续下面的中文内容，保持中文表达，不要切换到英文：\n\n{}",
    "en": "Continue the following English text in English:\n\n{}",
}

LANGUAGE_FOLLOWING_EXAMPLES = {
    "zh": [
        (
            "请用中文回答：图片理解回答时应该遵守什么原则？",
            "应使用中文回答，只描述图中清楚可见的主体、动作和背景，不猜测人物关系、情绪或具体时间。",
        ),
        (
            "如果用户用中文提问，但图中细节不确定，应该怎样回答？",
            "应继续用中文回答，并明确说明不确定的信息，不要把无法确认的内容说成事实。",
        ),
        (
            "请用中文简洁说明：如何避免看图说话时编造细节？",
            "先描述可见事实，再说明无法确认的部分；不要凭经验推断地点、身份、关系、情绪或隐藏物体。",
        ),
        (
            "请用中文回答：用户要求描述图片内容时，输出应该长还是短？",
            "输出应简洁、客观，优先覆盖主要主体、场景和动作，避免无依据的扩写。",
        ),
    ],
    "en": [
        (
            "Answer in English: what rule should image descriptions follow?",
            "Answer in English and describe only clearly visible subjects, actions, and background without guessing relationships, emotions, or exact time.",
        ),
        (
            "If the user asks in English and a visual detail is uncertain, how should you answer?",
            "Keep the answer in English, say that the detail is uncertain, and do not present unsupported details as facts.",
        ),
        (
            "Answer briefly in English: how can hallucinated image details be avoided?",
            "State visible facts first, mention uncertainty when needed, and avoid inferring locations, identities, relationships, emotions, or hidden objects.",
        ),
        (
            "In English, should an image caption be concise or speculative?",
            "It should be concise and grounded in visible evidence, covering the main subjects, scene, and actions without unsupported expansion.",
        ),
    ],
}

SPECULATIVE_CAPTION_WORDS = {
    "about to",
    "beautiful",
    "celebrating",
    "enjoying",
    "family",
    "friend",
    "friends",
    "happy",
    "lovely",
    "party",
    "probably",
    "relationship",
    "seems",
    "trying to",
    "waiting",
}


def _caption_speculation_score(caption: str) -> int:
    lowered = caption.lower()
    return sum(1 for word in SPECULATIVE_CAPTION_WORDS if word in lowered)


def _load_caption_translation_model():
    global _CAPTION_TRANSLATION_MODEL, _CAPTION_TRANSLATION_TOKENIZER

    if _CAPTION_TRANSLATION_MODEL is not None and _CAPTION_TRANSLATION_TOKENIZER is not None:
        return _CAPTION_TRANSLATION_TOKENIZER, _CAPTION_TRANSLATION_MODEL

    if COCO_TRANSLATION_MODEL_ID.strip().lower() in {"", "0", "false", "no", "none", "off"}:
        raise RuntimeError(
            "Chinese COCO prompts need Chinese assistant captions. Set COCO_TRANSLATION_MODEL_ID "
            "to an English-to-Chinese translation model, or provide a Chinese caption field via "
            "COCO_ZH_ANSWER_FIELD."
        )

    try:
        from transformers import AutoModelForCausalLM, AutoTokenizer
    except ImportError as exc:
        raise RuntimeError(
            "COCO caption translation requires transformers AutoTokenizer and AutoModelForCausalLM."
        ) from exc

    print(f"loading COCO caption translation model: {COCO_TRANSLATION_MODEL_ID}")
    tokenizer = AutoTokenizer.from_pretrained(COCO_TRANSLATION_MODEL_ID)
    device = torch.device(COCO_TRANSLATION_DEVICE)
    model = AutoModelForCausalLM.from_pretrained(
        COCO_TRANSLATION_MODEL_ID,
        torch_dtype=torch.bfloat16 if device.type == "cuda" else torch.float32,
        trust_remote_code=True,
    )
    model.to(device)
    model.eval()

    _CAPTION_TRANSLATION_TOKENIZER = tokenizer
    _CAPTION_TRANSLATION_MODEL = model
    return tokenizer, model


def _release_caption_translation_model() -> None:
    global _CAPTION_TRANSLATION_MODEL, _CAPTION_TRANSLATION_TOKENIZER

    if not RELEASE_TRANSLATION_MODEL:
        return
    if _CAPTION_TRANSLATION_MODEL is None and _CAPTION_TRANSLATION_TOKENIZER is None:
        return

    model = _CAPTION_TRANSLATION_MODEL
    tokenizer = _CAPTION_TRANSLATION_TOKENIZER
    device = None
    if model is not None:
        try:
            device = next(model.parameters()).device
        except StopIteration:
            device = None

    print(f"releasing COCO/VQA translation model from {device or 'memory'}")
    _CAPTION_TRANSLATION_MODEL = None
    _CAPTION_TRANSLATION_TOKENIZER = None
    del model
    del tokenizer
    gc.collect()

    if device is not None and device.type == "cuda":
        with torch.cuda.device(device):
            torch.cuda.empty_cache()
            try:
                torch.cuda.ipc_collect()
            except RuntimeError:
                pass


def _build_translation_prompt(text: str, prompt_template: str) -> str:
    return prompt_template.format(text)


def _calibration_chat_template_kwargs(enable_thinking: Optional[bool] = None) -> Dict[str, Any]:
    if enable_thinking is None:
        enable_thinking = CALIBRATION_ENABLE_THINKING
    return {"enable_thinking": enable_thinking}


def _build_translation_input(tokenizer, text: str, prompt_template: str) -> Dict[str, torch.Tensor]:
    prompt = _build_translation_prompt(text, prompt_template)
    messages = [{"role": "user", "content": prompt}]

    if hasattr(tokenizer, "apply_chat_template") and tokenizer.chat_template:
        encoded = tokenizer.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=COCO_TRANSLATION_ADD_GENERATION_PROMPT,
            return_dict=True,
            return_tensors="pt",
            **_calibration_chat_template_kwargs(),
        )
    else:
        encoded = tokenizer(prompt, return_tensors="pt")

    return dict(encoded)


def _translate_texts_to_zh(texts: List[str], prompt_template: str) -> List[str]:
    normalized_texts = [text.strip() for text in texts]
    if not normalized_texts:
        return []

    tokenizer, model = _load_caption_translation_model()
    device = next(model.parameters()).device
    translations: List[str] = []

    for start in range(0, len(normalized_texts), COCO_TRANSLATION_BATCH_SIZE):
        batch = normalized_texts[start : start + COCO_TRANSLATION_BATCH_SIZE]
        for text in batch:
            encoded = _build_translation_input(tokenizer, text, prompt_template)
            encoded = {key: value.to(device) for key, value in encoded.items()}
            input_len = encoded["input_ids"].shape[-1]
            with torch.no_grad():
                generated = model.generate(
                    **encoded,
                    max_new_tokens=COCO_TRANSLATION_MAX_NEW_TOKENS,
                    do_sample=COCO_TRANSLATION_TEMPERATURE > 0,
                    temperature=COCO_TRANSLATION_TEMPERATURE,
                    top_p=COCO_TRANSLATION_TOP_P,
                    top_k=COCO_TRANSLATION_TOP_K,
                    repetition_penalty=COCO_TRANSLATION_REPETITION_PENALTY,
                )
            output_ids = generated[0, input_len:]
            translations.append(tokenizer.decode(output_ids, skip_special_tokens=True).strip())

    return translations


def _translate_captions_to_zh(captions: List[str]) -> List[str]:
    return _translate_texts_to_zh(captions, COCO_TRANSLATION_PROMPT_TEMPLATE)


def _translate_vqa_texts_to_zh(texts: List[str]) -> List[str]:
    return _translate_texts_to_zh(texts, VQA_TRANSLATION_PROMPT_TEMPLATE)


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


def _calibration_cache_metadata() -> Dict[str, Any]:
    return {
        "cache_version": 7,
        "model_id": MODEL_ID,
        "target_language": TARGET_LANGUAGE,
        "num_calib": NUM_CALIB,
        "num_vqa_calib": NUM_VQA_CALIB,
        "num_coco_calib": NUM_COCO_CALIB,
        "num_language_calib": NUM_LANGUAGE_CALIB,
        "min_text_chars": MIN_TEXT_CHARS,
        "max_text_chars": MAX_TEXT_CHARS,
        "add_generation_prompt": ADD_GENERATION_PROMPT,
        "calibration_enable_thinking": CALIBRATION_ENABLE_THINKING,
        "no_thinking_calib_ratio": NO_THINKING_CALIB_RATIO,
        "text_language_pattern": TEXT_LANGUAGE_PATTERN,
        "coco_prompt_pattern": COCO_PROMPT_PATTERN,
        "vqa_prompt_pattern": VQA_PROMPT_PATTERN,
        "language_following_pattern": LANGUAGE_FOLLOWING_PATTERN,
        "text_as_assistant": TEXT_AS_ASSISTANT,
        "text_prompt_chars": TEXT_PROMPT_CHARS,
        "require_cjk_for_zh": REQUIRE_CJK_FOR_ZH,
        "min_cjk_ratio": MIN_CJK_RATIO,
        "coco_translation_model_id": COCO_TRANSLATION_MODEL_ID,
        "coco_translation_device": COCO_TRANSLATION_DEVICE,
        "coco_zh_answer_field": COCO_ZH_ANSWER_FIELD,
        "coco_translation_batch_size": COCO_TRANSLATION_BATCH_SIZE,
        "coco_translation_max_new_tokens": COCO_TRANSLATION_MAX_NEW_TOKENS,
        "coco_translation_temperature": COCO_TRANSLATION_TEMPERATURE,
        "coco_translation_top_p": COCO_TRANSLATION_TOP_P,
        "coco_translation_top_k": COCO_TRANSLATION_TOP_K,
        "coco_translation_repetition_penalty": COCO_TRANSLATION_REPETITION_PENALTY,
        "coco_translation_add_generation_prompt": COCO_TRANSLATION_ADD_GENERATION_PROMPT,
        "translation_prompt_template": TRANSLATION_PROMPT_TEMPLATE,
        "coco_translation_prompt_template": COCO_TRANSLATION_PROMPT_TEMPLATE,
        "vqa_translation_prompt_template": VQA_TRANSLATION_PROMPT_TEMPLATE,
        "wiki_data_source": WIKI_DATA_SOURCE,
        "wiki_dataset_id": WIKI_DATASET_ID,
        "wiki_en_config": WIKI_EN_CONFIG,
        "wiki_zh_config": WIKI_ZH_CONFIG,
        "datasets_cache_dir": DATASETS_CACHE_DIR,
        "coco_caption_data_files": COCO_CAPTION_DATA_FILES,
        "coco_caption_seed": COCO_CAPTION_SEED,
        "coco_image_field": COCO_IMAGE_FIELD,
        "coco_en_answer_field": COCO_EN_ANSWER_FIELD,
        "coco_caption_strategy": COCO_CAPTION_STRATEGY,
        "vqa_data_source": VQA_DATA_SOURCE,
        "vqa_dataset_id": VQA_DATASET_ID,
        "vqa_config": VQA_CONFIG,
        "vqa_data_files": VQA_DATA_FILES,
        "vqa_split": VQA_SPLIT,
        "vqa_seed": VQA_SEED,
        "vqa_image_field": VQA_IMAGE_FIELD,
        "vqa_question_field": VQA_QUESTION_FIELD,
        "vqa_answer_field": VQA_ANSWER_FIELD,
        "vqa_lang_field": VQA_LANG_FIELD,
        "vqa_zh_question_field": VQA_ZH_QUESTION_FIELD,
        "vqa_zh_answer_field": VQA_ZH_ANSWER_FIELD,
    }


def _cache_mismatch_reasons(metadata: Dict[str, Any]) -> List[str]:
    expected = _calibration_cache_metadata()
    reasons = []
    for key, expected_value in expected.items():
        if metadata.get(key) != expected_value:
            reasons.append(f"{key}: cached={metadata.get(key)!r}, expected={expected_value!r}")
    return reasons


def _load_calibration_cache() -> List[Dict[str, torch.Tensor]] | None:
    global _CALIBRATION_CACHE_INVALID, _CALIBRATION_CACHE_LOAD_ATTEMPTED, _LOADED_CALIBRATION_CACHE

    if _CALIBRATION_CACHE_LOAD_ATTEMPTED:
        return _LOADED_CALIBRATION_CACHE

    _CALIBRATION_CACHE_LOAD_ATTEMPTED = True
    if not _has_calibration_cache():
        return None

    if OVERWRITE_CALIBRATION_CACHE:
        _CALIBRATION_CACHE_INVALID = True
        print(f"OVERWRITE_CALIBRATION_CACHE=1; rebuilding calibration cache: {CALIBRATION_CACHE_PATH}")
        return None

    print(f"loading prepared calibration cache: {CALIBRATION_CACHE_PATH}")
    try:
        cache = torch.load(CALIBRATION_CACHE_PATH, map_location="cpu", weights_only=False)
    except TypeError:
        cache = torch.load(CALIBRATION_CACHE_PATH, map_location="cpu")

    metadata: Dict[str, Any] = {}
    if isinstance(cache, dict) and "encoded_batches" in cache:
        metadata = dict(cache.get("metadata") or {})
        encoded_batches = cache["encoded_batches"]
    else:
        encoded_batches = cache

    if not isinstance(encoded_batches, list) or not encoded_batches:
        raise ValueError(f"invalid calibration cache: {CALIBRATION_CACHE_PATH}")

    mismatch_reasons = _cache_mismatch_reasons(metadata)
    if STRICT_CALIBRATION_CACHE and mismatch_reasons:
        _CALIBRATION_CACHE_INVALID = True
        print("calibration cache metadata mismatch; rebuilding cache instead of reusing it")
        for reason in mismatch_reasons[:8]:
            print(f"  - {reason}")
        if len(mismatch_reasons) > 8:
            print(f"  - ... {len(mismatch_reasons) - 8} more differences")
        return None

    print(f"loaded {len(encoded_batches)} prepared calibration batches from cache")
    _LOADED_CALIBRATION_CACHE = encoded_batches
    return encoded_batches


def _save_calibration_cache(encoded_batches: List[Dict[str, torch.Tensor]]) -> None:
    if not CALIBRATION_CACHE_PATH:
        return

    if _has_calibration_cache() and not (OVERWRITE_CALIBRATION_CACHE or _CALIBRATION_CACHE_INVALID):
        print(
            f"not overwriting existing calibration cache: {CALIBRATION_CACHE_PATH}; "
            "set OVERWRITE_CALIBRATION_CACHE=1 to replace it"
        )
        return

    cache_dir = os.path.dirname(CALIBRATION_CACHE_PATH)
    if cache_dir:
        os.makedirs(cache_dir, exist_ok=True)

    tmp_path = f"{CALIBRATION_CACHE_PATH}.tmp"
    cache = {
        "metadata": _calibration_cache_metadata(),
        "encoded_batches": encoded_batches,
    }
    torch.save(cache, tmp_path)
    os.replace(tmp_path, CALIBRATION_CACHE_PATH)
    print(f"saved prepared calibration cache: {CALIBRATION_CACHE_PATH}")


def _language_counts(samples: List[Dict[str, Any]]) -> str:
    counts: Dict[str, int] = {}
    for sample in samples:
        lang = str(sample.get("lang", "unknown"))
        counts[lang] = counts.get(lang, 0) + 1
    return ", ".join(f"{lang}={count}" for lang, count in sorted(counts.items()))


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


def _load_wikipedia_from_modelscope(config_name: str):
    try:
        from modelscope.msdatasets import MsDataset
    except ImportError as exc:
        raise RuntimeError(
            "WIKI_DATA_SOURCE=modelscope requires `modelscope` to be installed. "
            "Install it or set WIKI_DATA_SOURCE=huggingface."
        ) from exc

    load_kwargs = {
        "subset_name": config_name,
        "split": "train",
        "use_streaming": True,
    }
    if MODELSCOPE_DATASET_CACHE_DIR is not None:
        load_kwargs["cache_dir"] = MODELSCOPE_DATASET_CACHE_DIR

    return MsDataset.load(WIKI_DATASET_ID, **load_kwargs)


def _load_wikipedia_from_huggingface(config_name: str):
    return load_dataset(
        WIKI_DATASET_ID,
        config_name,
        split="train",
        streaming=True,
        cache_dir=DATASETS_CACHE_DIR,
    )


def _load_wikipedia_stream(config_name: str, seed: int):
    if WIKI_DATA_SOURCE == "modelscope":
        dataset = _load_wikipedia_from_modelscope(config_name)
    elif WIKI_DATA_SOURCE in {"hf", "huggingface"}:
        dataset = _load_wikipedia_from_huggingface(config_name)
    else:
        raise ValueError("WIKI_DATA_SOURCE must be `modelscope`, `huggingface`, or `hf`.")

    return dataset.shuffle(seed=seed, buffer_size=SHUFFLE_BUFFER)


def _get_text_field(row: Dict[str, Any]) -> str:
    text = row.get("text", "")
    if isinstance(text, (list, tuple)):
        return str(next((item for item in text if str(item).strip()), ""))
    return str(text)


def _normalize_text(text: str) -> str:
    compact = " ".join(text.split()).strip()
    if MAX_TEXT_CHARS > 0:
        compact = compact[:MAX_TEXT_CHARS]
    return compact


def _cjk_ratio(text: str) -> float:
    non_space_chars = [char for char in text if not char.isspace()]
    if not non_space_chars:
        return 0.0
    cjk_chars = sum("\u4e00" <= char <= "\u9fff" for char in non_space_chars)
    return cjk_chars / len(non_space_chars)


def _is_valid_text_for_language(text: str, lang: str) -> bool:
    if len(text) < MIN_TEXT_CHARS:
        return False
    if lang == "zh" and REQUIRE_CJK_FOR_ZH:
        return _cjk_ratio(text) >= MIN_CJK_RATIO
    return True


def _next_text(streams: Dict[str, Iterator[Dict[str, str]]], lang: str) -> str:
    max_attempts = max(num * MAX_TEXT_SCAN_MULTIPLIER for num in (NUM_CALIB, 1))
    for _ in range(max_attempts):
        try:
            row = next(streams[lang])
        except StopIteration as exc:
            raise RuntimeError(f"Wikipedia stream for `{lang}` exhausted unexpectedly.") from exc

        text = _normalize_text(_get_text_field(row))
        if _is_valid_text_for_language(text, lang):
            return text

    raise RuntimeError(
        f"failed to collect a valid `{lang}` Wikipedia sample after {max_attempts} rows; "
        "check WIKI_ZH_CONFIG/WIKI_EN_CONFIG or lower MIN_TEXT_CHARS/MIN_CJK_RATIO"
    )


def _split_prompt_and_answer(text: str, lang: str) -> tuple[str, str]:
    if not TEXT_AS_ASSISTANT:
        return text, ""

    split_at = min(max(TEXT_PROMPT_CHARS, MIN_TEXT_CHARS), max(len(text) // 2, 1))
    prompt_text = text[:split_at].rstrip()
    answer_text = text[split_at:].lstrip()
    if len(answer_text) < max(16, MIN_TEXT_CHARS // 4):
        return text, ""
    prompt_template = TEXT_CONTINUE_PROMPTS.get(lang, TEXT_CONTINUE_PROMPTS["en"])
    return prompt_template.format(prompt_text), answer_text


def _build_text_messages(text: str, lang: str) -> List[Dict[str, Any]]:
    prompt_text, answer_text = _split_prompt_and_answer(text, lang)
    messages: List[Dict[str, Any]] = [
        {
            "role": "user",
            "content": [{"type": "text", "text": prompt_text}],
        }
    ]
    if answer_text:
        messages.append(
            {
                "role": "assistant",
                "content": [{"type": "text", "text": answer_text}],
            }
        )
    return messages


def _build_text_calibration_dataset(num_samples: int) -> List[Dict[str, Any]]:
    if num_samples <= 0:
        return []

    zh_stream: Iterator[Dict[str, str]] = iter(_load_wikipedia_stream(WIKI_ZH_CONFIG, seed=42))
    en_stream: Iterator[Dict[str, str]] = iter(_load_wikipedia_stream(WIKI_EN_CONFIG, seed=43))

    streams = {"zh": zh_stream, "en": en_stream}
    calibration_dataset: List[Dict[str, Any]] = []

    while len(calibration_dataset) < num_samples:
        lang = TEXT_LANGUAGE_PATTERN[len(calibration_dataset) % len(TEXT_LANGUAGE_PATTERN)]
        text = _next_text(streams, lang)
        calibration_dataset.append(
            {
                "messages": _build_text_messages(text, lang),
                "source": "wikipedia_text",
                "lang": lang,
            }
        )

        if len(calibration_dataset) % 64 == 0:
            print(
                f"collected {len(calibration_dataset)}/{num_samples} text samples "
                f"from Wikipedia via {WIKI_DATA_SOURCE} ({_language_counts(calibration_dataset)})"
            )

    return calibration_dataset


def _resolve_data_files(data_files_spec: str, env_name: str):
    data_files = [path.strip() for path in data_files_spec.split(",") if path.strip()]
    if not data_files:
        raise ValueError(f"{env_name} must contain at least one dataset file path.")
    return data_files[0] if len(data_files) == 1 else data_files


def _resolve_coco_data_files(data_files_spec: str):
    return _resolve_data_files(data_files_spec, "COCO_CAPTION_DATA_FILES")


def _load_vqa_dataset():
    if VQA_DATA_FILES:
        data_files = _resolve_data_files(VQA_DATA_FILES, "VQA_DATA_FILES")
        extension = os.path.splitext(data_files[0] if isinstance(data_files, list) else data_files)[1].lower()
        if extension == ".jsonl":
            extension = ".json"
        dataset_type = extension.lstrip(".") or "parquet"
        return load_dataset(
            dataset_type,
            data_files=data_files,
            split=VQA_SPLIT,
            cache_dir=DATASETS_CACHE_DIR,
        ).shuffle(seed=VQA_SEED)

    if VQA_DATA_SOURCE == "modelscope":
        try:
            from modelscope.msdatasets import MsDataset
        except ImportError as exc:
            raise RuntimeError(
                "VQA_DATA_SOURCE=modelscope requires `modelscope` to be installed. "
                "Install it or set VQA_DATA_SOURCE=huggingface."
            ) from exc

        load_kwargs = {"split": VQA_SPLIT}
        if VQA_CONFIG:
            load_kwargs["subset_name"] = VQA_CONFIG
        if MODELSCOPE_DATASET_CACHE_DIR is not None:
            load_kwargs["cache_dir"] = MODELSCOPE_DATASET_CACHE_DIR
        return MsDataset.load(VQA_DATASET_ID, **load_kwargs)

    if VQA_DATA_SOURCE in {"hf", "huggingface"}:
        load_args = [VQA_DATASET_ID]
        if VQA_CONFIG:
            load_args.append(VQA_CONFIG)
        return load_dataset(*load_args, split=VQA_SPLIT, cache_dir=DATASETS_CACHE_DIR).shuffle(seed=VQA_SEED)

    raise ValueError("VQA_DATA_SOURCE must be `modelscope`, `huggingface`, or `hf`.")


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
        if COCO_CAPTION_STRATEGY == "grounded":
            return min(captions, key=lambda caption: (_caption_speculation_score(caption), len(caption)))
        return min(captions, key=len)
    return str(answer).strip()


def _get_caption_from_row(row: Dict[str, Any], field_name: str, sample_index: int) -> str:
    if not field_name:
        return ""
    return _select_caption(row.get(field_name, ""), sample_index)


def _normalize_answer_field(answer: Any, sample_index: int) -> str:
    if isinstance(answer, list):
        values = [str(item).strip() for item in answer if str(item).strip()]
        if not values:
            return ""
        return values[sample_index % len(values)]
    if isinstance(answer, dict):
        for key in ("answer", "text", "label", "value"):
            if key in answer:
                return _normalize_answer_field(answer[key], sample_index)
        values = [str(value).strip() for value in answer.values() if str(value).strip()]
        return values[0] if values else ""
    return str(answer).strip()


def _get_text_from_row(row: Dict[str, Any], field_name: str, sample_index: int) -> str:
    if not field_name:
        return ""
    return _normalize_answer_field(row.get(field_name, ""), sample_index)


def _get_first_existing(row: Dict[str, Any], field_names: List[str]) -> Any:
    for field_name in field_names:
        if field_name and field_name in row and row[field_name] is not None:
            return row[field_name]
    return None


def _decode_image_value(image_value: Any) -> Any:
    if not isinstance(image_value, str):
        return image_value

    image_text = image_value.strip()
    if not image_text:
        return None

    if os.path.exists(image_text):
        return Image.open(image_text).convert("RGB")

    if image_text.startswith("data:image") and "," in image_text:
        image_text = image_text.split(",", 1)[1]

    try:
        return Image.open(BytesIO(base64.b64decode(image_text))).convert("RGB")
    except Exception:
        return image_value


def _normalize_dataset_lang(lang: Any) -> str:
    normalized = str(lang or "").strip().lower()
    if normalized in {"zh", "cn", "chinese", "中文", "汉语", "mandarin", "cmn"}:
        return "zh"
    if normalized in {"en", "eng", "english"}:
        return "en"
    return ""


def _build_coco_messages(image: Any, question: str, answer: str) -> List[Dict[str, Any]]:
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": image},
                {"type": "text", "text": question},
            ],
        }
    ]
    if answer:
        messages.append(
            {
                "role": "assistant",
                "content": [{"type": "text", "text": answer}],
            }
        )
    return messages


def _build_vqa_messages(image: Any, question: str, answer: str) -> List[Dict[str, Any]]:
    return _build_coco_messages(image, question, answer)


def _sample_lang(pattern: List[str], sample_index: int) -> str:
    return pattern[sample_index % len(pattern)]


def _build_language_following_dataset(num_samples: int) -> List[Dict[str, Any]]:
    if num_samples <= 0:
        return []

    calibration_dataset: List[Dict[str, Any]] = []
    while len(calibration_dataset) < num_samples:
        sample_index = len(calibration_dataset)
        lang = _sample_lang(LANGUAGE_FOLLOWING_PATTERN, sample_index)
        examples = LANGUAGE_FOLLOWING_EXAMPLES[lang]
        question, answer = examples[sample_index % len(examples)]
        calibration_dataset.append(
            {
                "messages": [
                    {
                        "role": "user",
                        "content": [{"type": "text", "text": question}],
                    },
                    {
                        "role": "assistant",
                        "content": [{"type": "text", "text": answer}],
                    },
                ],
                "source": "language_following",
                "lang": lang,
            }
        )

    return calibration_dataset


def _coco_prompt(sample_index: int) -> tuple[str, str]:
    lang = _sample_lang(COCO_PROMPT_PATTERN, sample_index)
    prompts = COCO_PROMPTS_ZH if lang == "zh" else COCO_PROMPTS_EN
    return prompts[sample_index % len(prompts)], lang


def _vqa_lang(sample_index: int) -> str:
    return _sample_lang(VQA_PROMPT_PATTERN, sample_index)


def _build_vqa_calibration_dataset(num_samples: int) -> List[Dict[str, Any]]:
    if num_samples <= 0:
        return []

    dataset = _load_vqa_dataset()
    pending_samples: List[Dict[str, Any]] = []
    for idx, row in enumerate(dataset):
        if len(pending_samples) >= num_samples:
            break

        image = _decode_image_value(
            _get_first_existing(row, [VQA_IMAGE_FIELD, "image", "image_base64", "img", "picture"])
        )
        sample_index = len(pending_samples)
        lang = _normalize_dataset_lang(_get_first_existing(row, [VQA_LANG_FIELD, "lang", "language"])) or _vqa_lang(sample_index)
        english_question = _normalize_answer_field(
            _get_first_existing(row, [VQA_QUESTION_FIELD, "question", "query", "prompt"]),
            sample_index,
        )
        english_answer = _normalize_answer_field(
            _get_first_existing(row, [VQA_ANSWER_FIELD, "answer", "answers", "response", "label"]),
            sample_index,
        )
        if image is None or not english_question or not english_answer:
            continue

        question = english_question
        answer = english_answer
        if lang == "zh":
            question = _normalize_answer_field(
                _get_first_existing(row, [VQA_ZH_QUESTION_FIELD, "question_zh", "zh_question", "question_cn"]),
                sample_index,
            )
            answer = _normalize_answer_field(
                _get_first_existing(row, [VQA_ZH_ANSWER_FIELD, "answer_zh", "zh_answer", "answer_cn"]),
                sample_index,
            )

        pending_samples.append(
            {
                "image": image,
                "question": question,
                "answer": answer,
                "english_question": english_question,
                "english_answer": english_answer,
                "source": "vqa",
                "lang": lang,
            }
        )

        if len(pending_samples) % 64 == 0:
            print(
                f"collected {len(pending_samples)}/{num_samples} VQA samples "
                f"({_language_counts(pending_samples)})"
            )

    zh_questions_needing_translation = [
        sample for sample in pending_samples if sample["lang"] == "zh" and not sample["question"]
    ]
    if zh_questions_needing_translation:
        print(
            f"translating {len(zh_questions_needing_translation)} VQA questions to Chinese "
            f"with {COCO_TRANSLATION_MODEL_ID}"
        )
        translated_questions = _translate_vqa_texts_to_zh(
            [sample["english_question"] for sample in zh_questions_needing_translation]
        )
        for sample, translated_question in zip(zh_questions_needing_translation, translated_questions):
            sample["question"] = translated_question

    zh_answers_needing_translation = [
        sample for sample in pending_samples if sample["lang"] == "zh" and not sample["answer"]
    ]
    if zh_answers_needing_translation:
        print(
            f"translating {len(zh_answers_needing_translation)} VQA answers to Chinese "
            f"with {COCO_TRANSLATION_MODEL_ID}"
        )
        translated_answers = _translate_vqa_texts_to_zh(
            [sample["english_answer"] for sample in zh_answers_needing_translation]
        )
        for sample, translated_answer in zip(zh_answers_needing_translation, translated_answers):
            sample["answer"] = translated_answer

    calibration_dataset: List[Dict[str, Any]] = []
    for sample in pending_samples:
        if not sample["question"] or not sample["answer"]:
            continue
        calibration_dataset.append(
            {
                "messages": _build_vqa_messages(sample["image"], sample["question"], sample["answer"]),
                "source": sample["source"],
                "lang": sample["lang"],
            }
        )

    return calibration_dataset


def _build_coco_caption_calibration_dataset(num_samples: int) -> List[Dict[str, Any]]:
    if num_samples <= 0:
        return []

    dataset = load_dataset(
        "parquet",
        data_files=_resolve_coco_data_files(COCO_CAPTION_DATA_FILES),
        split="train",
        cache_dir=DATASETS_CACHE_DIR,
    ).shuffle(seed=COCO_CAPTION_SEED)

    pending_samples: List[Dict[str, Any]] = []
    for idx, row in enumerate(dataset):
        if len(pending_samples) >= num_samples:
            break

        image = row.get(COCO_IMAGE_FIELD)
        sample_index = len(pending_samples)
        caption = _get_caption_from_row(row, COCO_EN_ANSWER_FIELD, sample_index)
        question, lang = _coco_prompt(sample_index)
        if image is None or not question:
            continue

        answer = _get_caption_from_row(row, COCO_ZH_ANSWER_FIELD, sample_index) if lang == "zh" else caption
        pending_samples.append(
            {
                "image": image,
                "question": question,
                "answer": answer,
                "english_caption": caption,
                "source": "coco_caption",
                "lang": lang,
            }
        )

        if len(pending_samples) % 64 == 0:
            print(
                f"collected {len(pending_samples)}/{num_samples} COCO-Caption samples "
                f"({_language_counts(pending_samples)})"
            )

    zh_samples_needing_translation = [
        sample for sample in pending_samples if sample["lang"] == "zh" and not sample["answer"]
    ]
    if zh_samples_needing_translation:
        captions_to_translate = [sample["english_caption"] for sample in zh_samples_needing_translation]
        if not all(captions_to_translate):
            raise RuntimeError(
                "Chinese COCO samples need assistant captions, but some rows have no English caption. "
                "Set COCO_EN_ANSWER_FIELD correctly or provide COCO_ZH_ANSWER_FIELD."
            )
        print(
            f"translating {len(captions_to_translate)} COCO captions to Chinese "
            f"with {COCO_TRANSLATION_MODEL_ID}"
        )
        translated_captions = _translate_captions_to_zh(captions_to_translate)
        for sample, translated_caption in zip(zh_samples_needing_translation, translated_captions):
            sample["answer"] = translated_caption

    calibration_dataset: List[Dict[str, Any]] = []
    for sample in pending_samples:
        calibration_dataset.append(
            {
                "messages": _build_coco_messages(sample["image"], sample["question"], sample["answer"]),
                "source": sample["source"],
                "lang": sample["lang"],
            }
        )

    return calibration_dataset


def build_calibration_dataset(num_samples: int) -> List[Dict[str, Any]]:
    language_samples = min(max(NUM_LANGUAGE_CALIB, 0), num_samples)
    vqa_samples = min(max(NUM_VQA_CALIB, 0), num_samples - language_samples)
    coco_samples = min(max(NUM_COCO_CALIB, 0), num_samples - language_samples - vqa_samples)
    text_samples = num_samples - vqa_samples - coco_samples - language_samples
    text_dataset = _build_text_calibration_dataset(text_samples)
    vqa_dataset = _build_vqa_calibration_dataset(vqa_samples)
    coco_dataset = _build_coco_caption_calibration_dataset(coco_samples)
    language_dataset = _build_language_following_dataset(language_samples)
    _release_caption_translation_model()

    calibration_dataset: List[Dict[str, Any]] = []
    text_iter = iter(text_dataset)
    vqa_iter = iter(vqa_dataset)
    coco_iter = iter(coco_dataset)
    language_iter = iter(language_dataset)

    while len(calibration_dataset) < num_samples:
        try:
            calibration_dataset.append(next(vqa_iter))
        except StopIteration:
            pass

        if len(calibration_dataset) >= num_samples:
            break

        try:
            calibration_dataset.append(next(coco_iter))
        except StopIteration:
            pass

        if len(calibration_dataset) >= num_samples:
            break

        try:
            calibration_dataset.append(next(language_iter))
        except StopIteration:
            pass

        if len(calibration_dataset) >= num_samples:
            break

        try:
            calibration_dataset.append(next(text_iter))
        except StopIteration:
            pass

        if len(calibration_dataset) >= len(text_dataset) + len(vqa_dataset) + len(coco_dataset) + len(language_dataset):
            break

    print(
        f"ready calibration mix: {len(vqa_dataset)} VQA samples, "
        f"{len(text_dataset)} text samples, "
        f"{len(coco_dataset)} COCO-Caption samples, "
        f"{len(language_dataset)} language-following samples; "
        f"vqa langs [{_language_counts(vqa_dataset)}], "
        f"text langs [{_language_counts(text_dataset)}], "
        f"coco langs [{_language_counts(coco_dataset)}], "
        f"language langs [{_language_counts(language_dataset)}]"
    )

    no_thinking_count = 0
    if NO_THINKING_CALIB_RATIO > 0:
        for idx, sample in enumerate(calibration_dataset):
            if not _should_use_no_thinking(idx):
                continue
            messages = sample.get("messages")
            if isinstance(messages, list):
                stripped_messages = _strip_assistant_messages(messages)
                if stripped_messages:
                    sample["messages"] = stripped_messages
                    sample["calibration_mode"] = "no_thinking"
                    no_thinking_count += 1

    if no_thinking_count:
        print(
            f"converted {no_thinking_count}/{len(calibration_dataset)} calibration samples "
            f"to no-thinking prompts (ratio={NO_THINKING_CALIB_RATIO:.2f})"
        )
    return calibration_dataset


def _build_or_reuse_calibration_dataset(num_samples: int) -> List[Dict[str, Any]]:
    if _load_calibration_cache() is not None:
        return _build_cached_calibration_placeholder()
    return build_calibration_dataset(num_samples)


def _has_assistant_message(messages: List[Dict[str, Any]]) -> bool:
    return any(message.get("role") == "assistant" for message in messages)


def _strip_assistant_messages(messages: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    return [dict(message) for message in messages if message.get("role") != "assistant"]


def _should_use_no_thinking(sample_index: int) -> bool:
    ratio = max(0.0, min(1.0, NO_THINKING_CALIB_RATIO))
    if ratio <= 0.0:
        return False
    if ratio >= 1.0:
        return True
    bucket = 1000
    threshold = int(round(ratio * bucket))
    hashed = (sample_index * 2654435761) % bucket
    return hashed < threshold


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
        is_no_thinking_sample = example.get("calibration_mode") == "no_thinking"
        add_generation_prompt = (
            True
            if is_no_thinking_sample
            else ADD_GENERATION_PROMPT and not _has_assistant_message(messages)
        )
        enable_thinking = False if is_no_thinking_sample else None

        encoded = processor.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=add_generation_prompt,
            return_dict=True,
            return_tensors="pt",
            **_calibration_chat_template_kwargs(enable_thinking),
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


def main() -> None:
    calibration_dataset = _build_or_reuse_calibration_dataset(NUM_CALIB)
    print(
        f"ready to quantize with {len(calibration_dataset)} mixed calibration samples; "
        f"target_language={TARGET_LANGUAGE}, text_pattern={TEXT_LANGUAGE_PATTERN}, "
        f"vqa_prompt_pattern={VQA_PROMPT_PATTERN}, coco_prompt_pattern={COCO_PROMPT_PATTERN}, "
        f"calibration_enable_thinking={CALIBRATION_ENABLE_THINKING}, "
        f"no_thinking_calib_ratio={NO_THINKING_CALIB_RATIO}"
    )

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
        act_group_aware=False,
    )

    model = GPTQModel.load(MODEL_ID, quant_config, device=MODEL_DEVICE)
    _patch_forward_for_precomputed_inputs(model)
    model.prepare_dataset = MethodType(_prepare_mixed_calibration_dataset, model)
    model.quantize(calibration_dataset, batch_size=BATCH_SIZE)
    model.save(QUANT_PATH)


if __name__ == "__main__":
    main()
