# Qwen3.5 模型转换说明

本目录用于把 HuggingFace 格式的 Qwen3.5 多模态模型转换为 AX650 可运行的部署产物。转换链路分为三部分：

1. 可选：使用 `GPTQModel` 对原始 HuggingFace 模型做 GPTQ Int4 量化。
2. 导出并编译 Vision Encoder：PyTorch -> ONNX -> `qwen3_5_vision.axmodel`。
3. 编译 Language Model：HuggingFace/GPTQ 模型 -> 多个 text layer `.axmodel` + post `.axmodel` + bf16 embedding。

> 注意：当前脚本中包含本机绝对路径，换机器或换模型版本时需要先检查 `INPUT_DIR`、`OUTPUT_DIR`、`MODEL_ID`、`QUANT_PATH`、`TRANSFORMERS_SRC` 等路径。

## 目录结构

```text
model_convert/
├── apply_gptq.py                         # GPTQ Int4 量化主脚本
├── apply_gptq.sh                         # GPTQ 量化启动示例
├── export_qwen3_5_vision_onnx.py          # 导出 Qwen3.5 Vision Encoder ONNX
├── modeling_qwen3_5_export.py             # 为导出/ONNX 替换重载 Qwen3.5 模型类
├── preprocess_qwen3_5_export.py           # 导出用图像预处理与 NCHW 输入整理
├── get_image_calib.py                      # 生成 Vision Encoder PTQ 校准数据和 tar 包
├── validate_qwen3_5_vision_onnx.py        # 图像输入下对比 Torch 与 ONNX
├── validate_qwen3_5_video_onnx.py         # 视频帧输入下对比 Torch 与 ONNX
├── compare_qwen3_5_vision_onnx_axmodel.py # 对比 ONNX 与 AXModel 仿真输出
├── build_VE.sh                            # Vision Encoder 的 Pulsar2 build 示例
├── build_llm_0.8b.sh                      # 0.8B Language Model 编译示例
├── build_llm_2b.sh                        # 2B Language Model 编译示例
├── build_llm_4b.sh                        # 4B Language Model 编译示例
├── build_llm_longctx.sh                   # 长上下文、多子图编译示例
├── config.json                            # Vision Encoder Pulsar2 量化/编译配置
├── tools/embed_process.sh                 # embedding 导出与 bf16 转换入口
├── tools/extract_embed.py                 # 从 HF 模型提取 embed_tokens 权重
├── tools/embed-process.py                 # `.npy` embedding 转 float32 `.bin`
└── GPTQ_OPTIMIZATION.md                   # GPTQ 量化效果调优记录
```

## 环境准备

建议使用 Python 3.12 环境：

```bash
conda create -n qwen3_5_convert python=3.12 -y
conda activate qwen3_5_convert
pip install -r requirements.txt
```

导出脚本默认使用本机 Transformers 源码：

```bash
export TRANSFORMERS_SRC=../../transformers/src
```

如果你的 Qwen3.5 实现已经安装在当前 Python 环境里，可以不设置该变量；否则请确保该路径下包含 `transformers.models.qwen3_5`。

Vision Encoder 和 Language Model 编译还需要可用的 `pulsar2` 工具链，并且目标芯片按当前脚本配置为 `AX650`。

## 一、GPTQ Int4 量化（可选）

如果输入模型已经是 GPTQ 量化后的 HuggingFace 目录，可以跳过本步骤。

当前 `apply_gptq.py` 使用 `GPTQModel==6.0.3`，默认量化配置为：

- `bits=4`
- `group_size=128`
- `desc_act=False`
- `static_groups=True`
- `sym=True`
- `mse=2.5`

量化数据由 Wikipedia 文本和 COCO Caption 图文样本混合构成。Wikipedia 默认从 ModelScope 的 `wikimedia/wikipedia` 流式读取，COCO Caption 默认读取本目录下的 `val-00001-of-00013.parquet`。

示例：

```bash
export CUDA_VISIBLE_DEVICES=1
export MODEL_ID=../../Qwen/Qwen3.5-2B/
export TARGET_LANGUAGE=zh
export QUANT_PATH=../../Qwen/Qwen3.5-2B-GPTQ-Int4-ZH
python apply_gptq.py | tee apply_gptq.log
```

常用环境变量：

- `NUM_CALIB`：总校准样本数，默认 `1024`。
- `NUM_COCO_CALIB`：图文样本数；`TARGET_LANGUAGE=zh` 时默认约为总数的 `1/2`，给中文文本/回答 token 留出更多校准比例。
- `TARGET_LANGUAGE`：目标输出语言，默认 `zh`；中文场景会默认使用更多中文 Wikipedia 文本和中文图文 prompt。
- `TEXT_LANGUAGE_PATTERN`：纯文本样本语言轮转，中文默认 `zh,zh,en`。
- `COCO_PROMPT_PATTERN`：图文 prompt 语言轮转，中文默认 `zh,zh,en`。
- `TEXT_AS_ASSISTANT`：是否把纯文本样本拆成 user/assistant 续写格式，默认 `1`，用于让校准覆盖中文输出 token。
- `WIKI_DATA_SOURCE`：Wikipedia 数据源，默认 `modelscope`；可设为 `huggingface` 或 `hf` 切回 HuggingFace。
- `WIKI_DATASET_ID`：Wikipedia 数据集 ID，默认 `wikimedia/wikipedia`。
- `WIKI_EN_CONFIG` / `WIKI_ZH_CONFIG`：Wikipedia 子集配置，默认分别使用 `20231101.en` / `20231101.zh`。
- `MODELSCOPE_DATASET_CACHE_DIR`：ModelScope 数据集缓存目录，默认使用 ModelScope SDK 默认缓存。
- `COCO_CAPTION_DATA_FILES`：COCO Caption parquet 路径，默认 `val-00001-of-00013.parquet`。
- `MODEL_DEVICE`：模型加载设备，默认 `cuda:0`。
- `CALIBRATION_DEVICE`：校准数据设备，默认同 `MODEL_DEVICE`。
- `PRECOMPUTE_INPUTS_DEVICE`：预计算 `inputs_embeds` 和视觉特征的设备，默认同 `CALIBRATION_DEVICE`。显存足够时建议用 `cuda:0`，比 CPU 快很多。
- `RELEASE_PRECOMPUTE_MODULES`：预计算结束后是否把 embedding/vision 模块移回 CPU，默认 `1`，可减少后续量化显存占用。
- `CALIBRATION_CACHE_PATH`：预处理后校准 batch 的缓存路径，例如 `cache/qwen3_5_2b_calib_1024_zh.pt`。第二次运行会跳过 Wikipedia/COCO 采样和视觉特征预计算。
- `STRICT_CALIBRATION_CACHE`：是否校验缓存元数据，默认 `1`；避免复用旧英文校准缓存。
- `OVERWRITE_CALIBRATION_CACHE`：是否强制重建并覆盖缓存，默认 `0`。
- `ENABLE_DYNAMIC_QUANT_CONFIG`：是否启用 `dynamic_quant_config`，默认 `1`；设为 `0`、`false`、`no` 或 `off` 可禁用。
- `SKIP_LINEAR_ATTN_FIRST_N`：跳过前 N 层 linear attention 的动态量化规则，默认 `4`。

反复调量化参数时建议开启缓存：

```bash
export PRECOMPUTE_INPUTS_DEVICE=cuda:0
export CALIBRATION_CACHE_PATH=cache/qwen3_5_2b_calib_1024_zh.pt
python apply_gptq.py | tee apply_gptq.log
```

如果只想先快速验证流程，可临时减小样本数：

```bash
NUM_CALIB=256 NUM_COCO_CALIB=128 python apply_gptq.py
```

更多量化效果排查建议见 `GPTQ_OPTIMIZATION.md`。

## 二、导出 Vision Encoder ONNX

`export_qwen3_5_vision_onnx.py` 会把 Vision Encoder 导出为固定输入形状的 ONNX。通过 `--image-size WIDTH HEIGHT` 指定输入图片尺寸，脚本会自动计算 `grid_thw`，在内存中生成相关静态张量并直接固化到 ONNX。

导出固定 `384×384` 输入的模型：

```bash
python export_qwen3_5_vision_onnx.py \
  --model-path ../../Qwen/Qwen3.5-2B/ \
  --image-size 384 384 \
  --image-path demo.jpeg \
  --onnx-output qwen3_5_vision.onnx
```

常用参数：

- `--model-path`：HuggingFace 模型目录。
- `--image-size WIDTH HEIGHT`：固定输入图片尺寸，宽和高必须是 `patch_size × merge_size` 的整数倍；默认 `384 384`。
- `--image-path`：可选的导出样例图片；脚本会将其缩放到 `--image-size`。
- `--onnx-output`：ONNX 输出路径。
- `--hidden-states`：可选，传入 `.pth` 输入张量，否则使用图片预处理结果或随机张量。

导出完成后脚本会执行 `onnx.checker`、shape inference 和 `onnxsim.simplify`。

## 三、校验 Vision Encoder ONNX

图像输入校验：

```bash
python validate_qwen3_5_vision_onnx.py \
  --model-path ../../Qwen/Qwen3.5-2B/ \
  --onnx-path qwen3_5_vision.onnx \
  --image-path demo.jpeg \
  --image-size 384 384 \
  --prompt "描述这张图片" \
  --max-new-tokens 64
```

视频帧输入校验：

```bash
python validate_qwen3_5_video_onnx.py \
  --model-path ../../Qwen/Qwen3.5-2B/ \
  --onnx-path qwen3_5_vision.onnx \
  --video-frames-dir video-test-03 \
  --image-size 384 384 \
  --prompt "请描述这个视频中的内容"
```

校验脚本会把 Torch 原始视觉分支与 ONNX 替换后的视觉分支进行特征和生成结果对比。若替换 ONNX 后的输出文本明显变化，应优先检查导出图片尺寸、`grid_thw`、预处理 layout 和静态张量是否一致。

## 四、编译 Vision Encoder AXModel

`config.json` 中当前 Vision Encoder 输入配置为：

- 输入 tensor：`hidden_states`
- 输入源格式：`U8` + `NHWC`
- 模型输入处理：`BGR` + `NCHW`
- 校准集：`calib_img/hidden_states.tar`
- 目标硬件：`AX650`

### 1. 生成 PTQ 校准数据集

校准数据的 `--image-size` 必须和导出 ONNX 时完全一致。`get_image_calib.py` 会复用导出路径的图像预处理，把原始图片转换为 Vision Encoder 的未归一化 `hidden_states` patch-grid，并自动打包为 `config.json` 所需的 `calib_img/hidden_states.tar`。

请准备至少 8 张有代表性的图片（场景、亮度、主体尽量多样）；可以直接用视频抽帧作为输入。以下命令生成默认固定网格 `grid_thw=(1, 24, 24)` 的 8 个样本：

```bash
python get_image_calib.py \
  --input-dir /path/to/calibration_images \
  --image-size 384 384 \
  --num-samples 8 \
  --output-dir calib_img
```

需要生成更多样本或使用其他固定图片尺寸时：

```bash
python get_image_calib.py \
  --input-dir /path/to/calibration_images \
  --image-size 512 384 \
  --num-samples 32 \
  --output-dir calib_img
```

`--image-size` 的顺序是 `WIDTH HEIGHT`，宽和高都必须是 `patch_size × merge_size = 32` 的整数倍。脚本按 `grid_thw=(1, HEIGHT/16, WIDTH/16)` 自动计算固定网格，并默认覆盖输出目录中同名的 `h*.jpg`、`hidden_states.tar` 和 `calib_manifest.json`。

脚本会生成 `h0.jpg`、`h1.jpg` 等 patch-grid 文件、`hidden_states.tar` 和记录来源及 shape 的 `calib_manifest.json`。这些 JPEG 是张量的序列化形式，尺寸和视觉内容看起来并不正常；不要再对它们做裁剪、缩放或颜色转换。

对于当前默认的 `--image-size 384 384`、`patch_size=16`、`temporal_patch_size=2`：脚本自动计算 `grid_thw=(1,24,24)`，ONNX 输入 shape 为 `[1, 3, 576, 512]`，生成的校准 JPEG 尺寸为 `512×576`。导出 ONNX 和生成校准数据时必须使用相同的 `--image-size`。同时需要在 `config.json` 中把 `calibration_size` 设为希望使用的样本数（且不大于 tar 包内样本数）。

可检查归档内容：

```bash
tar -tvf calib_img/hidden_states.tar
```

### 2. 执行模型转换

编译命令示例：

```bash
bash build_VE.sh
```

等价展开：

```bash
pulsar2 build --input qwen3_5_2b_vision.onnx \
  --config config.json \
  --output_dir build-output \
  --output_name qwen3_5_2b_vision.axmodel \
  --target_hardware AX650 \
  --compiler.check 0
```

如果换了 ONNX 名称、模型规模或输入 shape，需要同步修改：

- `build_VE.sh` 中的 `--input` 和 `--output_name`。
- `config.json` 中的 `calibration_dataset`。
- 校准图片/压缩包的 shape 和导出 ONNX 的输入 shape。

## 五、对比 ONNX 与 AXModel

Vision Encoder 编译完成后，可用仿真接口对比 ONNX 与 AXModel 输出：

```bash
python compare_qwen3_5_vision_onnx_axmodel.py \
  --model-path ../../Qwen/Qwen3.5-2B/ \
  --onnx-path qwen3_5_vision.onnx \
  --axmodel-paths ../../Qwen3_5.AXERA/ax-llm/Qwen3.5-2B-AX650-1/qwen3_5_vision.axmodel \
  --image-path demo.jpeg \
  --chip AX650
```

脚本会输出 cosine、最大绝对误差、最大相对误差、平均误差等指标。历史样例可参考 `check_axmodel.txt`。

## 六、编译 Language Model

本目录提供了四个主要示例脚本：

- `build_llm_0.8b.sh`：0.8B GPTQ Int4 模型。
- `build_llm_2b.sh`：2B GPTQ Int4 模型。
- `build_llm_4b.sh`：4B GPTQ Int4 模型。
- `build_llm_longctx.sh`：多 prefill、多 decode 的长上下文模型。

运行前请检查脚本里的输入、输出路径和上下文参数：

```bash
INPUT_DIR=../../Qwen/Qwen3.5-2B-GPTQ-Int4-0326
OUTPUT_DIR=../../Qwen3_5.AXERA/ax-llm/Qwen3.5-2B-GPTQ-Int4-0326-AX650-C128-P1152-CTX2048
```

编译入口示例：

```bash
bash build_llm_2b.sh
```

脚本统一使用 `pulsar2 llm_build2`。一个 prefill 总长度为 1152、chunk 为 128、最长上下文为 2048 的示例为：

```bash
export FLOAT_MATMUL_USE_CONV_EU=1
pulsar2 llm_build2 --input_path "$INPUT_DIR" \
  --output_path "$OUTPUT_DIR" \
  --hidden_state_type bf16 \
  --prefill_len 1152 \
  --prefill_step_size 128 \
  --max_context 2048 \
  --chip AX650 \
  --parallel 8
```

核心参数含义：

- `--prefill_len`：模型包需要覆盖的 prefill 总长度。
- `--prefill_step_size`：每个 prefill 子图的 token 数，也就是 chunk 长度，通常使用 64、128、256 等 2 的幂。未指定时，`llm_build2` 会根据 `prefill_len` 自动选择：不超过 512 使用 64，不超过 2048 使用 128，更长使用 256。
- `--max_context`：最大 decode attention 长度，应大于 prefill 总长度；底层最大 `kv_cache_len` 会自动生成成 `max_context - 1`。
- `--decode_step_size`：decode 上下文拆分步长。小于等于 0 表示只生成一个 decode 子图；未指定时，`max_context <= 2048` 使用单子图，更长上下文默认按 2048 拆分。
- `--hidden_state_type bf16`：hidden state 使用 bf16。
- `--chip AX650`：目标芯片。
- `--parallel 8`：并行编译进程数，按机器资源调整。


Qwen3.5 的 Language Model 使用 `qwen3_5_text` 适配，Vision Encoder 在前述章节中单独编译，因此这些脚本不传通用的 `--image_size`；视觉输入尺寸由 ONNX/AXModel 的 `--image-size` 流程控制。

输出目录按 `{原始模型名}-{chip}-C{prefill_step_size}-P{prefill_len}-CTX{max_context}` 命名，例如 `Qwen3.5-0.8B-GPTQ-Int4-AX650-C128-P1280-CTX2048`。

编译完成后，脚本会调用：

```bash
tools/embed_process.sh "$INPUT_DIR" "$OUTPUT_DIR"
```

该脚本依次生成：

- `model.embed_tokens.weight.npy`
- `model.embed_tokens.weight.float32.bin`
- `model.embed_tokens.weight.bfloat16.bin`

部署运行通常使用 bf16 版本的 `model.embed_tokens.weight.bfloat16.bin`。

## 七、输出产物

一个完整 AX650 运行目录通常包含：

```text
Qwen3.5-*-AX650-*/
├── config.json
├── model.embed_tokens.weight.bfloat16.bin
├── qwen3_5_vision.axmodel
├── qwen3_5_text_post.axmodel
├── qwen3_5_text_p128_l0_together.axmodel
├── qwen3_5_text_p128_l1_together.axmodel
├── ...
└── qwen3_5_text_p128_lN_together.axmodel
```

运行侧需要根据模型规模设置：

| 模型规模 | `HIDDEN_SIZE` | `NUM_LAYER` |
| --- | ---: | ---: |
| Qwen3.5-0.8B | 1024 | 24 |
| Qwen3.5-2B | 2048 | 24 |
| Qwen3.5-4B | 2560 | 32 |

## 常见问题

- 路径不一致：大部分脚本是本机实验路径，迁移时优先改路径。
- ONNX 输入 shape 不匹配：导出、校验、校准和部署必须使用一致的 `grid_thw`/图片尺寸。
- 量化后幻觉变多：先用同一 prompt 对比 FP/BF16 与 GPTQ 模型，再参考 `GPTQ_OPTIMIZATION.md` 调整校准数据和敏感层量化策略。
- AXModel 误差偏大：先用 `compare_qwen3_5_vision_onnx_axmodel.py` 确认 Vision Encoder，再检查 `config.json` 的输入 layout、mean/std 和校准集。
