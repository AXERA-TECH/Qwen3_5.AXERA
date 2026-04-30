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
├── validate_qwen3_5_vision_onnx.py        # 图像输入下对比 Torch 与 ONNX
├── validate_qwen3_5_video_onnx.py         # 视频帧输入下对比 Torch 与 ONNX
├── compare_qwen3_5_vision_onnx_axmodel.py # 对比 ONNX 与 AXModel 仿真输出
├── build_VE.sh                            # Vision Encoder 的 Pulsar2 build 示例
├── build_llm_0.8b.sh                      # 0.8B Language Model 编译示例
├── build_llm_2b.sh                        # 2B Language Model 编译示例
├── build_llm_4b.sh                        # 4B Language Model 编译示例
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

量化数据由 Wikipedia 文本和 COCO Caption 图文样本混合构成，默认会读取本目录下的 `val-00001-of-00013.parquet`。

示例：

```bash
export CUDA_VISIBLE_DEVICES=1
export MODEL_ID=../../Qwen/Qwen3.5-2B/
export QUANT_PATH=../../Qwen/Qwen3.5-2B-GPTQ-Int4-EN
python apply_gptq.py | tee apply_gptq.log
```

常用环境变量：

- `NUM_CALIB`：总校准样本数，默认 `1024`。
- `NUM_COCO_CALIB`：图文样本数，默认约为总数的 `3/4`。
- `COCO_CAPTION_DATA_FILES`：COCO Caption parquet 路径，默认 `val-00001-of-00013.parquet`。
- `MODEL_DEVICE`：模型加载设备，默认 `cuda:0`。
- `CALIBRATION_DEVICE`：校准数据设备，默认同 `MODEL_DEVICE`。
- `PRECOMPUTE_INPUTS_DEVICE`：预计算 `inputs_embeds` 和视觉特征的设备，默认同 `CALIBRATION_DEVICE`。显存足够时建议用 `cuda:0`，比 CPU 快很多。
- `RELEASE_PRECOMPUTE_MODULES`：预计算结束后是否把 embedding/vision 模块移回 CPU，默认 `1`，可减少后续量化显存占用。
- `CALIBRATION_CACHE_PATH`：预处理后校准 batch 的缓存路径，例如 `cache/qwen3_5_2b_calib_1024.pt`。第二次运行会跳过 Wikipedia/COCO 采样和视觉特征预计算。
- `ENABLE_DYNAMIC_QUANT_CONFIG`：是否启用 `dynamic_quant_config`，默认 `1`；设为 `0`、`false`、`no` 或 `off` 可禁用。
- `SKIP_LINEAR_ATTN_FIRST_N`：跳过前 N 层 linear attention 的动态量化规则，默认 `3`。

反复调量化参数时建议开启缓存：

```bash
export PRECOMPUTE_INPUTS_DEVICE=cuda:0
export CALIBRATION_CACHE_PATH=cache/qwen3_5_2b_calib_1024.pt
python apply_gptq.py | tee apply_gptq.log
```

如果只想先快速验证流程，可临时减小样本数：

```bash
NUM_CALIB=256 NUM_COCO_CALIB=192 python apply_gptq.py
```

更多量化效果排查建议见 `GPTQ_OPTIMIZATION.md`。

## 二、导出 Vision Encoder ONNX

`export_qwen3_5_vision_onnx.py` 会把 Vision Encoder 导出为固定输入形状的 ONNX。导出时会把和 `grid_thw` 相关的静态张量提前保存到 `vision_static_tensors.pth`，并重载视觉模型 forward 以适配 NCHW 输入。

使用图片自动计算 `grid_thw`：

```bash
python export_qwen3_5_vision_onnx.py \
  --model-path ../../Qwen/Qwen3.5-2B/ \
  --image-path demo.jpeg \
  --onnx-output qwen3_5_vision.onnx \
  --static-tensors-output vision_static_tensors.pth
```

或者手动指定固定网格：

```bash
python export_qwen3_5_vision_onnx.py \
  --model-path ../../Qwen/Qwen3.5-2B/ \
  --grid-thw 1 24 24 \
  --onnx-output qwen3_5_vision.onnx \
  --static-tensors-output vision_static_tensors.pth
```

常用参数：

- `--model-path`：HuggingFace 模型目录。
- `--image-path`：用于自动计算 `grid_thw` 和导出输入 shape 的图片。
- `--grid-thw T H W`：手动指定固定导出网格。
- `--onnx-output`：ONNX 输出路径。
- `--static-tensors-output`：静态张量输出路径，视频 ONNX 校验也会用到。
- `--hidden-states`：可选，传入 `.pth` 输入张量，否则使用图片预处理结果或随机张量。

导出完成后脚本会执行 `onnx.checker`、shape inference 和 `onnxsim.simplify`。

## 三、校验 Vision Encoder ONNX

图像输入校验：

```bash
python validate_qwen3_5_vision_onnx.py \
  --model-path ../../Qwen/Qwen3.5-2B/ \
  --onnx-path qwen3_5_vision.onnx \
  --image-path demo.jpeg \
  --prompt "描述这张图片" \
  --max-new-tokens 64
```

视频帧输入校验：

```bash
python validate_qwen3_5_video_onnx.py \
  --model-path ../../Qwen/Qwen3.5-2B/ \
  --onnx-path qwen3_5_vision.onnx \
  --video-frames-dir video-test-03 \
  --static-tensors-path vision_static_tensors.pth \
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

本目录提供了三个示例脚本：

- `build_llm_0.8b.sh`：0.8B GPTQ Int4 模型。
- `build_llm_2b.sh`：2B GPTQ Int4 模型。
- `build_llm_4b.sh`：4B GPTQ Int4 模型。

运行前请检查脚本里的路径和量化 scale 文件：

```bash
INPUT_DIR=../../Qwen/Qwen3.5-2B-GPTQ-Int4-0326
OUTPUT_DIR=../../Qwen3_5.AXERA/ax-llm/Qwen3.5-2B-AX650-GPTQ-Int4-C128-P1152-CTX2047-0326
```

编译入口示例：

```bash
bash build_llm_2b.sh
```

核心 `pulsar2 llm_build` 参数含义：

- `--kv_cache_len 2047`：KV cache 最大长度。
- `--hidden_state_type bf16`：hidden state 使用 bf16。
- `--prefill_len 128`：基础 prefill 分块长度。
- `--last_kv_cache_len ...`：生成多个 prefill 分组，最大值决定单次 prefill 支持的最大 token 数。
- `--chip AX650`：目标芯片。
- `--parallel 8`：并行编译进程数，按机器资源调整。
- `--linear_conv_scale_mode use` / `--linear_conv_scale_file`：使用已生成的 linear conv scale 文件。
- `--linear_attn_chunk_size 32`：linear attention 编译分块大小。

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
