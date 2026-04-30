# Qwen3.5 AX650 转换与部署项目

本项目用于将 Qwen3.5 多模态模型转换为爱芯 AX650 平台可运行的模型文件，并配套 `ax-llm` 运行时进行图像/视频理解验证。

整体链路：

```text
HuggingFace Qwen3.5
  ├─ 可选：GPTQ Int4 量化
  ├─ Vision Encoder: PyTorch -> ONNX -> qwen3_5_vision.axmodel
  └─ Language Model: HF/GPTQ -> qwen3_5_text_*.axmodel + qwen3_5_text_post.axmodel + bf16 embedding
      ↓
AX650 板端运行：ax-llm/main 或 OpenAI API Demo
```

## 目录结构

```text
Qwen3_5.AXERA/
├── model_convert/   # 模型量化、ONNX 导出、Pulsar2 编译与精度校验脚本
└── ax-llm/          # AX650 C++/Python 运行时、启动脚本和模型产物目录
```

## 关键组件

### `model_convert/`

负责模型转换，主要包含：

- `apply_gptq.py`：使用 `GPTQModel` 对 HuggingFace 模型做 GPTQ Int4 量化。
- `export_qwen3_5_vision_onnx.py`：导出固定输入网格的 Vision Encoder ONNX。
- `validate_qwen3_5_vision_onnx.py` / `validate_qwen3_5_video_onnx.py`：用 ONNX 替换 Torch 视觉分支后做图像/视频一致性校验。
- `compare_qwen3_5_vision_onnx_axmodel.py`：对比 ONNX 与 AXModel 仿真输出。
- `build_VE.sh`：调用 `pulsar2 build` 编译 Vision Encoder。
- `build_llm_0.8b.sh`、`build_llm_2b.sh`、`build_llm_4b.sh`：调用 `pulsar2 llm_build` 编译 Language Model。
- `tools/embed_process.sh`：提取 `embed_tokens` 并转换为板端使用的 bf16 bin。

详细转换步骤见 `model_convert/README.md`。

### `ax-llm/`

负责 AX650 侧运行，主要包含：

- `src/`：C++ 推理运行时源码。
- `scripts/run_image.sh`：图像理解交互式运行示例。
- `scripts/run_video.sh`：视频理解交互式运行示例。
- `scripts/openai_cli.py`：OpenAI-compatible API 调用示例。
- `scripts/gradio_demo.py`：基于 OpenAI-compatible API 的 Gradio 演示界面。
- `build.sh`：交叉编译 `main`、`main_api` 等可执行文件。
- `Qwen3.5-*AX650*/`：已转换或实验中的模型产物目录。

## 快速流程

### 1. 准备 Python 转换环境

```bash
cd Qwen3_5.AXERA/model_convert
conda create -n qwen3_5_convert python=3.12 -y
conda activate qwen3_5_convert
pip install -r requirements.txt
```

如需使用本地 Transformers 源码：

```bash
export TRANSFORMERS_SRC=../../transformers/src
```

### 2. 可选：量化 HuggingFace 模型

如果已有 GPTQ 模型目录，可跳过。

```bash
cd Qwen3_5.AXERA/model_convert
export MODEL_ID=../../Qwen/Qwen3.5-2B/
export QUANT_PATH=../../Qwen/Qwen3.5-2B-GPTQ-Int4-EN
python apply_gptq.py | tee apply_gptq.log
```

### 3. 导出 Vision Encoder ONNX

```bash
python export_qwen3_5_vision_onnx.py \
  --model-path ../../Qwen/Qwen3.5-2B/ \
  --image-path demo.jpeg \
  --onnx-output qwen3_5_vision.onnx \
  --static-tensors-output vision_static_tensors.pth
```

如果部署输入尺寸固定，也可以直接指定 `grid_thw`，例如单张 `384x384` 图片常见为：

```bash
python export_qwen3_5_vision_onnx.py \
  --model-path ../../Qwen/Qwen3.5-2B/ \
  --grid-thw 1 24 24 \
  --onnx-output qwen3_5_vision.onnx
```

### 4. 编译 Vision Encoder

```bash
cd Qwen3_5.AXERA/model_convert
bash build_VE.sh
```

编译产物默认输出到 `model_convert/build-output/`，之后需要复制或重命名为运行目录中的 `qwen3_5_vision.axmodel`。

### 5. 编译 Language Model

按模型规模选择脚本，运行前先检查脚本里的 `INPUT_DIR`、`OUTPUT_DIR`、量化 scale 文件路径和 prefill 配置。

```bash
cd Qwen3_5.AXERA/model_convert
bash build_llm_0.8b.sh
# 或
bash build_llm_2b.sh
# 或
bash build_llm_4b.sh
```

输出目录会包含 text layer `.axmodel`、post `.axmodel`、`config.json` 和 `model.embed_tokens.weight.bfloat16.bin`。

### 6. 构建 AX650 运行时

```bash
cd Qwen3_5.AXERA/ax-llm
./build.sh
```

`build.sh` 会准备 BSP、交叉编译工具链和 OpenCV，并生成 `build_650/install/bin/main`、`main_api` 以及运行脚本。若网络下载不可用，需要提前把依赖包放到脚本期望的位置。

### 7. 运行图像/视频理解

先确认 `ax-llm/scripts/run_image.sh` 或 `ax-llm/scripts/run_video.sh` 中的变量与实际模型目录匹配：

```bash
AXMODEL_DIR=Qwen3.5-0.8B-GPTQ-Int4-AX650-C128-P1280-CTX2047-0430-EN
HIDDEN_SIZE=1024
NUM_LAYER=24
```

常用模型规模配置：

| 模型规模 | `HIDDEN_SIZE` | `NUM_LAYER` |
| --- | ---: | ---: |
| Qwen3.5-0.8B | 1024 | 24 |
| Qwen3.5-2B | 2048 | 24 |
| Qwen3.5-4B | 2560 | 32 |

运行示例：

```bash
cd Qwen3_5.AXERA/ax-llm
bash scripts/run_image.sh
bash scripts/run_video.sh
```

## 模型产物目录约定

一个完整的 AX650 模型目录通常包含：

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

其中：

- `qwen3_5_vision.axmodel`：Vision Encoder。
- `qwen3_5_text_p128_l%d_together.axmodel`：Language Model 每层模型，`%d` 对应层号。
- `qwen3_5_text_post.axmodel`：LM 后处理/输出头相关模型。
- `model.embed_tokens.weight.bfloat16.bin`：token embedding 权重，运行时通常使用 mmap 加载。
- `config.json`：模型结构、hidden size、token id、vision config 等运行所需元信息。

## 精度校验建议

转换后建议按以下顺序确认问题边界：

1. 使用 `model_convert/validate_qwen3_5_vision_onnx.py` 对比 Torch Vision 与 ONNX Vision。
2. 使用 `model_convert/compare_qwen3_5_vision_onnx_axmodel.py` 对比 ONNX 与 Vision AXModel。
3. 板端运行 `run_image.sh`，先使用固定图片和固定 prompt 做回归。
4. 如果 GPTQ 模型出现明显幻觉，参考 `model_convert/GPTQ_OPTIMIZATION.md` 对比 FP/BF16 原模型、调整校准数据和敏感层量化策略。

## 注意事项

- 当前仓库包含大量实验目录，脚本默认值不一定对应你要部署的最终目录，运行前务必检查路径。
- Vision Encoder 是固定 shape 导出，导出、校准、Pulsar2 编译和板端输入尺寸必须一致。
- `pulsar2 llm_build` 的 `last_kv_cache_len` 最大值决定 prefill 阶段最大 token 数，需要和运行侧预期保持一致。
- 0.8B、2B、4B 的 `HIDDEN_SIZE` 和 `NUM_LAYER` 不同，运行脚本中必须同步修改。
- 生成的 `.axmodel` 和 embedding 文件体积较大，不建议提交到 Git 历史中。
