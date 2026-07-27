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
└── ax-llm/          # AX650 C++/Python 运行时、启动脚本和模型产物目录。没有提交到本仓库，源码在 https://github.com/AXERA-TECH/ax-llm  
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
- `build.sh`：交叉编译 `main`、`main_api` 等可执行文件。

