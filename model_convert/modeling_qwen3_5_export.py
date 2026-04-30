import os
from typing import Any

import onnxruntime as ort
import torch
import torch.nn.functional as F
from transformers.modeling_outputs import BaseModelOutputWithPooling

try:
    from transformers import Qwen3_5ForConditionalGeneration, Qwen3_5Model, Qwen3_5VisionModel
except ImportError:  # fallback for environments that do not expose these at transformers top-level
    from transformers.models.qwen3_5.modeling_qwen3_5 import (
        Qwen3_5ForConditionalGeneration,
        Qwen3_5Model,
        Qwen3_5VisionModel,
    )


def _torch_load(path: str) -> Any:
    try:
        return torch.load(path, map_location="cpu", weights_only=True)
    except TypeError:
        return torch.load(path, map_location="cpu")


def to_grid_thw_tensor(grid_thw: torch.Tensor | list[int] | tuple[int, int, int]) -> torch.Tensor:
    if isinstance(grid_thw, torch.Tensor):
        grid = grid_thw.detach().clone().to(dtype=torch.long, device="cpu")
    else:
        grid = torch.tensor(grid_thw, dtype=torch.long)

    if grid.numel() == 3:
        grid = grid.reshape(1, 3)
    if grid.ndim != 2 or grid.shape[1] != 3:
        raise ValueError(f"`grid_thw` must have shape (N, 3), got {tuple(grid.shape)}")
    return grid


@torch.no_grad()
def compute_static_vision_tensors(vision_model: Qwen3_5VisionModel, grid_thw: torch.Tensor) -> dict[str, torch.Tensor]:
    grid = to_grid_thw_tensor(grid_thw).to(vision_model.pos_embed.weight.device)
    pos_embeds = vision_model.fast_pos_embed_interpolate(grid)

    rotary_pos_emb = vision_model.rot_pos_emb(grid)
    seq_len = rotary_pos_emb.shape[0]
    rotary_pos_emb = rotary_pos_emb.reshape(seq_len, -1)
    emb = torch.cat((rotary_pos_emb, rotary_pos_emb), dim=-1)
    position_cos = emb.cos()
    position_sin = emb.sin()

    cu_seqlens = torch.repeat_interleave(grid[:, 1] * grid[:, 2], grid[:, 0]).cumsum(dim=0, dtype=torch.int32)
    cu_seqlens = F.pad(cu_seqlens, (1, 0), value=0)

    return {
        "grid_thw": grid.to("cpu"),
        "pos_embeds": pos_embeds.to("cpu", dtype=torch.float32),
        "position_cos": position_cos.to("cpu", dtype=torch.float32),
        "position_sin": position_sin.to("cpu", dtype=torch.float32),
        "cu_seqlens": cu_seqlens.to("cpu", dtype=torch.int32),
    }


@torch.no_grad()
def save_static_vision_tensors(
    model_path: str,
    grid_thw: torch.Tensor | list[int] | tuple[int, int, int],
    output_path: str,
    dtype: torch.dtype = torch.float32,
) -> None:
    model = Qwen3_5ForConditionalGeneration.from_pretrained(
        model_path,
        torch_dtype=dtype,
        device_map="cpu",
    )
    model.eval()
    static_tensors = compute_static_vision_tensors(model.model.visual, to_grid_thw_tensor(grid_thw))
    torch.save(static_tensors, output_path)


class Qwen3_5VisionModelExport(Qwen3_5VisionModel):
    def __init__(self, config, *inputs, static_tensors_path: str | None = None, **kwargs) -> None:
        super().__init__(config, *inputs, **kwargs)
        self.static_pos_embeds = None
        self.static_position_cos = None
        self.static_position_sin = None
        self.static_cu_seqlens = None
        self.static_grid_thw = None
        if static_tensors_path:
            self.load_static_tensors(static_tensors_path)

    def load_static_tensors(self, static_tensors_path: str) -> None:
        if not os.path.exists(static_tensors_path):
            raise FileNotFoundError(f"static tensor file not found: {static_tensors_path}")
        static = _torch_load(static_tensors_path)
        self.static_pos_embeds = static["pos_embeds"].to("cpu")
        self.static_position_cos = static["position_cos"].to("cpu")
        self.static_position_sin = static["position_sin"].to("cpu")
        self.static_cu_seqlens = static["cu_seqlens"].to("cpu")
        self.static_grid_thw = static["grid_thw"].to("cpu")

    def forward_export_nchw(self, hidden_states: torch.Tensor) -> torch.Tensor:
        if self.static_pos_embeds is None:
            raise RuntimeError("static tensors are not loaded. Call `load_static_tensors` before export forward.")
        if hidden_states.ndim != 4:
            raise ValueError(
                "export input must be 4D tensor in layout [t, c, seq, tpp], "
                f"got shape {tuple(hidden_states.shape)}"
            )

        t, c, seq_len, tpp = hidden_states.shape
        hidden_states = hidden_states.permute(0, 2, 1, 3).reshape(t * seq_len, c * tpp)

        hidden_states = self.patch_embed(hidden_states)
        if hidden_states.shape[0] != self.static_pos_embeds.shape[0]:
            raise ValueError(
                f"token length mismatch: export input produces {hidden_states.shape[0]} tokens, "
                f"but static tensors expect {self.static_pos_embeds.shape[0]}"
            )

        hidden_states = hidden_states + self.static_pos_embeds.to(hidden_states.device, hidden_states.dtype)

        position_embeddings = (
            self.static_position_cos.to(hidden_states.device, hidden_states.dtype),
            self.static_position_sin.to(hidden_states.device, hidden_states.dtype),
        )
        cu_seqlens = self.static_cu_seqlens.to(hidden_states.device)

        for blk in self.blocks:
            hidden_states = blk(
                hidden_states,
                cu_seqlens=cu_seqlens,
                position_embeddings=position_embeddings,
            )

        merged_hidden_states = self.merger(hidden_states)
        return merged_hidden_states


class Qwen3_5ModelExport(Qwen3_5Model):
    def __init__(self, config, static_tensors_path: str | None = None):
        super().__init__(config)
        config.vision_config._attn_implementation = "eager"
        self.visual = Qwen3_5VisionModelExport._from_config(
            config.vision_config,
        )
        if static_tensors_path:
            self.visual.load_static_tensors(static_tensors_path)


class Qwen3_5ForConditionalGenerationExport(Qwen3_5ForConditionalGeneration):
    def __init__(self, config, static_tensors_path: str | None = None):
        super().__init__(config)
        self.model = Qwen3_5ModelExport(config, static_tensors_path=static_tensors_path)


class Qwen3_5VisionModelONNX(Qwen3_5VisionModel):
    def __init__(self, config, *inputs, **kwargs) -> None:
        super().__init__(config, *inputs, **kwargs)
        self.session = None

    def init_onnx_session(self, onnx_path: str, providers: list[str] | None = None) -> None:
        providers = providers or ["CPUExecutionProvider"]
        self.session = ort.InferenceSession(onnx_path, providers=providers)

    def forward_onnx_nchw(self, hidden_states: torch.Tensor, grid_thw: torch.Tensor | None = None, **kwargs):
        if self.session is None:
            raise RuntimeError("onnx session is not initialized. Call `init_onnx_session` first.")

        device = hidden_states.device
        inputs = {"hidden_states": hidden_states.to(torch.float32).cpu().numpy()}
        outputs = self.session.run(None, inputs)
        pooled = torch.from_numpy(outputs[0]).to(device=device)

        # `get_image_features` only consumes `pooler_output`, but we keep output type compatible.
        empty_last_hidden = torch.empty((0, self.config.hidden_size), device=device, dtype=pooled.dtype)
        return BaseModelOutputWithPooling(last_hidden_state=empty_last_hidden, pooler_output=pooled)


class Qwen3_5ModelONNX(Qwen3_5Model):
    def __init__(self, config):
        super().__init__(config)
        self.visual = Qwen3_5VisionModelONNX._from_config(config.vision_config)


class Qwen3_5ForConditionalGenerationONNX(Qwen3_5ForConditionalGeneration):
    def __init__(self, config):
        super().__init__(config)
        self.model = Qwen3_5ModelONNX(config)
