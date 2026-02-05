"""
从 Qwen3 提取各层 token 表示
- 模型冻结，仅前向传播
- 提供 extract_single_layer：仅提取单层，显存友好（长序列可用）
"""
import torch
from typing import Optional


def extract_hidden_states(
    model,
    input_ids: torch.Tensor,
    attention_mask: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """
    提取所有层的 hidden states（显存占用大，长序列慎用）。
    """
    model.eval()
    with torch.no_grad():
        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=True,
        )
        hidden_states = torch.stack(outputs.hidden_states, dim=0)
    return hidden_states


def get_layer_representations(
    hidden_states: torch.Tensor,
    layer_idx: int,
) -> torch.Tensor:
    return hidden_states[layer_idx]


def extract_single_layer(
    model,
    input_ids: torch.Tensor,
    layer_idx: int,
    attention_mask: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """
    仅提取指定层的表示，不保存其他层，显存占用小。
    返回 (batch, seq_len, hidden_dim)。
    适用长序列、显存紧张场景。
    """
    model.eval()
    captured = [None]

    def hook(module, inp, out):
        # out 可能是 tuple (hidden_states, ...) 或单个 tensor
        h = out[0] if isinstance(out, tuple) else out
        captured[0] = h.detach().clone()

    try:
        # Qwen3/Llama 等：model.model 含 embed_tokens 和 layers
        core = getattr(model, "model", model)
        if layer_idx == 0:
            target = getattr(core, "embed_tokens", None)
            if target is None:
                target = getattr(core, "wte", None)  # GPT-2 style
        else:
            layers = getattr(core, "layers", None) or getattr(core, "h", None)
            if layers is None:
                raise AttributeError("无法找到 model.model.layers")
            target = layers[layer_idx - 1]

        if target is None:
            raise AttributeError(f"无法找到 layer {layer_idx}")

        handle = target.register_forward_hook(hook)
        with torch.no_grad():
            model(input_ids=input_ids, attention_mask=attention_mask, output_hidden_states=False)
        handle.remove()

        out = captured[0]
        if out is None:
            raise RuntimeError("Hook 未捕获到输出")
        return out
    except Exception:
        # 回退：使用 output_hidden_states 取单层
        with torch.no_grad():
            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                output_hidden_states=True,
            )
        return outputs.hidden_states[layer_idx]
