"""
从 Qwen3 提取各层 token 表示
- 模型冻结，仅前向传播
- 返回每层的 hidden states: (num_layers+1, batch, seq_len, hidden_dim)
"""
import torch
from typing import Optional


def extract_hidden_states(
    model,
    input_ids: torch.Tensor,
    attention_mask: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """
    提取所有层的 hidden states。
    Args:
        model: Qwen3ForCausalLM
        input_ids: (batch, seq_len)
        attention_mask: (batch, seq_len), 可选
    Returns:
        hidden_states: (num_layers+1, batch, seq_len, hidden_dim)
        第 0 层为 embedding，1~N 层为 transformer 层
    """
    model.eval()
    with torch.no_grad():
        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=True,
        )
        # outputs.hidden_states: tuple of (batch, seq_len, hidden_dim), len = num_layers+1
        hidden_states = torch.stack(outputs.hidden_states, dim=0)
    return hidden_states


def get_layer_representations(
    hidden_states: torch.Tensor,
    layer_idx: int,
) -> torch.Tensor:
    """
    取指定层的表示。
    Args:
        hidden_states: (num_layers+1, batch, seq_len, hidden_dim)
        layer_idx: 0-based，0=embedding, 1~N=transformer 层
    Returns:
        (batch, seq_len, hidden_dim)
    """
    return hidden_states[layer_idx]
