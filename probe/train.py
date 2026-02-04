"""
探针训练与评估
- 仅更新探针参数，目标模型冻结
- 支持每层独立训练
- 对 -100 标签的 token 忽略损失
"""
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from typing import List, Optional, Tuple
import numpy as np

from .probe_model import LinearProbe
from .data_loader import ProbeSample, NUM_LABELS


class ProbeDataset(Dataset):
    """(hidden_reprs, labels) 数据集，用于探针训练"""

    def __init__(self, hidden_reprs: torch.Tensor, labels: torch.Tensor):
        # hidden_reprs: (total_tokens, hidden_dim)
        # labels: (total_tokens,), -100 表示忽略
        self.hidden_reprs = hidden_reprs
        self.labels = labels
        # 只保留有效标签的索引
        self.valid_mask = labels != -100
        self.valid_reprs = hidden_reprs[self.valid_mask]
        self.valid_labels = labels[self.valid_mask]

    def __len__(self):
        return len(self.valid_labels)

    def __getitem__(self, idx):
        return self.valid_reprs[idx], self.valid_labels[idx]


def extract_and_stack(
    model,
    samples: List[ProbeSample],
    layer_idx: int,
    device: torch.device,
    batch_size: int = 32,
    normalize: bool = True,
    pad_token_id: int = 0,
    norm_mean: Optional[torch.Tensor] = None,
    norm_std: Optional[torch.Tensor] = None,
) -> Tuple[torch.Tensor, torch.Tensor, Optional[Tuple[torch.Tensor, torch.Tensor]]]:
    """
    对所有样本提取指定层的表示，并堆叠为 (N, hidden_dim) 和 (N,) 标签。
    同时做 Z-score 归一化（可选）。
    """
    from .extractor import extract_hidden_states, get_layer_representations

    all_reprs = []
    all_labels = []

    model.eval()
    with torch.no_grad():
        for i in range(0, len(samples), batch_size):
            batch_samples = samples[i : i + batch_size]
            max_len = max(len(s.input_ids) for s in batch_samples)

            padded_ids = []
            padded_mask = []
            padded_labels = []

            for s in batch_samples:
                pad_len = max_len - len(s.input_ids)
                padded_ids.append(s.input_ids + [pad_token_id] * pad_len)
                padded_mask.append(s.attention_mask + [0] * pad_len)
                # 填充位置标签为 -100
                padded_labels.append(s.token_labels + [-100] * pad_len)

            input_ids = torch.tensor(padded_ids, dtype=torch.long, device=device)
            attention_mask = torch.tensor(padded_mask, dtype=torch.long, device=device)

            hidden_states = extract_hidden_states(model, input_ids, attention_mask)
            layer_repr = get_layer_representations(hidden_states, layer_idx)
            # layer_repr: (batch, seq_len, hidden_dim)

            labels_batch = torch.tensor(padded_labels, dtype=torch.long, device=device)
            for j in range(len(batch_samples)):
                valid = labels_batch[j] != -100
                if valid.any():
                    all_reprs.append(layer_repr[j][valid])
                    all_labels.append(labels_batch[j][valid])

    if not all_reprs:
        hidden_dim = model.config.hidden_size
        return torch.zeros(0, hidden_dim, device=device), torch.zeros(0, dtype=torch.long, device=device), None

    reprs = torch.cat(all_reprs, dim=0)
    labels = torch.cat(all_labels, dim=0)
    norm_stats = None

    if normalize and len(reprs) > 0:
        if norm_mean is not None and norm_std is not None:
            mean, std = norm_mean.to(device), norm_std.to(device)
        else:
            mean = reprs.float().mean(dim=0)
            std = reprs.float().std(dim=0) + 1e-8
            norm_stats = (mean.detach().cpu(), std.detach().cpu())
        reprs = (reprs.float() - mean) / std

    return reprs, labels, norm_stats


def train_probe(
    probe: LinearProbe,
    train_reprs: torch.Tensor,
    train_labels: torch.Tensor,
    device: torch.device,
    epochs: int = 50,
    lr: float = 1e-3,
    batch_size: int = 256,
) -> List[float]:
    """训练探针，返回每轮 loss 列表"""
    dataset = ProbeDataset(train_reprs, train_labels)
    if len(dataset) == 0:
        return []

    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    optimizer = torch.optim.AdamW(probe.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()

    probe.train()
    losses = []
    for epoch in range(epochs):
        total_loss = 0.0
        for reprs_batch, labels_batch in loader:
            reprs_batch = reprs_batch.to(device)
            labels_batch = labels_batch.to(device)

            optimizer.zero_grad()
            logits = probe(reprs_batch)
            loss = criterion(logits, labels_batch)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        avg_loss = total_loss / len(loader)
        losses.append(avg_loss)
        if (epoch + 1) % 10 == 0 or epoch == 0:
            print(f"  Epoch {epoch + 1}/{epochs}, Loss: {avg_loss:.4f}")

    return losses


def evaluate_probe(
    probe: LinearProbe,
    reprs: torch.Tensor,
    labels: torch.Tensor,
    device: torch.device,
) -> float:
    """计算准确率（只统计有效标签）"""
    valid = labels != -100
    if not valid.any():
        return 0.0

    probe.eval()
    with torch.no_grad():
        reprs_valid = reprs[valid].to(device)
        labels_valid = labels[valid].to(device)
        logits = probe(reprs_valid)
        preds = logits.argmax(dim=1)
        correct = (preds == labels_valid).sum().item()
        total = labels_valid.size(0)
    return correct / total if total > 0 else 0.0
