"""
线性探针：输入 hidden_dim，输出 num_labels
"""
import torch.nn as nn
from typing import Optional


class LinearProbe(nn.Module):
    def __init__(self, input_dim: int, num_labels: int):
        super().__init__()
        self.linear = nn.Linear(input_dim, num_labels)

    def forward(self, x):
        return self.linear(x)
