from typing import Optional
import torch
from torch import nn


class Concat(nn.Module):
    def __init__(self, dim=0):
        super().__init__()
        self.dim = dim

    def forward(self, input, dim: Optional[int] = None):
        if dim is None:
            dim = self.dim
        return torch.cat(input, dim=dim)
