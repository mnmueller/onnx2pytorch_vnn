import torch
from torch import nn


class Mul(nn.Module):
    def forward(self, input, other):
        return input * other
