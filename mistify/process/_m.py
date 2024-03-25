# 3rd party
from torch import nn
import torch

# local
from .._functional import binarize, signify, clamp

class Argmax(nn.Module):

    def __init__(self, dim=-1):
        super().__init__()
        self._dim = dim

    def forward(self, x: torch.Tensor) -> torch.LongTensor:
        return torch.argmax(x, dim=-1)


class Sign(nn.Module):

    def __init__(self, grad: bool = True):
        super().__init__()
        self._grad = grad

    def forward(self, x: torch.Tensor):
        return signify(x)


class Boolean(nn.Module):

    def __init__(self, grad: bool = True):
        super().__init__()
        self._grad = grad

    def forward(self, x: torch.Tensor):
        return binarize(x)


class Clamp(nn.Module):

    def __init__(self, lower: float=-1.0, upper: float=1.0, grad: bool = True):
        super().__init__()
        self._lower = lower
        self._upper = upper
        self._grad = grad

    def forward(self, x: torch.Tensor):
        return clamp(x)
