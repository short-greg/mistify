import torch
from .._base import CompositionBase, maxmin, ComplementBase,get_comp_weight_size
from torch import nn

def rand(*size: int, dtype=torch.float32, device='cpu'):

    return (torch.rand(*size, device=device, dtype=dtype)).round()


def negatives(*size: int, dtype=torch.float32, device='cpu'):

    return torch.zeros(*size, dtype=dtype, device=device)


def positives(*size: int, dtype=torch.float32, device='cpu'):

    return torch.ones(*size, dtype=dtype, device=device)
